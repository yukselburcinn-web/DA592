"""
Knowledge graph construction (Graph-RAG substrate).
"""
import json
import math
from pathlib import Path

import networkx as nx
import pandas as pd
from neo4j import GraphDatabase

HERE = Path(__file__).parent
DATA_DIR = HERE.parent / "data"

CATEGORY_AFFINITY = {
    "Culture Enthusiast": {"museum": 1.0, "landmark": 0.9, "history": 0.9, "religion": 0.6, "culture": 0.8},
    "Beach & Relax": {"beach": 1.0, "nature": 0.8, "food": 0.5, "culture": 0.3},
    "Budget Backpacker": {"nightlife": 0.9, "culture": 0.6, "shopping": 0.3, "food": 0.7, "nature": 0.6},
    "Luxury Traveler": {"culture": 0.7, "museum": 0.6, "shopping": 0.9, "food": 0.8, "landmark": 0.7},
    "Nightlife Seeker": {"nightlife": 1.0, "food": 0.6, "culture": 0.3},
    "Nature & Adventure": {"nature": 1.0, "beach": 0.6, "history": 0.3},
    "Family Traveler": {"nature": 0.8, "culture": 0.6, "museum": 0.6, "landmark": 0.6, "food": 0.5},
}



# How much of a POI's score survives being completely unknown. At 0.0 affinity
# and prominence multiply outright, and the least-known member of the favourite
# category scores zero -- which over-mixes: a Nightlife Seeker's top-weighted
# `nightlife` (1.0) loses to `food` (0.6) for every venue below the 60th
# percentile of fame, so the answer drifts off the category the traveler asked
# about. At 1.0 prominence stops mattering and the lexicographic starvation
# this function exists to remove comes straight back.
#
# Swept against both objectives at once -- iconic coverage@24 over 14 city x
# archetype cells, and recall_at_k on the eleven queries graded independently
# of the retriever's own traversal (`dependence_level`, #48). Baseline before
# any of this was 0.193 / 0.126:
#
#     floor   iconic_cov   recall@k   recall (independent)
#      0.00      0.707       0.218          0.098   <- recall regresses
#      0.25      0.607       0.223          0.115
#      0.40      0.557       0.230          0.138   <- chosen
#      0.50      0.450       0.230          0.150
#      0.75      0.286       0.228          0.156
#
# 0.40 is the largest relevance gain that still improves *every* other measure
# against the baseline rather than trading one off. Below it the independently
# graded recall falls under what the old ordering achieved, which would mean
# buying famous suggestions with worse ones -- not a trade worth making
# quietly, and not one this issue needs to make at all.
PROMINENCE_FLOOR = 0.40


def score_by_affinity_and_prominence(pois: list[dict]) -> list[dict]:
    """Annotate each POI with `affinity_prominence` = archetype weight x how
    well-known the place is, and return the same list.

    This replaces a lexicographic `(weight, popularity_score)` ordering, which
    ranked *every* member of the top-weighted category above *every* member of
    the next one. That is fine when a category holds a handful of POIs and
    ruinous at catalogue scale: Culture Enthusiast weights `museum` at 1.0 and
    `landmark` at 0.9, and Paris holds 57 museums -- so ranks 1..57 were all
    museums and the Eiffel Tower, the single most popular POI in the entire
    catalogue (`popularity_score` 5.00), sat at rank 58, behind Paris's 57th
    best museum. Retrieval asks for 48 candidates, so it was cut. Notre-Dame
    (`religion`, 0.6) did not appear in the first 200 at all (#63).

    Multiplying instead lets a very famous place in a slightly less preferred
    category outrank an obscure one in the favourite category, while keeping
    the favourite category on top where prominence is equal -- the Louvre
    (museum, 1.0 x 0.997) still leads the Eiffel Tower (landmark, 0.9 x 1.0).

    `popularity_score` is min-max normalised over `pois` rather than used raw,
    because its floor is 2.67 and not 0: as a raw multiplier its 1.9x spread
    would be swamped by weight's 3.3x, and the starvation would survive in
    milder form. Normalising makes the two factors comparable. A degenerate
    range (one POI, or all equally popular) leaves prominence at 1.0 so
    ordering falls back to weight alone.

    PROMINENCE_FLOOR keeps prominence a modifier rather than a co-equal factor
    -- see the constant.
    """
    if not pois:
        return pois
    scores = [p["popularity_score"] for p in pois]
    lo, hi = min(scores), max(scores)
    span = hi - lo
    for poi in pois:
        prominence = (poi["popularity_score"] - lo) / span if span else 1.0
        poi["affinity_prominence"] = poi["weight"] * (
            PROMINENCE_FLOOR + (1.0 - PROMINENCE_FLOOR) * prominence)
    return pois


def rank_preferred(pois: list[dict], top_k: int) -> list[dict]:
    """Rank an archetype's preferred POIs so every category it asks for is
    represented, in proportion to how much it is asked for (issue #113).

    `score_by_affinity_and_prominence` already fixed the crude version of this
    -- see its docstring -- but it fixed it *within* a single ranking, and one
    ranking cut at `top_k` still starves the weakest category whenever the
    stronger ones hold enough POIs to fill the cut. Measured on the shipped
    settings: a Culture Enthusiast in Paris ranks 241 preferred POIs, and the
    first `religion` (weight 0.6, the lowest that archetype asks for) sat at
    **rank 132** -- far beyond the 72 a three-day trip retrieves. The top 72
    was 34 landmarks, 26 museums, 8 culture and 4 history, and none of Paris's
    churches, so 83 of the catalogue's 84 religion POIs could not reach any
    traveler at all.

    So the categories are merged rather than pooled. Each category keeps its
    own ranking, best-known first, and emits its i-th POI at position
    `i / weight`: a weight-1.0 category emits at 1, 2, 3..., a weight-0.6 one
    at 1.67, 3.33, 5.00... The head of the list still belongs to what the
    traveler most wants -- the first three slots are the three strongest
    categories -- and each category ends up with roughly its share of the cut.
    Prominence still decides who leads *inside* a category, so the ordering the
    prominence floor was swept for is unchanged there.
    """
    if not pois:
        return pois
    score_by_affinity_and_prominence(pois)
    by_category: dict[str, list[dict]] = {}
    for poi in pois:
        by_category.setdefault(poi.get("category"), []).append(poi)

    merged = []
    for category, group in by_category.items():
        group.sort(key=lambda p: (p["affinity_prominence"], p["popularity_score"]),
                   reverse=True)
        for i, poi in enumerate(group):
            weight = poi["weight"] or 1e-9
            # Ties resolve towards the more-wanted category, then the
            # better-known POI, so the order stays deterministic.
            merged.append(((i + 1) / weight, -weight, -poi["affinity_prominence"], poi))
    merged.sort(key=lambda row: row[:3])
    return [row[3] for row in merged[:top_k]]


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))

def build_graph() -> nx.MultiDiGraph:
    destinations = pd.read_csv(DATA_DIR / "destinations.csv")
    pois = pd.read_csv(DATA_DIR / "poi.csv")
    transport = pd.read_csv(DATA_DIR / "transport.csv")

    g = nx.MultiDiGraph()

    for _, d in destinations.iterrows():
        g.add_node(
            d.destination_id, type="City", name=d.city, country=d.country,
            lat=d.lat, lon=d.lon, budget_level=int(d.budget_level),
            tags=json.loads(d.tags),
        )

    for _, p in pois.iterrows():
        g.add_node(
            p.poi_id, type="POI", name=p["name"], category=p.category,
            lat=p.lat, lon=p.lon, avg_visit_minutes=int(p.avg_visit_minutes),
            price_level=int(p.price_level), popularity_score=float(p.popularity_score),
            description=p.description, destination_id=p.destination_id,
            open_hour=int(p.open_hour), close_hour=int(p.close_hour),
            # The pair above cannot say "shut on Mondays"; the tag it was
            # squeezed out of can, and the router reads it when it knows what
            # day it is (issue #70). Carried as text, empty where OSM never
            # described the place -- GML has no null. pd.isna, not `or ""`:
            # a missing cell reads back as NaN, which is truthy, so `or ""`
            # would write the string "nan" into every untagged node.
            opening_hours_raw=("" if pd.isna(p.get("opening_hours_raw"))
                               else str(p["opening_hours_raw"]).strip()),
        )
        g.add_edge(p.destination_id, p.poi_id, relation="HAS_POI")

    for _, t in transport.iterrows():
        g.add_node(
            t.transport_id, type="Transport", name=t["name"], ttype=t.type,
            lat=t.lat, lon=t.lon, destination_id=t.destination_id,
        )
        g.add_edge(t.destination_id, t.transport_id, relation="HAS_TRANSPORT")

    pois_by_dest = pois.groupby("destination_id")
    for dest_id, group in pois_by_dest:
        recs = group.to_dict("records")
        for i, a in enumerate(recs):
            for b in recs[i + 1:]:
                dist = haversine_km(a["lat"], a["lon"], b["lat"], b["lon"])
                if dist <= 2.0:
                    g.add_edge(a["poi_id"], b["poi_id"], relation="NEAR", distance_km=round(dist, 3))
                    g.add_edge(b["poi_id"], a["poi_id"], relation="NEAR", distance_km=round(dist, 3))
                if a["category"] == b["category"]:
                    g.add_edge(a["poi_id"], b["poi_id"], relation="SAME_CATEGORY")
                    g.add_edge(b["poi_id"], a["poi_id"], relation="SAME_CATEGORY")

    for archetype, affinities in CATEGORY_AFFINITY.items():
        node_id = f"ARCH::{archetype}"
        g.add_node(node_id, type="ArchetypeProfile", name=archetype, affinities=affinities)
        for category, weight in affinities.items():
            matches = pois[pois.category == category]["poi_id"].tolist()
            for poi_id in matches:
                g.add_edge(node_id, poi_id, relation="PREFERS", weight=weight)

    return g


class GraphIndex:
    """Query surface over the knowledge graph supporting NetworkX and Neo4j backends."""

    def __init__(self, graph: nx.MultiDiGraph = None, backend="networkx", neo4j_uri="bolt://localhost:7687", neo4j_auth=("neo4j", "password")):
        self.backend = backend
        if self.backend == "neo4j":
            self.driver = GraphDatabase.driver(neo4j_uri, auth=neo4j_auth)
        else:
            self.g = graph if graph is not None else build_graph()

    def close(self):
        if self.backend == "neo4j":
            self.driver.close()

    def city_pois(self, destination_id: str, category: str = None):
        if self.backend == "neo4j":
            query = "MATCH (c:City {id: $dest_id})-[:HAS_POI]->(p:POI) "
            if category:
                query += "WHERE p.category = $category "
            query += "RETURN p"
            with self.driver.session() as session:
                result = session.run(query, dest_id=destination_id, category=category)
                return [{"poi_id": record["p"].element_id, **record["p"]} for record in result]
        
        out = []
        for _, poi_id, data in self.g.out_edges(destination_id, data=True):
            if data.get("relation") == "HAS_POI":
                node = self.g.nodes[poi_id]
                if category is None or node.get("category") == category:
                    out.append({"poi_id": poi_id, **node})
        return out

    def city_transport(self, destination_id: str):
        if self.backend == "neo4j":
            query = "MATCH (c:City {id: $dest_id})-[:HAS_TRANSPORT]->(t:Transport) RETURN t"
            with self.driver.session() as session:
                result = session.run(query, dest_id=destination_id)
                return [{"transport_id": record["t"].element_id, **record["t"]} for record in result]

        out = []
        for _, tid, data in self.g.out_edges(destination_id, data=True):
            if data.get("relation") == "HAS_TRANSPORT":
                out.append({"transport_id": tid, **self.g.nodes[tid]})
        return out

    def nearby_pois(self, poi_id: str, max_km: float = 2.0):
        if self.backend == "neo4j":
            query = "MATCH (p1:POI {id: $poi_id})-[r:NEAR]->(p2:POI) WHERE r.distance_km <= $max_km RETURN p2, r.distance_km AS dist ORDER BY dist"
            with self.driver.session() as session:
                result = session.run(query, poi_id=poi_id, max_km=max_km)
                return [{"poi_id": record["p2"].element_id, "distance_km": record["dist"], **record["p2"]} for record in result]

        out = []
        for _, other_id, data in self.g.out_edges(poi_id, data=True):
            if data.get("relation") == "NEAR" and data.get("distance_km", 999) <= max_km:
                out.append({"poi_id": other_id, "distance_km": data["distance_km"], **self.g.nodes[other_id]})
        return sorted(out, key=lambda x: x["distance_km"])

    def archetype_preferred_pois(self, archetype: str, destination_id: str = None, top_k: int = 20):
        """The POIs this archetype prefers, ranked so every category it asks
        for is actually represented (issue #113).

        Both backends now fetch the rows and rank them here, in one place. The
        Cypher query used to do its own ordering and `LIMIT`, which meant the
        two backends could drift apart on the thing that decides what a
        traveler is shown.
        """
        if self.backend == "neo4j":
            query = "MATCH (a:ArchetypeProfile {name: $arch})-[r:PREFERS]->(p:POI) "
            if destination_id:
                query += "WHERE p.destination_id = $dest_id "
            query += "RETURN p, r.weight AS weight"
            with self.driver.session() as session:
                result = session.run(query, arch=archetype, dest_id=destination_id)
                out = [{"poi_id": record["p"].element_id, "weight": record["weight"],
                        **record["p"]} for record in result]
            return rank_preferred(out, top_k)

        node_id = f"ARCH::{archetype}"
        if node_id not in self.g:
            return []
        out = []
        for _, poi_id, data in self.g.out_edges(node_id, data=True):
            if data.get("relation") != "PREFERS":
                continue
            node = self.g.nodes[poi_id]
            if destination_id and node.get("destination_id") != destination_id:
                continue
            out.append({"poi_id": poi_id, "weight": data["weight"], **node})
        return rank_preferred(out, top_k)

    # 1.0 km, not the 3.0 this used to default to. The radius has to be small
    # enough that "near a transport hub" separates some POIs from the rest, and
    # 3.0 no longer did: against a 7 km-radius catalogue with real interchanges
    # in it, 96% of Paris POIs and 92% of Berlin's sat inside 3 km, so the
    # relation returned nearly the whole city and distinguished nothing. It is
    # geometry rather than a data problem -- a 3 km circle covers most of the
    # disc whatever hubs you pick. 1.0 km keeps 18% / 55%, and it is also what
    # the queries actually ask for: a hub is "within walking distance", which
    # 3 km is not.
    def multi_hop_transport_to_poi(self, destination_id: str, category: str, max_km: float = 1.0):
        if self.backend == "neo4j":
            hubs = self.city_transport(destination_id)
            pois = self.city_pois(destination_id, category=category)
            results = []
            for poi in pois:
                best = min(
                    (haversine_km(poi["lat"], poi["lon"], h["lat"], h["lon"]) for h in hubs),
                    default=999,
                )
                if best <= max_km:
                    results.append({**poi, "nearest_hub_km": round(best, 2)})
            return sorted(results, key=lambda x: x["nearest_hub_km"])

        hubs = self.city_transport(destination_id)
        pois = self.city_pois(destination_id, category=category)
        results = []
        for poi in pois:
            best = min(
                (haversine_km(poi["lat"], poi["lon"], h["lat"], h["lon"]) for h in hubs),
                default=999,
            )
            if best <= max_km:
                results.append({**poi, "nearest_hub_km": round(best, 2)})
        return sorted(results, key=lambda x: x["nearest_hub_km"])

    def stats(self):
        if self.backend == "neo4j":
            query = "MATCH (n) RETURN labels(n)[0] AS type, count(n) AS count"
            with self.driver.session() as session:
                result = session.run(query)
                by_type = {record["type"]: record["count"] for record in result}
                return {"nodes": sum(by_type.values()), "edges": 0, "by_type": by_type} # Edge count requires separate query

        types = {}
        for _, data in self.g.nodes(data=True):
            types[data.get("type", "?")] = types.get(data.get("type", "?"), 0) + 1
        return {"nodes": self.g.number_of_nodes(), "edges": self.g.number_of_edges(), "by_type": types}

def save_graph(g: nx.MultiDiGraph, path: Path):
    nx.write_gml(g, path, stringizer=str)

def load_graph(path: Path) -> nx.MultiDiGraph:
    return nx.read_gml(path, destringizer=None)

if __name__ == "__main__":
    g = build_graph()
    idx = GraphIndex(g)
    print("Knowledge graph stats:", idx.stats())
    # Was hardcoded to "IST", which prints nothing once the catalogue no longer
    # ships Istanbul. Take whichever city the graph actually holds.
    demo_city = next(n for n, d in g.nodes(data=True) if d.get("type") == "City")
    print(f"Sample: landmark POIs near {demo_city} transport hubs ->")
    for r in idx.multi_hop_transport_to_poi(demo_city, "landmark")[:3]:
        print(" ", r["name"], f"{r['nearest_hub_km']}km from nearest hub")
    out_path = DATA_DIR / "knowledge_graph.gml"
    save_graph(g, out_path)
    print("Saved graph to", out_path)