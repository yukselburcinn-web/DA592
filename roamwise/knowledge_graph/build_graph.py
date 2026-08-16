"""
Knowledge graph construction (Graph-RAG substrate).

We use NetworkX rather than a live Neo4j server: it needs no external service,
is trivially inspectable/testable, and supports every traversal this project
needs (multi-hop neighbor queries, shortest path, filtered subgraphs). The
schema mirrors what a Neo4j property graph would hold, so swapping the
backend later means re-implementing GraphIndex.query_* against the Neo4j
Python driver -- the rest of the pipeline (agents, fusion, evaluation) is
backend-agnostic because it only talks to GraphIndex.

Node types: City, POI, Transport, ArchetypeProfile
Edge types:
  City -[HAS_POI]-> POI
  City -[HAS_TRANSPORT]-> Transport
  POI  -[NEAR]-> POI               (spatial proximity within a city)
  POI  -[SAME_CATEGORY]-> POI
  ArchetypeProfile -[PREFERS]-> POI category (via CATEGORY_AFFINITY weight)
"""
import json
import math
from pathlib import Path

import networkx as nx
import pandas as pd

HERE = Path(__file__).parent
DATA_DIR = HERE.parent / "data"

# category affinity per traveler archetype -- feeds the ArchetypeProfile PREFERS edges
CATEGORY_AFFINITY = {
    "Culture Enthusiast": {"museum": 1.0, "landmark": 0.9, "history": 0.9, "religion": 0.6, "culture": 0.8},
    "Beach & Relax": {"beach": 1.0, "nature": 0.8, "food": 0.5, "culture": 0.3},
    "Budget Backpacker": {"nightlife": 0.9, "culture": 0.6, "shopping": 0.3, "food": 0.7, "nature": 0.6},
    "Luxury Traveler": {"culture": 0.7, "museum": 0.6, "shopping": 0.9, "food": 0.8, "landmark": 0.7},
    "Nightlife Seeker": {"nightlife": 1.0, "food": 0.6, "culture": 0.3},
    "Nature & Adventure": {"nature": 1.0, "beach": 0.6, "history": 0.3},
    "Family Traveler": {"nature": 0.8, "culture": 0.6, "museum": 0.6, "landmark": 0.6, "food": 0.5},
}


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
        )
        g.add_edge(p.destination_id, p.poi_id, relation="HAS_POI")

    for _, t in transport.iterrows():
        g.add_node(
            t.transport_id, type="Transport", name=t["name"], ttype=t.type,
            lat=t.lat, lon=t.lon, destination_id=t.destination_id,
        )
        g.add_edge(t.destination_id, t.transport_id, relation="HAS_TRANSPORT")

    # POI <-> POI edges: spatial proximity (NEAR, <=2km) and category affinity
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

    # Archetype profile nodes with PREFERS edges into category-representative POIs
    for archetype, affinities in CATEGORY_AFFINITY.items():
        node_id = f"ARCH::{archetype}"
        g.add_node(node_id, type="ArchetypeProfile", name=archetype, affinities=affinities)
        for category, weight in affinities.items():
            matches = pois[pois.category == category]["poi_id"].tolist()
            for poi_id in matches:
                g.add_edge(node_id, poi_id, relation="PREFERS", weight=weight)

    return g


class GraphIndex:
    """Query surface over the knowledge graph used by the Fusion RAG / agent layer."""

    def __init__(self, graph: nx.MultiDiGraph = None):
        self.g = graph if graph is not None else build_graph()

    def city_pois(self, destination_id: str, category: str = None):
        out = []
        for _, poi_id, data in self.g.out_edges(destination_id, data=True):
            if data.get("relation") == "HAS_POI":
                node = self.g.nodes[poi_id]
                if category is None or node.get("category") == category:
                    out.append({"poi_id": poi_id, **node})
        return out

    def city_transport(self, destination_id: str):
        out = []
        for _, tid, data in self.g.out_edges(destination_id, data=True):
            if data.get("relation") == "HAS_TRANSPORT":
                out.append({"transport_id": tid, **self.g.nodes[tid]})
        return out

    def nearby_pois(self, poi_id: str, max_km: float = 2.0):
        out = []
        for _, other_id, data in self.g.out_edges(poi_id, data=True):
            if data.get("relation") == "NEAR" and data.get("distance_km", 999) <= max_km:
                out.append({"poi_id": other_id, "distance_km": data["distance_km"], **self.g.nodes[other_id]})
        return sorted(out, key=lambda x: x["distance_km"])

    def archetype_preferred_pois(self, archetype: str, destination_id: str = None, top_k: int = 20):
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
        out.sort(key=lambda x: (x["weight"], x["popularity_score"]), reverse=True)
        return out[:top_k]

    def multi_hop_transport_to_poi(self, destination_id: str, category: str, max_km: float = 3.0):
        """Multi-hop traversal: City -> Transport, City -> POI(category), ranked by
        proximity to the nearest transport hub. This is the kind of chained
        relational query flat document retrieval cannot answer directly."""
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
    print("Sample: culture POIs near Istanbul transport hubs ->")
    for r in idx.multi_hop_transport_to_poi("IST", "landmark")[:3]:
        print(" ", r["name"], f"{r['nearest_hub_km']}km from nearest hub")
    out_path = DATA_DIR / "knowledge_graph.gml"
    save_graph(g, out_path)
    print("Saved graph to", out_path)
