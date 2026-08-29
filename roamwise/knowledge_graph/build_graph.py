"""
Knowledge graph construction (Graph-RAG substrate).

The schema, after #126:

    City      -[HAS_POI]->        POI
    City      -[HAS_TRANSPORT]->  Transport
    City      -[SERVES {minutes, source}]->     POI
    Transport -[SERVES {minutes, source}]->     POI
    POI       -[REACHABLE {minutes, source}]->  POI   (directed; see the constant)
    ArchetypeProfile -[PREFERS {weight}]->      POI   (one profile per city)

`SERVES` and `REACHABLE` are the two relations this module exists for, and
between them they are the chain the retrieval layer walks:

    anchor -[SERVES]-> POI_a -[REACHABLE]-> POI_b

Both carry how long the journey takes on the public transport that actually
runs there, solved over GTFS timetables. They replaced `NEAR` (a 2 km
haversine circle between POIs) and `SAME_CATEGORY` (an equality on a field
already stored on the node), which together were 97% of the graph's 95,845
edges and which nothing read: `SAME_CATEGORY` had no consumer at all, and
`NEAR`'s only one had no callers, tests included.

That is the whole argument for holding this in a graph rather than in the
document store the other retrievers use. A timetable is a relation between two
places that neither place's own text can state; a category is not.
"""
import functools
import json
import math
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from neo4j import GraphDatabase

from roamwise.optimization.street_network import load_city_network, point_key
from roamwise.optimization.travel_modes import WALKING

HERE = Path(__file__).parent
DATA_DIR = HERE.parent / "data"

# What "near a transport hub" means for a city we hold no timetable for, in
# kilometres. This is now the *fallback* rule, not the rule: where a
# `{CITY}_transit.npz` exists, `SERVES` is built from solved journey times
# instead (see SERVES_MAX_MIN). It stays at 1.0 rather than the 3.0 it once
# defaulted to (#113), for the reason it was lowered: against a 7 km-radius
# catalogue with real interchanges in it, 96% of Paris POIs and 92% of
# Berlin's sat inside 3 km, so the traversal returned nearly the whole city
# and distinguished nothing. It is geometry rather than a data problem -- a
# 3 km circle covers most of the disc whatever hubs you pick.
#
# It lives here, at module scope, because `pipeline/build_transport.py` reports
# the same share when it selects hubs and had its own copy of the number. That
# copy said 3.0 long after this one moved, so every run of the hub pipeline
# printed a warning about a radius nothing used.
HUB_WALK_KM = 1.0

# What "reachable from here" means on the `SERVES` edge, in minutes of solved
# public-transport journey time (`data/street_network/{CITY}_transit.npz`,
# built by `pipeline/build_transit_matrix.py`).
#
# The graph used to answer "near a transport hub" with a 1 km haversine circle,
# which is a different question -- and a worse one. Measured over the committed
# catalogue, the two rules overlap by a Jaccard of 0.17 (Paris) and 0.34
# (Berlin); 15 minutes lifts POI coverage from 18% to 70% in Paris and 54% to
# 74% in Berlin, and the 372 / 325 POIs that only the transit rule finds are
# exactly the "not walkable but reachable" ones the old rule had no way to
# express. The Louvre is the example worth remembering: over a kilometre from
# any mainline station, twelve minutes from Saint-Lazare.
#
# 15 rather than something tighter because this is the *first* leg of the
# chain -- the traveler getting from where they arrived into the city. The
# second leg (`REACHABLE`) is what has to discriminate, and it is held much
# shorter for that reason.
SERVES_MAX_MIN = 15.0

# What "reachable from there" means on the `REACHABLE` edge, in minutes of the
# same solved journey time. This is the chain's *second* leg -- one POI to the
# next -- and it is the leg that has to discriminate, so it is held much
# shorter than SERVES_MAX_MIN.
#
# Swept over the committed matrices rather than guessed. Directed edges, and
# how much of the catalogue each threshold puts within reach of an average POI:
#
#     <=10 min   PAR 8,530 · BER 5,072    23 / 18 per POI   ~6% of the catalogue
#     <=15 min   PAR 28,324 · BER 13,015  76 / 46 per POI
#     <=20 min   PAR 56,379 · BER 25,823  152 / 91 per POI  40% of all pairs
#
# 20 minutes is #113's 3 km radius in a new unit: a relation that holds
# two fifths of every pair distinguishes nothing, and a "multi-hop query"
# answered with it is a catalogue dump. 10 is where the relation still says
# something. Phase 5 sweeps it for real alongside SERVES_MAX_MIN; going above
# 15 is already measured to be wrong.
#
# One constant, at module scope, deliberately: HUB_WALK_KM was split into two
# copies and one of them went stale for an entire issue (#113).
REACHABLE_MAX_MIN = 10.0

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


def archetype_node_id(archetype: str, destination_id: str) -> str:
    """The profile node for one archetype *in one city*. See `build_graph` for
    why the pair is the identity rather than the archetype alone."""
    return f"ARCH::{archetype}::{destination_id}"


def _transit_minutes(destination_id: str):
    """The solved journey-time matrix for one city, or None if we hold none.

    Returns `(key -> row index, minutes)` where the key is
    `street_network.point_key(lat, lon)`. Going through that function rather
    than formatting the coordinate here is deliberate: a matrix keyed one way
    and read another shows up as "the graph quietly has no SERVES edges", not
    as a crash.
    """
    net = load_city_network(destination_id, "transit")
    if net is None:
        return None
    return net["index"], np.asarray(net["data"]["matrix_min"], dtype=np.float64)


def _haversine_km_array(lat, lon, lats, lons):
    """`haversine_km` from one point to many, in one numpy expression."""
    r = 6371.0
    p1 = math.radians(lat)
    p2 = np.radians(lats)
    dphi = np.radians(lats - lat)
    dl = np.radians(lons - lon)
    a = np.sin(dphi / 2) ** 2 + math.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def serves_builder(destination_id: str, pois: pd.DataFrame):
    """Returns `origin -> [(poi_id, minutes, source), ...]` for one city.

    The city's POIs are resolved against the transit matrix once, here, rather
    than per origin: there are eleven origins per city (ten hubs and the
    centre) and 371 POIs, so keying every POI once per origin meant formatting
    four thousand coordinate strings to answer a question already answered.
    Each origin is then a single row slice and a comparison.

    Two sources, and the edge says which. `"transit"` is a journey time solved
    over the GTFS timetable (`pipeline/build_transit_matrix.py`). `"haversine"`
    is the old 1 km circle at walking speed, kept for a city with no committed
    timetable and *marked*, because minutes from dividing a straight line by a
    speed are a different claim from minutes off a timetable, and a graph that
    presented them identically would let the weaker one be reported as the
    stronger. A POI the matrix does not hold falls back the same way rather
    than dropping out: catalogue and matrix are built by separate pipelines and
    can drift, and a silently missing edge is the failure mode hardest to
    notice.
    """
    poi_ids = pois["poi_id"].to_numpy()
    lats = pois["lat"].to_numpy(dtype=np.float64)
    lons = pois["lon"].to_numpy(dtype=np.float64)

    transit = _transit_minutes(destination_id)
    if transit is None:
        index, minutes = None, None
        columns = np.full(len(poi_ids), -1)
    else:
        index, minutes = transit
        columns = np.array([index.get(point_key(lat, lon), -1)
                            for lat, lon in zip(lats, lons)])
    known = columns >= 0

    def edges(origin: dict):
        row = (index.get(point_key(origin["lat"], origin["lon"]), -1)
               if index is not None else -1)
        out = []
        if row >= 0:
            travel = minutes[row, columns[known]]
            close = travel <= SERVES_MAX_MIN
            out += [(poi_id, round(float(m), 1), "transit")
                    for poi_id, m in zip(poi_ids[known][close], travel[close])]
            missed = ~known
        else:
            missed = np.ones(len(poi_ids), dtype=bool)
        if missed.any():
            km = _haversine_km_array(origin["lat"], origin["lon"],
                                     lats[missed], lons[missed])
            walk = km <= HUB_WALK_KM
            out += [(poi_id, round(float(d) / WALKING.speed_kmh * 60.0, 1), "haversine")
                    for poi_id, d in zip(poi_ids[missed][walk], km[walk])]
        return out

    return edges


def reachable_edges(destination_id: str, pois: pd.DataFrame):
    """`(from_poi, to_poi, minutes, source)` for every ordered pair of this
    city's POIs inside REACHABLE_MAX_MIN -- the chain's second leg.

    **Directed, and the direction matters.** `matrix_min` is not symmetric:
    over the committed matrices 25% of pairs differ by more than 5% between
    the two directions and the worst gap is 10.8 minutes, because a timetable
    is not a distance. The graph is a `MultiDiGraph` already, so each direction
    is checked and stored on its own edge; assuming symmetry here would quietly
    invent a return journey that no service makes.

    **Data limit, which belongs in REPORT §5 rather than only here.**
    `matrix_min` is the median over thirteen departures between 08:00 and
    20:00 (`pipeline/build_transit_matrix.py`). Night service is outside it.
    Measured: 96% of the chains this relation supports depart before 20:00, so
    the remaining 4% are edge cases priced at a daytime rate -- small, but not
    nothing, and not something the reader should have to discover.

    `source` marks provenance exactly as `serves_builder` does, for the same
    reason. The fallback radius is REACHABLE_MAX_MIN at walking speed rather
    than a constant of its own: the two paths have to express the same rule,
    and a second number here is a second number to go stale.
    """
    poi_ids = pois["poi_id"].to_numpy()
    lats = pois["lat"].to_numpy(dtype=np.float64)
    lons = pois["lon"].to_numpy(dtype=np.float64)
    walk_km = REACHABLE_MAX_MIN / 60.0 * WALKING.speed_kmh

    transit = _transit_minutes(destination_id)
    if transit is None:
        columns = np.full(len(poi_ids), -1)
        index = minutes = None
    else:
        index, minutes = transit
        columns = np.array([index.get(point_key(lat, lon), -1)
                            for lat, lon in zip(lats, lons)])
    known = columns >= 0

    out = []
    if known.any():
        rows = columns[known]
        block = minutes[np.ix_(rows, rows)].copy()
        np.fill_diagonal(block, np.inf)
        ids = poi_ids[known]
        a, b = np.nonzero(block <= REACHABLE_MAX_MIN)
        out += [(ids[i], ids[j], round(float(block[i, j]), 1), "transit")
                for i, j in zip(a, b)]

    # A POI the matrix does not hold still gets its pair, both ways, off the
    # walking estimate -- see `serves_builder` on why these are marked rather
    # than dropped.
    for position in np.nonzero(~known)[0]:
        km = _haversine_km_array(lats[position], lons[position], lats, lons)
        km[position] = np.inf
        for other in np.nonzero(km <= walk_km)[0]:
            travel = round(float(km[other]) / WALKING.speed_kmh * 60.0, 1)
            out.append((poi_ids[position], poi_ids[other], travel, "haversine"))
            if known[other]:
                out.append((poi_ids[other], poi_ids[position], travel, "haversine"))
    return out


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
            # Where those hours came from: "osm", "gmaps", or
            # "category_default" -- the last meaning nobody ever observed this
            # venue and the catalogue filled in what a museum usually does.
            # Carried onto the node because the chain traversal has to be able
            # to *exclude* the defaults: 277 of 654 rows hold them, they are
            # effectively "always open", and a sequencing constraint every one
            # of them satisfies for free is not a constraint (#126).
            hours_source=("" if pd.isna(p.get("hours_source"))
                          else str(p["hours_source"]).strip()),
        )
        g.add_edge(p.destination_id, p.poi_id, relation="HAS_POI")

    for _, t in transport.iterrows():
        g.add_node(
            t.transport_id, type="Transport", name=t["name"], ttype=t.type,
            lat=t.lat, lon=t.lon, destination_id=t.destination_id,
        )
        g.add_edge(t.destination_id, t.transport_id, relation="HAS_TRANSPORT")

    # The first leg of the chain: what each of a city's starting points reaches
    # inside SERVES_MAX_MIN. Two kinds of origin, and the traversal treats them
    # the same way:
    #
    #   Transport -[SERVES]-> POI   the hub the traveler arrived at, when
    #                               `arrival_hub_id` names one
    #   City      -[SERVES]-> POI   the city centre, which is where the router
    #                               already starts every other day (#32)
    #
    # This is the pair of relations that replaces `NEAR` and `SAME_CATEGORY`.
    # Those two were 97% of the graph's edges and nothing read either of them:
    # `SAME_CATEGORY` re-encoded a field already on the node, and `NEAR`'s only
    # consumer -- `nearby_pois` -- had no callers at all, tests included. What
    # the graph is *for* is the relation a document store cannot express, and a
    # timetable is that relation; a category equality is not.
    for _, d in destinations.iterrows():
        city_pois = pois[pois.destination_id == d.destination_id]
        serves = serves_builder(d.destination_id, city_pois)
        origins = [(d.destination_id, {"lat": d.lat, "lon": d.lon})]
        origins += [(t.transport_id, {"lat": t.lat, "lon": t.lon})
                    for t in transport[transport.destination_id == d.destination_id].itertuples()]
        for origin_id, origin in origins:
            for poi_id, minutes, source in serves(origin):
                g.add_edge(origin_id, poi_id, relation="SERVES",
                           minutes=minutes, source=source)
        for from_id, to_id, minutes, source in reachable_edges(d.destination_id, city_pois):
            g.add_edge(from_id, to_id, relation="REACHABLE",
                       minutes=minutes, source=source)

    # One profile node per (archetype, city), not one per archetype. The single
    # node used to hang `PREFERS` edges on every POI of *every* city, so the
    # relation said "a Culture Enthusiast prefers the Louvre and the Pergamon",
    # which is not a preference anybody holds -- a traveler is in one city. It
    # only looked harmless because the one consumer re-filtered on the POI's
    # `destination_id` after traversing, i.e. the graph carried an edge in
    # order for a Python loop to throw it away. Scoping the node makes the
    # traversal itself the filter, which is the whole point of holding this in
    # a graph.
    for archetype, affinities in CATEGORY_AFFINITY.items():
        for dest_id in destinations.destination_id:
            node_id = archetype_node_id(archetype, dest_id)
            g.add_node(node_id, type="ArchetypeProfile", name=archetype,
                       destination_id=dest_id, affinities=affinities)
            city_pois = pois[pois.destination_id == dest_id]
            for category, weight in affinities.items():
                for poi_id in city_pois[city_pois.category == category]["poi_id"]:
                    g.add_edge(node_id, poi_id, relation="PREFERS", weight=weight)

    return g


@functools.lru_cache(maxsize=1)
def shared_graph() -> nx.MultiDiGraph:
    """The graph a plain `GraphIndex()` reads, built once per process.

    `build_graph` re-reads three CSVs and rebuilds every node and edge on each
    call. A single
    `RoamWiseOrchestrator` builds two of these (its own and the one inside
    GraphSearchIndex), and the test suite builds 19 orchestrators, so the same
    graph was being constructed dozens of times per run from the same files.

    Sharing one object is safe because nothing mutates the graph after
    construction: every query method copies node attributes out
    (`{"poi_id": poi_id, **node}`) rather than handing back the live dict, and
    the only writers are inside `build_graph` itself. Callers that do want an
    independent graph -- the pipeline, or a caller pointing at rewritten CSVs
    in the same process -- still call `build_graph()` directly, or clear this
    with `shared_graph.cache_clear()`.
    """
    return build_graph()


class GraphIndex:
    """Query surface over the knowledge graph supporting NetworkX and Neo4j backends."""

    def __init__(self, graph: nx.MultiDiGraph = None, backend="networkx", neo4j_uri="bolt://localhost:7687", neo4j_auth=("neo4j", "password")):
        self.backend = backend
        if self.backend == "neo4j":
            self.driver = GraphDatabase.driver(neo4j_uri, auth=neo4j_auth)
        else:
            self.g = graph if graph is not None else shared_graph()

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
                query += "WHERE a.destination_id = $dest_id "
            query += "RETURN p, r.weight AS weight"
            with self.driver.session() as session:
                result = session.run(query, arch=archetype, dest_id=destination_id)
                out = [{"poi_id": record["p"].element_id, "weight": record["weight"],
                        **record["p"]} for record in result]
            return rank_preferred(out, top_k)

        # The profile node is per (archetype, city), so naming a destination is
        # a node lookup rather than a filter over the other city's edges. With
        # no destination the archetype's nodes are unioned, which is what the
        # single node used to return -- callers that want one city pass one.
        if destination_id is not None:
            nodes = [archetype_node_id(archetype, destination_id)]
        else:
            nodes = [n for n, data in self.g.nodes(data=True)
                     if data.get("type") == "ArchetypeProfile"
                     and data.get("name") == archetype]
        out = []
        for node_id in nodes:
            if node_id not in self.g:
                continue
            for _, poi_id, data in self.g.out_edges(node_id, data=True):
                if data.get("relation") != "PREFERS":
                    continue
                out.append({"poi_id": poi_id, "weight": data["weight"],
                            **self.g.nodes[poi_id]})
        return rank_preferred(out, top_k)

    def multi_hop_transport_to_poi(self, destination_id: str, category: str = None,
                                   max_minutes: float = SERVES_MAX_MIN):
        """POIs a transport hub of this city reaches, nearest in time first.

        `City -[HAS_TRANSPORT]-> Transport -[SERVES]-> POI`: two hops, both of
        them edges. It used to be two one-hop fetches and a haversine cross
        product computed in Python -- 371 x 10 great-circle distances per call,
        recomputed every call, over a relation the graph did not hold. Calling
        that "multi-hop traversal" was the single largest gap between what the
        proposal claims Graph-RAG does here and what the code did.

        `nearest_hub_minutes` rather than `nearest_hub_km`, because the answer
        now comes from a timetable. The unit change is the finding: measured
        over the committed catalogue, "museums near a Paris transport hub" goes
        from 6 results to 36, and the ones it gains are places like the Louvre
        -- over a kilometre from any mainline station, twelve minutes from
        Saint-Lazare. An edge built by `serves_builder`'s haversine fallback
        still reports minutes, but says `source="haversine"`; see there.
        """
        if self.backend == "neo4j":
            query = ("MATCH (c:City {id: $dest_id})-[:HAS_TRANSPORT]->(:Transport)"
                     "-[r:SERVES]->(p:POI) WHERE r.minutes <= $max_minutes ")
            if category:
                query += "AND p.category = $category "
            query += ("RETURN p, min(r.minutes) AS minutes ORDER BY minutes")
            with self.driver.session() as session:
                result = session.run(query, dest_id=destination_id, category=category,
                                     max_minutes=max_minutes)
                return [{"poi_id": record["p"].element_id,
                         "nearest_hub_minutes": record["minutes"], **record["p"]}
                        for record in result]

        # Hub ids straight off the edge, not `city_transport` -- this hop only
        # needs the identity, and materialising each hub's node attributes to
        # throw them away is most of what the traversal would cost.
        hubs = [tid for _, tid, data in self.g.out_edges(destination_id, data=True)
                if data.get("relation") == "HAS_TRANSPORT"]
        best: dict[str, float] = {}
        for hub_id in hubs:
            for _, poi_id, data in self.g.out_edges(hub_id, data=True):
                if data.get("relation") != "SERVES":
                    continue
                minutes = data["minutes"]
                if minutes > max_minutes:
                    continue
                if category and self.g.nodes[poi_id].get("category") != category:
                    continue
                if minutes < best.get(poi_id, math.inf):
                    best[poi_id] = minutes
        results = [{"poi_id": poi_id, "nearest_hub_minutes": minutes,
                    **self.g.nodes[poi_id]} for poi_id, minutes in best.items()]
        return sorted(results, key=lambda x: x["nearest_hub_minutes"])

    def chains_from(self, anchor_id: str, max_serves_min: float = SERVES_MAX_MIN,
                    max_reachable_min: float = REACHABLE_MAX_MIN):
        """Every `anchor -[SERVES]-> POI_a -[REACHABLE]-> POI_b` in this city.

        Two edges, both walked -- this is the traversal the proposal's
        Graph-RAG claim rests on, and until #126 the graph held neither of
        them. Returns `(poi_a, minutes_to_a, poi_b, minutes_a_to_b)` with the
        POIs as node dicts, unranked and unfiltered: whether a chain is *usable*
        depends on opening hours and on what the traveler asked for, and
        neither of those is something the graph knows. `retrieval/graph_search.py`
        decides that.

        `anchor_id` is a `City` or a `Transport` node -- both carry `SERVES`,
        which is why `GraphSearchIndex.anchor_for` can hand over either without
        the caller branching.
        """
        if self.backend == "neo4j":
            query = ("MATCH (anchor {id: $anchor_id})-[s:SERVES]->(a:POI)"
                     "-[r:REACHABLE]->(b:POI) "
                     "WHERE s.minutes <= $max_serves AND r.minutes <= $max_reachable "
                     "RETURN a, s.minutes AS to_a, b, r.minutes AS a_to_b")
            with self.driver.session() as session:
                result = session.run(query, anchor_id=anchor_id, max_serves=max_serves_min,
                                     max_reachable=max_reachable_min)
                return [({"poi_id": record["a"].element_id, **record["a"]}, record["to_a"],
                         {"poi_id": record["b"].element_id, **record["b"]}, record["a_to_b"])
                        for record in result]

        if anchor_id not in self.g:
            return []
        out = []
        for _, a_id, first in self.g.out_edges(anchor_id, data=True):
            if first.get("relation") != "SERVES" or first["minutes"] > max_serves_min:
                continue
            node_a = {"poi_id": a_id, **self.g.nodes[a_id]}
            for _, b_id, second in self.g.out_edges(a_id, data=True):
                if second.get("relation") != "REACHABLE" or second["minutes"] > max_reachable_min:
                    continue
                out.append((node_a, first["minutes"],
                            {"poi_id": b_id, **self.g.nodes[b_id]}, second["minutes"]))
        return out

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
    print(f"Sample: landmark POIs served by {demo_city} transport hubs ->")
    for r in idx.multi_hop_transport_to_poi(demo_city, "landmark")[:3]:
        print(" ", r["name"], f"{r['nearest_hub_minutes']}min from nearest hub")
    out_path = DATA_DIR / "knowledge_graph.gml"
    save_graph(g, out_path)
    print("Saved graph to", out_path)