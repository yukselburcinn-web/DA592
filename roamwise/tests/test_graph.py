"""The knowledge graph itself: what nodes and relations it holds, what the
transit-derived edges are allowed to say, and whether the committed export
still describes the graph the code builds.
"""

import collections

from roamwise.knowledge_graph.build_graph import (
    GraphIndex,
    REACHABLE_MAX_MIN,
    SERVES_MAX_MIN,
    archetype_node_id,
    load_graph,
    shared_graph,
)
from roamwise.tests.helpers import CITY_CODES, DATA_DIR, FULL_CITIES, MAIN_CITY, POI_COUNT


def test_knowledge_graph_builds_and_traverses():
    idx = GraphIndex()
    stats = idx.stats()
    assert stats["by_type"]["City"] == len(CITY_CODES)
    assert stats["by_type"]["POI"] == POI_COUNT
    hop = idx.multi_hop_transport_to_poi(FULL_CITIES[0] if FULL_CITIES else MAIN_CITY,
                                         "landmark")
    assert len(hop) > 0
    # Minutes, not kilometres: the hop walks `SERVES` edges carrying solved
    # journey times now, rather than recomputing a haversine circle (#126).
    assert all("nearest_hub_minutes" in r for r in hop)
    assert all(r["nearest_hub_minutes"] <= SERVES_MAX_MIN for r in hop)


def test_graph_holds_only_relations_something_reads():
    """Issue #126. `NEAR` (65,838 edges) and `SAME_CATEGORY` (27,232) were 97%
    of the graph and had no consumers -- `SAME_CATEGORY` re-encoded a field
    already on the node, and `NEAR`'s only reader had no callers, tests
    included. Asserted as an absence rather than as an edge count, because
    what must not come back is the relation, at any threshold."""
    relations = {data.get("relation")
                 for _, _, data in shared_graph().edges(data=True)}
    assert "NEAR" not in relations
    assert "SAME_CATEGORY" not in relations
    assert {"SERVES", "REACHABLE"} <= relations


def test_transit_edges_stay_inside_their_thresholds_and_declare_their_source():
    """Both legs of the chain carry solved journey times, and both say where
    the number came from (#126).

    The `source` check is the one that matters: minutes off a GTFS timetable
    and minutes from dividing a straight line by a walking speed are different
    claims, and an edge that did not distinguish them would let the fallback be
    reported as the real thing.
    """
    limits = {"SERVES": SERVES_MAX_MIN, "REACHABLE": REACHABLE_MAX_MIN}
    seen = collections.Counter()
    for _, _, data in shared_graph().edges(data=True):
        relation = data.get("relation")
        if relation not in limits:
            continue
        seen[relation] += 1
        assert data["source"] in ("transit", "haversine")
        assert 0 <= data["minutes"] <= limits[relation]
    assert seen["SERVES"] and seen["REACHABLE"]


def test_reachable_edges_are_directed():
    """`matrix_min` is not symmetric -- a timetable is not a distance -- so the
    second leg is checked and stored per direction. Over the committed
    matrices 804 of 13,602 edges exist one way only; a symmetric build would
    invent the return journeys."""
    minutes = {(a, b): data["minutes"]
               for a, b, data in shared_graph().edges(data=True)
               if data.get("relation") == "REACHABLE"}
    one_way = [pair for pair in minutes if (pair[1], pair[0]) not in minutes]
    assert one_way, "a symmetric threshold would have hidden the asymmetry"


def test_archetype_prefers_only_the_city_it_is_scoped_to():
    """One profile node per (archetype, city), so the traversal is the filter
    (#126). The single node used to hang `PREFERS` on every POI of every city
    and rely on the caller re-filtering afterwards -- the graph carried an edge
    in order for a Python loop to throw it away."""
    graph = shared_graph()
    for city in CITY_CODES:
        node_id = archetype_node_id("Culture Enthusiast", city)
        assert node_id in graph
        for _, poi_id, data in graph.out_edges(node_id, data=True):
            if data.get("relation") != "PREFERS":
                continue
            assert graph.nodes[poi_id]["destination_id"] == city


def test_the_committed_graph_export_describes_the_graph_the_code_builds():
    """`data/knowledge_graph.gml` is the concrete artifact behind the
    proposal's "Knowledge Graph" deliverable (#128), and nothing reads it at
    runtime -- `load_graph` has no callers, so a wrong file breaks nothing and
    is therefore never noticed. It went stale exactly that way: #126 replaced
    `NEAR`/`SAME_CATEGORY` with `SERVES`/`REACHABLE` and split the archetype
    node per city, and the committed export kept describing the old schema for
    four days and nineteen commits (#145).

    Counts rather than the relation set alone, because the way this file goes
    wrong is by being *old*: a catalogue change moves the counts while leaving
    every relation name intact. Both sides are derived -- the expectation is
    whatever `build_graph()` produces today -- so this fails when the export is
    stale, never when the catalogue legitimately changes.
    """
    export_path = DATA_DIR / "knowledge_graph.gml"
    assert export_path.exists(), (
        "the deliverable artifact is missing; rebuild it with "
        "`python -m roamwise.knowledge_graph.build_graph`")
    exported = load_graph(export_path)
    built = shared_graph()

    def relations(graph):
        return collections.Counter(data.get("relation")
                                   for _, _, data in graph.edges(data=True))

    def node_types(graph):
        return collections.Counter(data.get("type") for _, data in graph.nodes(data=True))

    stale = ("`data/knowledge_graph.gml` no longer matches what build_graph() "
             "produces -- regenerate it with "
             "`python -m roamwise.knowledge_graph.build_graph`")
    assert relations(exported) == relations(built), stale
    assert node_types(exported) == node_types(built), stale
