"""The trip starts where the traveler lands (#32): day one from the arrival
hub, later days from the city, and the warning before someone walks in from
the airport.
"""

import datetime

import pandas as pd
import pytest

from roamwise.agents.orchestrator import RETRIEVED_POIS_PER_DAY, RoamWiseOrchestrator
from roamwise.knowledge_graph.build_graph import GraphIndex
from roamwise.optimization.toptw import build_multi_day_itinerary
from roamwise.retrieval import graph_search
from roamwise.retrieval.fusion import FusionRetriever
from roamwise.retrieval.graph_search import GraphSearchIndex
from roamwise.retrieval.query import archetype_query
from roamwise.tests.helpers import CITY_CODES, DATA_DIR, MAIN_CITY


# --- issue #32: the trip starts where the traveler lands ---

def _arrival_hub(city_code):
    """The city's furthest-out gateway, read from the catalogue. An airport is
    the interesting case precisely because it is far from everything -- a hub
    inside the centre would pass these tests without the code doing anything."""
    hubs = pd.read_csv(DATA_DIR / "transport.csv")
    hubs = hubs[hubs["destination_id"] == city_code]
    dests = pd.read_csv(DATA_DIR / "destinations.csv").set_index("destination_id")
    centre = dests.loc[city_code]
    from roamwise.optimization.routing import haversine_km
    ranked = sorted(hubs.itertuples(index=False),
                    key=lambda h: haversine_km(centre.lat, centre.lon, h.lat, h.lon))
    return ranked[-1]


def _city_centre_hub(city_code):
    node = GraphIndex().g.nodes[city_code]
    return {"lat": node["lat"], "lon": node["lon"], "name": node["name"]}


@pytest.mark.slow
def test_day_one_starts_from_the_arrival_hub():
    """`router_agent.py` has carried a comment promising this since #19 -- "the
    airport hub only matters for the single arrival leg" -- with no
    implementation behind it, so every day began at the city centre and the
    transfer in was invisible."""
    from roamwise.optimization.routing import haversine_km

    hub = _arrival_hub(MAIN_CITY)
    arrival = {"lat": hub.lat, "lon": hub.lon, "name": hub.name}
    pois = GraphIndex().city_pois(MAIN_CITY)[:40]

    days = build_multi_day_itinerary(pois, 3, start_hub=_city_centre_hub(MAIN_CITY),
                                     arrival_hub=arrival, daily_minutes_budget=720)

    day_one = days[0]
    assert day_one["route"], "day 1 came back empty"
    assert day_one["starts_from"] == hub.name
    first = day_one["route"][0]
    transfer = haversine_km(arrival["lat"], arrival["lon"], first["lat"], first["lon"])
    assert day_one["distance_km"] >= transfer, \
        "day 1's distance does not include the leg in from the gateway"


@pytest.mark.slow
def test_later_days_still_start_in_the_city():
    """Anchoring every day at the gateway would walk the traveler back out to
    the edge of town each morning. Only day 1 moves."""
    from roamwise.optimization.routing import haversine_km

    hub = _arrival_hub(MAIN_CITY)
    arrival = {"lat": hub.lat, "lon": hub.lon, "name": hub.name}
    centre = _city_centre_hub(MAIN_CITY)
    pois = GraphIndex().city_pois(MAIN_CITY)[:40]

    days = build_multi_day_itinerary(pois, 3, start_hub=centre, arrival_hub=arrival,
                                     daily_minutes_budget=720)

    for day in days[1:]:
        assert day["starts_from"] is None
        if not day["route"]:
            continue
        first = day["route"][0]
        from_centre = haversine_km(centre["lat"], centre["lon"], first["lat"], first["lon"])
        from_hub = haversine_km(arrival["lat"], arrival["lon"], first["lat"], first["lon"])
        assert from_centre < from_hub, f"day {day['day']} opens out by the gateway"


@pytest.mark.slow
def test_arrival_hub_reaches_the_router_from_plan_trip():
    """The failure mode this repo keeps hitting is a parameter that exists on
    the router and cannot be reached from the app (#59 for day_start_hour, #76
    for start_date on the LangGraph path). Plumbing, tested as plumbing."""
    hub = _arrival_hub(MAIN_CITY)
    orch = RoamWiseOrchestrator()
    prefs = {"budget": 0.6, "culture": 0.9, "nature": 0.2, "nightlife": 0.2, "relax": 0.3, "adventure": 0.2}

    planned = orch.plan_trip(prefs, destination_id=MAIN_CITY, n_days=2,
                             daily_minutes_budget=12 * 60,
                             arrival_hub_id=hub.transport_id)

    assert planned["routing"]["itinerary"][0]["starts_from"] == hub.name


def test_arrival_hub_reaches_retrieval_from_plan_trip():
    """The other half of the same wire (#126). `arrival_hub_id` reached the
    router and stopped there, so the one component holding a relation from a
    starting point -- the graph -- never learned where the traveler started.

    Asserted on `GraphSearchIndex.search`'s keyword rather than on the plan,
    because at this phase the anchor is carried and not yet dispatched on; the
    traversal that reads it lands behind a flag. Plumbing, tested as plumbing,
    for the reason the test above exists.
    """
    hub = _arrival_hub(MAIN_CITY)
    seen = []
    original = GraphSearchIndex.search

    def spy(self, *args, **kwargs):
        seen.append(kwargs.get("arrival_hub_id"))
        return original(self, *args, **kwargs)

    orch = RoamWiseOrchestrator()
    prefs = {"budget": 0.6, "culture": 0.9, "nature": 0.2, "nightlife": 0.2, "relax": 0.3, "adventure": 0.2}
    GraphSearchIndex.search = spy
    try:
        orch.plan_trip(prefs, destination_id=MAIN_CITY, n_days=2,
                       daily_minutes_budget=12 * 60,
                       arrival_hub_id=hub.transport_id)
    finally:
        GraphSearchIndex.search = original

    assert seen, "the graph retriever was never reached"
    assert seen[0] == hub.transport_id


def test_the_retrieval_anchor_is_the_gateway_when_named_and_the_centre_otherwise():
    """`anchor_for` is the one place the two starting points are reconciled, so
    the traversal reading it never has to branch (#126).

    A hub belonging to another city, or one `transport.csv` no longer holds,
    resolves to the centre rather than raising -- the same rule a stale hub id
    already gets in the router, and the same rule an unknown travel mode gets.
    """
    graph = GraphSearchIndex()
    hubs = pd.read_csv(DATA_DIR / "transport.csv")
    own = hubs[hubs.destination_id == MAIN_CITY].iloc[0].transport_id
    foreign = hubs[hubs.destination_id != MAIN_CITY]

    assert graph.anchor_for(MAIN_CITY) == MAIN_CITY
    assert graph.anchor_for(MAIN_CITY, own) == own
    assert graph.anchor_for(MAIN_CITY, "TR-does-not-exist") == MAIN_CITY
    if not foreign.empty:
        assert graph.anchor_for(MAIN_CITY, foreign.iloc[0].transport_id) == MAIN_CITY


def test_the_chain_is_on_by_default_and_the_flag_can_still_turn_it_off():
    """#126 phase 5. The chain shipped behind `ROAMWISE_GRAPH_CHAIN` default
    off, and the flag gate (KN-3) passed, so the default moved.

    Both halves are asserted. That it is on is the shipped behaviour. That the
    flag still works is what keeps the measurement the decision rested on
    reproducible -- a gate whose "before" arm can no longer be run is a gate
    nobody can check.
    """
    assert graph_search.CHAIN_ENABLED is True, "the gate passed; the chain ships on"

    retriever = FusionRetriever()
    query = archetype_query("Culture Enthusiast")
    # The pool the orchestrator actually retrieves, not a round number: the
    # chain's share of the top 8 depends on how long the lists RRF is fusing
    # are, and phase 4 chose its weight at top_k=8 and got that wrong.
    top_k = 3 * RETRIEVED_POIS_PER_DAY

    on = retriever.retrieve(query, destination_id=MAIN_CITY,
                            archetype="Culture Enthusiast", top_k=top_k)
    assert any("chain" in r.get("retrieved_by", ()) for r in on)

    graph_search.CHAIN_ENABLED = False
    try:
        off = retriever.retrieve(query, destination_id=MAIN_CITY,
                                 archetype="Culture Enthusiast", top_k=top_k)
    finally:
        graph_search.CHAIN_ENABLED = True
    assert all("chain" not in r.get("retrieved_by", ()) for r in off)


def test_the_chain_weight_keeps_it_inside_the_band_kn2_set():
    """K4, re-measured at the right pool size (#126 phase 5).

    KN-2's band is 2-4 of the top 8. Phase 4 swept the weight at `top_k=8` and
    picked 1.5; at the 72-POI pool a three-day trip actually retrieves, 1.5
    puts the chain at 5.9 of 8 and at 8 of 8 in one cell -- the domination the
    checkpoint exists to catch. 0.15 is the value whose every cell lands
    inside the band.

    Asserted as the property rather than the number, so re-tuning the weight
    for a good reason does not have to edit a test, but breaking the band does.
    """
    retriever = FusionRetriever()
    top_k = 3 * RETRIEVED_POIS_PER_DAY
    shares = []
    for archetype in ("Culture Enthusiast", "Family Traveler"):
        results = retriever.retrieve(archetype_query(archetype), destination_id=MAIN_CITY,
                                     archetype=archetype, top_k=top_k)
        shares.append(sum(1 for r in results[:8] if "chain" in r.get("retrieved_by", ())))

    assert max(shares) <= 4, f"the chain is deciding the answer on its own: {shares}"
    assert sum(shares), f"the chain reaches nothing at all: {shares}"


def test_the_chain_walks_two_real_edges_and_reports_the_path_it_walked():
    """`anchor -[SERVES]-> POI_a -[REACHABLE]-> POI_b`, with POI_b still open
    once POI_a closes (#126).

    The reason text is asserted because it is not decoration: it reaches the
    traveler through the "What the plan was grounded in" block, which turns
    provenance from a claim the report makes into a path they can follow.
    """
    graph = GraphSearchIndex()
    docs = graph.chain_search(destination_id=MAIN_CITY, top_k=20,
                              start_date=datetime.date(2026, 9, 24))
    assert docs, "the shipped catalogue and matrices should support chains"

    # K2: two ordinary POI documents, not a new first-class type.
    assert all(d["type"] == "poi" for d in docs)

    path = docs[0]["text"].split("[graph: ")[1].rstrip("]")
    assert path.count("\u2192") == 4, f"expected two timed legs, got {path!r}"
    assert "closes " in path


def test_the_chain_anchor_follows_the_arrival_hub_when_one_is_named():
    """Centre by default, the gateway when the traveler names one -- and an
    airport 45 minutes out honestly returns nothing rather than being widened
    until it returns something (#126, K3)."""
    graph = GraphSearchIndex()
    day = datetime.date(2026, 9, 24)
    hubs = pd.read_csv(DATA_DIR / "transport.csv")
    city_hubs = hubs[hubs.destination_id == MAIN_CITY]

    centre = {d["poi_id"] for d in graph.chain_search(destination_id=MAIN_CITY, top_k=500,
                                                     start_date=day)}
    assert centre

    by_hub = {}
    for hub_id in city_hubs.transport_id:
        by_hub[hub_id] = {d["poi_id"] for d in graph.chain_search(
            destination_id=MAIN_CITY, top_k=500, arrival_hub_id=hub_id, start_date=day)}

    assert any(pois != centre for pois in by_hub.values()), \
        "naming a gateway must move the anchor"
    # An id this city does not hold falls back to the centre rather than
    # returning nothing -- the rule `anchor_for` documents.
    assert {d["poi_id"] for d in graph.chain_search(
        destination_id=MAIN_CITY, top_k=500, arrival_hub_id="TR-does-not-exist",
        start_date=day)} == centre


def test_the_chain_is_not_raw_two_hop_expansion():
    """The formulation #126 rejected: take the union of both hops and you get
    186 of Paris's 371 POIs from Gare du Nord, 200 from Gare de Lyon -- half
    the catalogue, which is #113's 3 km radius in a new unit.

    Measured on the shipped data, the hour-constrained chain reaches 28% (PAR)
    and 18% (BER).

    The bound is 35%, and #144 is where that number stopped being provisional.
    #126 phase 4 wrote 25% into its acceptance criterion and then measured
    28.3% in Paris, so the test shipped at 30% with a comment admitting the
    gap. Neither number was measured against the thing the bound protects.
    The failure mode is raw two-hop expansion, and the sweep puts it at
    41-54%: a bound has to sit below that and above what a working chain
    actually returns.

    35% is where two independent criteria agree. It clears the shipped 10-minute
    threshold (28.3%) with room, it rejects 15 minutes (36.1%) -- and KN-2
    rejects 15 minutes too, at every weight from 0.02 to 0.40, for the separate
    reason that the chain then takes 5 of the fused top 8. A bound that lands in
    the same place as an unrelated checkpoint is doing more than restating one
    measurement (`evaluation/chain_threshold_weight_sweep.py`).
    """
    graph = GraphSearchIndex()
    day = datetime.date(2026, 9, 24)
    for city in CITY_CODES:
        catalogue = graph.idx.city_pois(city)
        if not catalogue:
            continue
        reached = {d["poi_id"] for d in graph.chain_search(
            destination_id=city, top_k=len(catalogue), start_date=day)}
        share = len(reached) / len(catalogue)
        assert share <= 0.35, f"{city}: chain returns {share:.0%} of the catalogue"


def test_the_chain_only_uses_hours_somebody_actually_observed():
    """277 of the catalogue's 654 rows carry a category default -- "a museum is
    usually open 10-18" -- which nobody checked. Against those the sequencing
    constraint is satisfied for free, so a chain built on them measures
    nothing.

    Both halves are asserted: that the filter is applied, and that it is what
    makes the relation discriminate. Without it the chain reaches roughly half
    the catalogue and separates nothing; with it, under a third.
    """
    graph = GraphSearchIndex()
    day = datetime.date(2026, 9, 24)
    docs = graph.chain_search(destination_id=MAIN_CITY, top_k=500, start_date=day)
    assert docs
    nodes = graph.idx.g.nodes
    assert all(nodes[d["poi_id"]].get("hours_source") in graph_search.CHAIN_REAL_HOURS_SOURCES
               for d in docs)

    catalogue = graph.idx.city_pois(MAIN_CITY)
    observed = [p for p in catalogue
                if p.get("hours_source") in graph_search.CHAIN_REAL_HOURS_SOURCES]
    # The filter's own coverage limit, recorded rather than left to be found:
    # it is why the chain can never reach more than this share (#126, REPORT 5).
    assert len(observed) / len(catalogue) < 0.7
    assert len(docs) / len(catalogue) < len(observed) / len(catalogue)


def test_the_chain_does_not_reimplement_opening_hours():
    """#126 asks for `routing._opening_intervals` to be used rather than
    rewritten: OSM grammar, lunchtime closures and past-midnight hours are
    solved there (#70), and a second reading of the same tags is a second
    reading to drift."""
    from roamwise.optimization import routing

    assert graph_search._opening_intervals is routing._opening_intervals


def test_the_chain_weight_is_recorded_where_rrf_reads_it():
    """K4: the chain's RRF weight is a measured number, not a default. KN-2
    swept it over 8 cells and 1.5 is the value that lands the chain in the
    2-4-of-8 band; the sweep is in `fusion.py`'s comment."""
    from roamwise.retrieval.fusion import DEFAULT_RETRIEVER_WEIGHT, RETRIEVER_WEIGHTS

    assert RETRIEVER_WEIGHTS["chain"] != DEFAULT_RETRIEVER_WEIGHT
    assert RETRIEVER_WEIGHTS["chain"] < RETRIEVER_WEIGHTS["graph"]


@pytest.mark.slow
def test_an_unknown_arrival_hub_is_ignored_rather_than_raised():
    """A stale id from the UI -- a city switched after the gateway was picked --
    must plan a normal trip, the same way an unknown travel mode falls back to
    walking instead of breaking planning."""
    orch = RoamWiseOrchestrator()
    prefs = {"budget": 0.6, "culture": 0.9, "nature": 0.2, "nightlife": 0.2, "relax": 0.3, "adventure": 0.2}

    planned = orch.plan_trip(prefs, destination_id=MAIN_CITY, n_days=2,
                             daily_minutes_budget=12 * 60,
                             arrival_hub_id="TR-does-not-exist")

    itinerary = planned["routing"]["itinerary"]
    assert itinerary[0]["starts_from"] is None
    assert any(day["route"] for day in itinerary)


def test_arrival_options_offer_the_pinned_city_gateways():
    """The picker has to be whatever `transport.csv` holds for the chosen city
    -- the destination dropdown outlived its dataset once already (#65)."""
    from roamwise.views.itinerary import _arrival_options

    options = _arrival_options(MAIN_CITY)
    hubs = pd.read_csv(DATA_DIR / "transport.csv")
    expected = set(hubs.loc[hubs["destination_id"] == MAIN_CITY, "transport_id"])

    assert list(options)[0] == "Already in the city"
    assert options["Already in the city"] is None
    assert set(options.values()) - {None} == expected
    # An unpinned destination has no city yet, so it can offer no gateways.
    assert _arrival_options(None) == {"Already in the city": None}



# --- issue #32: warn before a traveller walks in from the airport ---


def _hub_id(city_code, name_fragment):
    hubs = pd.read_csv(DATA_DIR / "transport.csv")
    hubs = hubs[hubs["destination_id"] == city_code]
    return hubs[hubs.name.str.contains(name_fragment, case=False, na=False)].iloc[0].transport_id


def test_a_long_walk_in_from_the_airport_is_flagged():
    """The itinerary already tells the truth -- day 1 comes back 23.84 km and
    two stops shorter -- but only after planning, and only to someone who
    compares it against day 2. Someone who picked an airport and "on foot" has
    asked for a five-hour walk without knowing it."""
    from roamwise.views.itinerary import _arrival_transfer_hint

    hint = _arrival_transfer_hint(MAIN_CITY, _hub_id(MAIN_CITY, "Charles de Gaulle"), "walking")

    assert hint is not None
    assert "Public transport" in hint, "a warning with no way out is just nagging"


def test_the_hint_stays_quiet_when_switching_would_barely_help():
    """Driving in from Charles de Gaulle is 65 minutes against transit's 50.
    Real, and not worth interrupting anyone over -- a warning that fires on
    every gateway teaches people to dismiss it."""
    from roamwise.views.itinerary import _arrival_transfer_hint

    cdg = _hub_id(MAIN_CITY, "Charles de Gaulle")

    assert _arrival_transfer_hint(MAIN_CITY, cdg, "driving") is None
    assert _arrival_transfer_hint(MAIN_CITY, cdg, "hybrid") is None


def test_the_hint_stays_quiet_once_there_is_nothing_to_suggest():
    from roamwise.views.itinerary import _arrival_transfer_hint

    cdg = _hub_id(MAIN_CITY, "Charles de Gaulle")

    assert _arrival_transfer_hint(MAIN_CITY, cdg, "transit") is None, "already taking it"
    assert _arrival_transfer_hint(MAIN_CITY, None, "walking") is None, "no gateway picked"
    assert _arrival_transfer_hint(MAIN_CITY, _hub_id(MAIN_CITY, "Gare du Nord"),
                                  "walking") is None, "a central station is a short walk"


def test_the_hint_works_in_both_cities():
    """Berlin's timetable shipped after Paris', and the hint follows the data
    rather than naming a city: Brandenburg is as far out as Charles de Gaulle
    and walking in is as bad an idea."""
    from roamwise.views.itinerary import _arrival_transfer_hint

    hint = _arrival_transfer_hint("BER", _hub_id("BER", "Brandenburg"), "walking")

    assert hint is not None and "Public transport" in hint


def test_no_hint_where_there_is_nothing_faster_to_offer(monkeypatch):
    """A warning with no way out is just nagging. Take the timetable away and
    the hint goes quiet, rather than telling someone to use a mode that is not
    on offer."""
    from roamwise.views.itinerary import _arrival_transfer_hint
    # Patched where it is used, not where it is defined: routing.py imported
    # the name, so it holds its own reference.
    import roamwise.optimization.routing as routing_module

    real = routing_module.fetch_distance_duration_matrix
    monkeypatch.setattr(routing_module, "fetch_distance_duration_matrix",
                        lambda points, profile="foot": (None if profile == "transit"
                                                        else real(points, profile=profile)))

    assert _arrival_transfer_hint(MAIN_CITY, _hub_id(MAIN_CITY, "Charles de Gaulle"),
                                  "walking") is None
