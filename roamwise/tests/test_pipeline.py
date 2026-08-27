"""End-to-end smoke tests covering every module in the pipeline. Run with:
    cd roamwise && ../venv/bin/pytest tests/ -v
"""
import collections
import datetime
import importlib
import math
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from roamwise.knowledge_graph.build_graph import CATEGORY_AFFINITY, GraphIndex
from roamwise.pipeline.common import NOT_A_SIGHT_TYPES
from roamwise.models.forecasting import forecast_city, best_months_to_visit
from roamwise.models.segmentation import TravelerSegmenter, POIZoner
from roamwise.retrieval.fusion import FusionRetriever
from roamwise.retrieval.graph_search import GraphSearchIndex
from roamwise.optimization.routing import NIGHTLIFE_EARLIEST_HOUR, optimize_day_route
from roamwise.optimization.toptw import build_multi_day_itinerary
from roamwise.optimization.travel_modes import get_travel_mode
from roamwise.agents.orchestrator import RoamWiseOrchestrator
from roamwise.agents.router_agent import DAY_START_HOURS, RouterAgent, start_hour_for
from roamwise.evaluation import comparative_analysis
from roamwise.evaluation.comparative_analysis import (
    TEST_QUERIES, dependence_level, gold_for, paired_significance,
    run_comparative_analysis, summarize,
)


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _dataset():
    """Read the city list and counts out of the committed CSVs.

    These were literals -- "ROM", "VIE", City == 8, POI == 1200 -- which pinned
    the suite to one particular catalogue. Changing the destination list then
    turned tests red for reasons that had nothing to do with the code under
    test. Reading the dataset keeps every assertion just as strong while
    letting the catalogue change underneath it.

    `FULL_CITIES` is the narrower set: a city needs rows in demand_timeseries
    and transport as well, and those files do not necessarily cover every
    destination. Tests that need them skip with a clear message rather than
    failing somewhere confusing.
    """
    dests = pd.read_csv(DATA_DIR / "destinations.csv")
    pois = pd.read_csv(DATA_DIR / "poi.csv")
    demand = pd.read_csv(DATA_DIR / "demand_timeseries.csv")
    transport = pd.read_csv(DATA_DIR / "transport.csv")

    counts = pois.destination_id.value_counts()
    # Deepest catalogue first: several tests slice [:40] or [:24], and a shallow
    # city would make those assertions vacuous rather than wrong.
    with_pois = sorted((c for c in dests.destination_id if counts.get(c, 0)),
                       key=lambda c: -counts[c])
    full = [c for c in with_pois
            if c in set(demand.destination_id) and c in set(transport.destination_id)]
    return with_pois, full, pois


CITY_CODES, FULL_CITIES, _POIS = _dataset()
POI_COUNT = len(_POIS)
MAIN_CITY = CITY_CODES[0]

# Applied to the tests that index FULL_CITIES[0]: without it an incomplete
# dataset fails with an IndexError at collection rather than saying why.
needs_full_city = pytest.mark.skipif(
    not FULL_CITIES,
    reason="hicbir sehirde hem demand_timeseries hem transport satiri yok")


def city_with_category(category, cities=None):
    """The deepest city holding POIs in this category, or None.

    The catalogue is not obliged to carry every category in every city -- the
    two-city set has no `beach` at all -- so a test that needs one asks for it
    instead of naming a city that happened to have it.
    """
    for code in (cities or CITY_CODES):
        if len(_POIS[(_POIS.destination_id == code) & (_POIS.category == category)]):
            return code
    return None


@needs_full_city

def _flat(zones: dict):
    """Zones were how the old router was *told* which POI belonged to which
    day. TOPTW decides that itself, jointly with selection and ordering, so
    these tests hand it the same POIs as one pool and say how many days they
    have. What each test asserts about the result is unchanged."""
    return [poi for zone in zones.values() for poi in zone], len(zones)


def test_knowledge_graph_builds_and_traverses():
    idx = GraphIndex()
    stats = idx.stats()
    assert stats["by_type"]["City"] == len(CITY_CODES)
    assert stats["by_type"]["POI"] == POI_COUNT
    hop = idx.multi_hop_transport_to_poi(FULL_CITIES[0] if FULL_CITIES else MAIN_CITY,
                                         "landmark")
    assert len(hop) > 0
    assert all("nearest_hub_km" in r for r in hop)


@needs_full_city
def test_forecasting_produces_crowding_labels():
    fc = forecast_city(FULL_CITIES[0])
    assert len(fc) == 12
    assert set(fc.crowding_level.astype(str)) <= {"low", "medium", "high"}
    months = best_months_to_visit(FULL_CITIES[0], top_k=3)
    assert len(months) == 3


def test_traveler_segmentation_classifies():
    seg = TravelerSegmenter()
    result = seg.classify({"budget": 0.1, "culture": 0.2, "nature": 0.9, "nightlife": 0.1, "relax": 0.3, "adventure": 0.9})
    assert result["archetype"] == "Nature & Adventure"


def test_preference_levels_produce_distinct_archetypes():
    """Issue #23: the sidebar exposes Low/Medium/High instead of raw 0-1
    sliders, so the three tiers must actually separate archetype centroids --
    not just satisfy TravelerSegmenter.classify()'s float contract."""
    from roamwise.views.itinerary import PREFERENCE_LEVELS
    from roamwise.models.segmentation import FEATURES

    seg = TravelerSegmenter()
    archetypes = {
        level: seg.classify({f: value for f in FEATURES})["archetype"]
        for level, value in PREFERENCE_LEVELS.items()
    }
    assert len(set(archetypes.values())) == len(archetypes)


def test_poi_zoning_covers_all_pois():
    idx = GraphIndex()
    pois = idx.city_pois(MAIN_CITY)
    zones = POIZoner().zone(pois, n_zones=3)
    total = sum(len(v) for v in zones.values())
    assert total == len(pois)


def test_fusion_retrieval_all_configs_run():
    fr = FusionRetriever()
    for config in ["fusion", "hybrid", "standard"]:
        results = fr.retrieve("museums near a transport hub", config=config,
                              destination_id=MAIN_CITY, top_k=5)
        if config == "standard":
            assert results == []
        else:
            assert len(results) > 0
            assert all(r["destination_id"] == MAIN_CITY for r in results)


def test_a_crowded_category_does_not_starve_a_more_famous_one():
    """Ordering archetype preferences lexicographically on (weight, popularity)
    let one category bury every lower-weighted one outright. Culture Enthusiast
    weights `museum` 1.0 and `landmark` 0.9, and the deepest city holds far
    more museums than retrieval asks for, so every rank up to the museum count
    was a museum and the catalogue's single most popular POI -- a landmark --
    sat behind the city's *last* museum, past the depth anything reads (#63).

    Asserted as a property rather than against a POI name: what must hold is
    that the top-weighted category cannot monopolise the ranking while a
    better-known place in a nearly-as-preferred category waits behind it.
    """
    idx = GraphIndex()
    city = city_with_category("landmark")
    if city is None or not city_with_category("museum", [city]):
        pytest.skip("needs a city holding both museums and landmarks")

    ranked = idx.archetype_preferred_pois("Culture Enthusiast", city, top_k=200)
    assert ranked, "the archetype should prefer something in this city"

    # The most popular POI in any category this archetype prefers at all.
    most_popular = max(ranked, key=lambda p: p["popularity_score"])
    rank = next(i for i, p in enumerate(ranked, 1) if p["poi_id"] == most_popular["poi_id"])

    top_category = max(CATEGORY_AFFINITY["Culture Enthusiast"].items(), key=lambda kv: kv[1])[0]
    n_top_category = sum(1 for p in ranked if p.get("category") == top_category)
    if most_popular.get("category") == top_category:
        pytest.skip("the most popular POI is already in the top-weighted category")

    assert rank < n_top_category, (
        f"{most_popular['name']} ({most_popular['category']}, "
        f"popularity {most_popular['popularity_score']}) ranks {rank}, behind all "
        f"{n_top_category} {top_category} POIs -- the starvation of #63")
    # And it has to survive the depth retrieval actually reads.
    assert rank <= 48


def test_retrieval_query_describes_what_the_traveler_wants_not_the_label():
    """The query was built by interpolating the archetype's name, which made
    BM25 match the literal label word: "culture" surfaced a television channel
    whose description happens to use it (#63). The query has to name
    categories, not the archetype."""
    from roamwise.retrieval.query import CATEGORY_PHRASE, archetype_query

    for archetype, affinities in CATEGORY_AFFINITY.items():
        query = archetype_query(archetype).lower()
        assert archetype.lower() not in query, \
            f"{archetype!r} query still contains the archetype label: {query!r}"
        strongest = max(affinities.items(), key=lambda kv: kv[1])[0]
        assert CATEGORY_PHRASE[strongest] in query, \
            f"{archetype!r} query omits its strongest category {strongest!r}: {query!r}"


def test_the_graph_router_understands_words_travelers_actually_use():
    """It matched only the catalogue's taxonomy words, so "places of worship" --
    the phrasing both the evaluation grid and the orchestrator emit -- routed as
    naming no category at all (#63). Matching is on whole words: "pub" inside
    "public transit" must not make a nightlife query."""
    from roamwise.retrieval.graph_search import categories_in

    assert categories_in("places of worship") == ["religion"]
    assert categories_in("quiet gardens away from the crowds") == ["nature"]
    assert "nightlife" not in categories_in("accessible via late-night public transit")
    assert "nature" not in categories_in("is there parking nearby")
    assert "food" not in categories_in("a great theatre")
    # A profile query names several; a constrained one names exactly one. The
    # retriever routes on that difference.
    assert len(categories_in("museums close to a train station")) == 1
    assert len(categories_in(
        "the best museums, landmarks, history sites, culture venues "
        "and places of worship to visit in this city")) > 1


def test_zoning_returns_every_day_it_was_asked_for():
    """`zone` returned one zone per POI when candidates ran short, so a 5-day
    trip with 3 sightseeing POIs quietly became a 3-day itinerary, and an empty
    pool returned no zones at all -- which `_rebalance_days` crashed on with
    "min() iterable argument is empty" once a query surfaced nothing but food
    (#63)."""
    zoner = POIZoner()
    pois = [{"lat": 48.85 + i / 100, "lon": 2.35 + i / 100} for i in range(3)]

    assert sorted(zoner.zone(pois, n_zones=5)) == [0, 1, 2, 3, 4]
    assert sum(len(z) for z in zoner.zone(pois, n_zones=5).values()) == len(pois)
    assert zoner.zone([], n_zones=1) == {0: []}

    # The crash this guards: every candidate is food, so the sightseeing pool
    # the zoner is handed is empty.
    idx = GraphIndex()
    city = city_with_category("food")
    if city is None:
        pytest.skip("no city holds food POIs")
    food = idx.city_pois(city, category="food")[:8]
    itinerary = RouterAgent(idx).run(city, food, n_days=3, narrate=False)["itinerary"]
    assert len(itinerary) == 3


def test_the_catalogue_holds_only_places_a_traveller_can_visit():
    """46 of 700 catalogue rows were entities documented like a landmark that
    nobody can visit -- 25 universities and a hospital filed as `culture`, 15
    metro and mainline stations filed as `landmark`, a television channel, a
    radio station and the 2019 fire at Notre-Dame. They reached itineraries:
    5 of 14 city x archetype plans contained at least one, and a Budget
    Backpacker's Paris culture day opened Sorbonne University -> Sciences Po
    (#65).

    `data/dropped_pois.csv` records each removal with its Wikidata types, so
    the decision stays auditable without a network call.
    """
    catalogue = pd.read_csv(DATA_DIR / "poi.csv")
    dropped = pd.read_csv(DATA_DIR / "dropped_pois.csv")

    assert not set(catalogue.poi_id) & set(dropped.poi_id), \
        "a row recorded as dropped is still in the catalogue"
    assert dropped.drop_reason.notna().all(), "every drop must record its reason"
    assert set(dropped.drop_reason) <= set(NOT_A_SIGHT_TYPES), \
        f"unknown drop reason: {set(dropped.drop_reason) - set(NOT_A_SIGHT_TYPES)}"

    # poi_id is the join key the graph and every retriever use; a rebuild that
    # renumbered the survivors would silently invalidate dropped_pois.csv.
    assert catalogue.poi_id.is_unique


def test_a_disqualifying_type_only_counts_when_it_is_the_whole_story():
    """The rule has to remove an institution without taking a real place that
    merely carries an administrative second type, and REPORT.md section 5's
    argument applies: an entity ending is not its site being gone (#65)."""
    from roamwise.pipeline.common import not_a_sight

    # Removed: the disqualifying type is all there is.
    assert not_a_sight(["comprehensive university", "public research university"])
    assert not_a_sight(["university hospital", "medical school"])
    assert not_a_sight(["multi-level interchange railway station", "central station"])
    assert not_a_sight(["structure fire"])
    # "television station" is not a railway; it must not rescue itself either.
    assert not_a_sight(["television channel", "television station"])
    assert not_a_sight(["broadcaster", "radio station"])

    # Kept: a real place type survives the disqualifying one.
    assert not_a_sight(["art museum", "nonprofit organization"]) is None
    assert not_a_sight(["museum", "event venue"]) is None
    assert not_a_sight(["urban park", "event venue", "public garden"]) is None
    assert not_a_sight(["flea market", "business", "shopping center"]) is None
    assert not_a_sight(["railway station", "palace"]) is None

    # Kept: a heritage listing overrules any disqualifying type at all.
    assert not_a_sight(["cordon", "destroyed building or structure"],
                       has_heritage_listing=True) is None
    assert not_a_sight(["gallows", "destroyed building or structure"]) == "demolished"


def test_the_city_guide_counts_match_the_catalogue_it_describes():
    """The guides are generated from poi.csv and state exact counts, and they
    enter the retrieval corpus as the `guide::<CITY>` document -- so a stale
    guide is a retrievable false statement about the catalogue. Berlin's said
    "300 stops" and anchored the city's central cluster on a university (#65).
    """
    catalogue = pd.read_csv(DATA_DIR / "poi.csv")
    for code in CITY_CODES:
        guide = DATA_DIR / "city_guides" / f"{code}.txt"
        if not guide.exists():
            continue
        text = guide.read_text()
        n = len(catalogue[catalogue.destination_id == code])
        assert f"{n} stops" in text, \
            f"{code} guide does not state the catalogue's real size ({n}); regenerate " \
            f"it with `cd roamwise/pipeline && python city_guide.py --write`"


def test_fusion_beats_hybrid_on_archetype_grounding():
    fr = FusionRetriever()
    city = city_with_category("nightlife") or MAIN_CITY
    fusion_results = fr.retrieve("things to do", config="fusion", destination_id=city,
                                 archetype="Nightlife Seeker", top_k=8)
    fusion_categories = {r.get("text", "") for r in fusion_results}
    # The old assertion also accepted "trastevere", a Rome neighbourhood -- a
    # content literal that cannot survive a change of destination. What this
    # test measures is archetype grounding, and that lives in the first half.
    assert any("nightlife" in t.lower() for t in fusion_categories), \
        f"Nightlife Seeker retrieval for {city} surfaced no nightlife text"


def test_routing_respects_time_budget():
    idx = GraphIndex()
    pois = idx.city_pois(MAIN_CITY)
    result = optimize_day_route(pois, daily_minutes_budget=300)
    assert result["total_minutes"] <= 300 + 1e-6


def test_routing_skips_poi_closed_for_the_rest_of_the_day():
    poi = {"name": "Late Museum", "lat": 41.0, "lon": 29.0, "avg_visit_minutes": 60,
           "open_hour": 9, "close_hour": 18}
    result = optimize_day_route([poi], day_start_hour=23.0, daily_minutes_budget=1440)
    assert result["route"] == []


def test_routing_waits_for_poi_to_open():
    poi = {"name": "Afternoon Gallery", "lat": 41.0, "lon": 29.0, "avg_visit_minutes": 60,
           "open_hour": 14, "close_hour": 20}
    result = optimize_day_route([poi], day_start_hour=9.0, daily_minutes_budget=1440)
    assert [p["name"] for p in result["route"]] == ["Afternoon Gallery"]
    # 9:00 -> waits until 14:00 -> 60min visit -> finishes 15:00 = 360 elapsed minutes,
    # not just the 60-minute visit duration, proving the wait is accounted for.
    assert result["total_minutes"] == 360


# --- issue #59: the day's time model (start hour, length, category timing) ---

def _nightlife_poi(name, lat, lon, open_hour=18, close_hour=2):
    """Defaults mirror the catalogue's well-sourced venues (18:00-02:00)."""
    return {"name": name, "lat": lat, "lon": lon, "avg_visit_minutes": 60,
            "category": "nightlife", "open_hour": open_hour, "close_hour": close_hour}


def test_nightlife_is_not_the_first_stop_of_the_day():
    """Issue #59: 2-opt orders on geography alone, so a bar whose OSM hours
    start early (Bijou Bar opens at 07:00) was legitimately "open" at 09:00
    and won the opening slot. Measured over 32 days, 6 began with one."""
    # The bar sits right next to the hub, so pure geography would open with it.
    bar = _nightlife_poi("Corner Bar", 48.2005, 16.370, open_hour=7, close_hour=23)
    sights = _sights_along_a_line(3, lat=48.21, minutes=60)
    hub = {"lat": 48.20, "lon": 16.37, "name": "hub"}

    result = optimize_day_route([bar] + sights, start_hub=hub,
                                 daily_minutes_budget=13 * 60, day_start_hour=9.0)

    names = [p["name"] for p in result["route"]]
    assert names, "the day should not come back empty"
    assert names[0] != "Corner Bar", f"day opened with a bar: {names}"


def test_nightlife_is_never_scheduled_before_the_evening():
    """Reordering alone is not enough: on a short day the bar would simply be
    last among early stops, still at 15:00. The category carries its own
    earliest sensible hour."""
    bar = _nightlife_poi("Corner Bar", 48.2005, 16.370, open_hour=7, close_hour=23)
    sights = _sights_along_a_line(2, lat=48.21, minutes=60)
    hub = {"lat": 48.20, "lon": 16.37, "name": "hub"}

    result = optimize_day_route([bar] + sights, start_hub=hub,
                                 daily_minutes_budget=13 * 60, day_start_hour=9.0)

    for poi, slot in zip(result["route"], result["schedule"]):
        if poi.get("category") == "nightlife":
            assert slot["arrival"] >= NIGHTLIFE_EARLIEST_HOUR, \
                f"{poi['name']} scheduled at {slot['arrival']:.2f}"


def test_a_day_too_short_to_reach_the_evening_gets_no_nightlife():
    """The honest consequence of the rule above: an 8-hour day from 09:00 ends
    at 17:00, so it drops the bar rather than scheduling a 15:00 club visit."""
    bar = _nightlife_poi("Corner Bar", 48.2005, 16.370, open_hour=7, close_hour=23)
    hub = {"lat": 48.20, "lon": 16.37, "name": "hub"}

    result = optimize_day_route([bar], start_hub=hub,
                                 daily_minutes_budget=8 * 60, day_start_hour=9.0)

    assert result["route"] == []


def test_a_long_day_still_gets_its_evening_stop():
    """Regression guard for the fix's own side effect. The fill passes route
    against the day minus the meal reserve, which on a 12-hour day from 09:00
    stops right where NIGHTLIFE_EARLIEST_HOUR begins -- so once bars could no
    longer be scheduled in the morning, they stopped being scheduled at all
    and a nightlife-heavy day collapsed. _ensure_evening_stops gives them the
    same full-budget pass meals get."""
    zones = {0: _sights_along_a_line(3, minutes=60)}
    bars = [_nightlife_poi(f"Bar {i}", 48.201 + i * 0.002, 16.371) for i in range(3)]
    hub = {"lat": 48.20, "lon": 16.37, "name": "hub"}

    day = build_multi_day_itinerary(
        *_flat({0: zones[0] + bars}), start_hub=hub, daily_minutes_budget=12 * 60,
        day_start_hour=9.0, food_pois=[], min_food_per_day=0)[0]

    evening = [(p, s) for p, s in zip(day["route"], day["schedule"])
               if p.get("category") == "nightlife"]
    assert evening, f"a 12h day reaching past 18:00 should fit a bar: {[p['name'] for p in day['route']]}"
    assert all(s["arrival"] >= NIGHTLIFE_EARLIEST_HOUR for _, s in evening)
    assert day["route"][-1].get("category") == "nightlife", "the bar should close the day"


def test_day_start_hour_reaches_the_router_from_plan_trip():
    """day_start_hour sat only on RouterAgent.run()'s signature with nothing
    able to pass it, so every itinerary began at 09:00 (issue #59)."""
    orch = RoamWiseOrchestrator()
    prefs = {"budget": 0.6, "culture": 0.9, "nature": 0.2, "nightlife": 0.2, "relax": 0.3, "adventure": 0.2}

    late = orch.plan_trip(prefs, destination_id=MAIN_CITY, n_days=2,
                          day_start_hour=11.0, daily_minutes_budget=12 * 60)

    first_arrivals = [d["schedule"][0]["arrival"] for d in late["routing"]["itinerary"] if d["schedule"]]
    assert first_arrivals, "expected at least one scheduled stop"
    assert all(a >= 11.0 for a in first_arrivals), \
        f"a day started before the requested 11:00: {first_arrivals}"


def test_routing_falls_back_when_no_street_network_covers_the_points(monkeypatch):
    import roamwise.optimization.routing as routing_module
    monkeypatch.setattr(routing_module, "fetch_distance_duration_matrix",
                        lambda points, profile="foot": None)

    idx = GraphIndex()
    pois = idx.city_pois(MAIN_CITY)[:4]
    result = optimize_day_route(pois, use_real_routing=True)
    assert result["used_real_routing"] is False
    assert len(result["route"]) > 0  # still produced a usable route via the haversine fallback


def test_routing_uses_injected_real_routing_matrix(monkeypatch):
    import roamwise.optimization.routing as routing_module

    def fake_matrix(points, profile="foot"):
        n = len(points)
        # every pairwise "real" distance/duration is a fixed, distinguishable
        # value the haversine fallback could never produce, so we can tell
        # the matrix path (not haversine) actually supplied the numbers.
        return [[7.0] * n for _ in range(n)], [[42.0] * n for _ in range(n)]

    monkeypatch.setattr(routing_module, "fetch_distance_duration_matrix", fake_matrix)

    idx = GraphIndex()
    pois = idx.city_pois(MAIN_CITY)[:2]
    result = optimize_day_route(pois, use_real_routing=True, daily_minutes_budget=1440,
                                 respect_opening_hours=False)
    assert result["used_real_routing"] is True
    assert result["distance_km"] == 7.0  # single leg from the fake matrix, not a haversine value


def _daytime_poi(name, lat, lon, minutes=60):
    return {"name": name, "lat": lat, "lon": lon, "avg_visit_minutes": minutes,
            "open_hour": 8, "close_hour": 20}


# --- issue #19: day balance, budget filling, and travel modes ---

def test_balanced_zoning_evens_out_day_zones():
    """Plain KMeans collapses a dense city centre into one zone and leaves
    outlying sights in zones of their own -- one packed day, several starved
    ones. Balanced assignment has to narrow that spread without losing POIs.

    Measured across cities rather than on one fixed sample: whether plain
    KMeans happens to be balanced already depends on the exact coordinates of
    whichever POIs the current poi.csv pull put first, and a refresh of that
    real data turned the single-city version of this test red without anything
    in the zoner changing."""
    idx = GraphIndex()
    cities = CITY_CODES

    def spread(pois, balanced):
        sizes = [len(v) for v in POIZoner().zone(pois, n_zones=5, balanced=balanced).values()]
        assert sum(sizes) == len(pois)  # nothing dropped either way
        return max(sizes) - min(sizes), min(sizes)

    strictly_better = 0
    for city in cities:
        pois = idx.city_pois(city)[:11]
        balanced_spread, balanced_min = spread(pois, True)
        plain_spread, _ = spread(pois, False)

        assert balanced_min >= 1, f"{city}: no zone may be left empty"
        assert balanced_spread <= plain_spread, f"{city}: balancing made the spread worse"
        strictly_better += balanced_spread < plain_spread

    # max(1, ...) keeps the claim meaningful on a small destination list: with
    # two cities, len//2 == 1 anyway, but a one-city catalogue must still show
    # the improvement somewhere rather than passing on an empty requirement.
    assert strictly_better >= max(1, len(cities) // 2), \
        "balancing should visibly help in most cities, not merely never hurt"


def test_multi_day_itinerary_fills_every_day_near_its_budget():
    """The reported symptom: a multi-day plan where some days were empty and
    others held a two-hour route. Every day must now come back non-empty and
    reasonably full."""
    idx = GraphIndex()
    pois = idx.city_pois(MAIN_CITY)[:40]
    zones = POIZoner().zone(pois, n_zones=5)
    days = build_multi_day_itinerary(*_flat(zones), daily_minutes_budget=480)

    assert len(days) == 5
    assert all(day["route"] for day in days), "no day may come back empty"
    assert all(day["total_minutes"] >= 240 for day in days), \
        "every day should reach at least half its 8-hour budget"
    assert all(day["total_minutes"] <= 480 for day in days)


def test_day_is_not_stranded_by_a_poi_that_cannot_fit_its_window():
    """The actual cause of the empty days: a zone whose only POI was a
    nightlife venue opening at 18:00, against a 09:00-17:00 day. That day
    used to render as 'no stops fit the time budget' while other zones had
    spare POIs. The fill pass must rescue it."""
    club = {"name": "Club", "lat": 48.20, "lon": 16.37, "avg_visit_minutes": 120,
            "open_hour": 18, "close_hour": 2}
    spares = [_daytime_poi(f"Museum {i}", 48.21 + i * 0.002, 16.36) for i in range(4)]
    zones = {0: [club], 1: spares}

    days = build_multi_day_itinerary(*_flat(zones), daily_minutes_budget=480, day_start_hour=9.0)

    assert all(day["route"] for day in days), "the club's day should be filled from the pool"
    assert club not in days[0]["route"]  # still correctly refused: it never opens in the window


def test_driving_mode_fits_a_stop_walking_cannot_reach():
    """Same two stops, same budget: 10km is over two hours on foot and blows
    the budget, but is a short drive. The mode has to change the outcome."""
    far_apart = [_daytime_poi("A", 48.20, 16.37), _daytime_poi("B", 48.29, 16.37)]

    walking = optimize_day_route(far_apart, daily_minutes_budget=200, travel_mode="walking")
    driving = optimize_day_route(far_apart, daily_minutes_budget=200, travel_mode="driving")

    assert len(walking["route"]) == 1
    assert len(driving["route"]) == 2


def test_hybrid_mode_walks_short_legs_and_drives_long_ones():
    walking, driving, hybrid = (get_travel_mode(m) for m in ("walking", "driving", "hybrid"))
    short_km, long_km = 0.4, 6.0

    assert hybrid.leg_minutes(short_km) == walking.leg_minutes(short_km)
    assert hybrid.leg_minutes(long_km) == driving.leg_minutes(long_km)
    assert hybrid.leg_minutes(long_km) < walking.leg_minutes(long_km)


def test_driving_mode_requests_the_car_network_profile(monkeypatch):
    """Real routing must price a drive on the road network, not on footpaths."""
    import roamwise.optimization.routing as routing_module
    requested = []

    def fake_matrix(points, profile="foot"):
        requested.append(profile)
        n = len(points)
        return [[1.0] * n for _ in range(n)], [[5.0] * n for _ in range(n)]

    monkeypatch.setattr(routing_module, "fetch_distance_duration_matrix", fake_matrix)

    pois = [_daytime_poi("A", 48.20, 16.37), _daytime_poi("B", 48.21, 16.38)]
    optimize_day_route(pois, use_real_routing=True, travel_mode="driving")
    assert requested == ["car"]

    requested.clear()
    optimize_day_route(pois, use_real_routing=True, travel_mode="hybrid")
    assert sorted(requested) == ["car", "foot"]  # hybrid needs both to choose per leg


# --- issue #20: every day gets meals ---

def _food_poi(name, lat, lon, minutes=60):
    return {"name": name, "lat": lat, "lon": lon, "avg_visit_minutes": minutes,
            "category": "food", "open_hour": 8, "close_hour": 22}


def _sights_along_a_line(n, lat=48.20, lon=16.37, step=0.004, minutes=90):
    return [_daytime_poi(f"Sight {i}", lat + i * step, lon, minutes) for i in range(n)]


def _meals_in(day):
    return [p for p in day["route"] if p.get("category") == "food"]


def test_every_day_gets_its_minimum_meals():
    """A day with enough sightseeing content to actually span the day (unlike
    the sparser #29 scenarios below) should still get both meals -- the
    sightseeing floor added for #29 only trades a meal away when the day
    genuinely can't fit both 3h apart, not by default."""
    zones = {0: _sights_along_a_line(6, minutes=60), 1: _sights_along_a_line(6, lat=48.24, minutes=60)}
    food = [_food_poi(f"Cafe {i}", 48.20 + i * 0.006, 16.372) for i in range(8)]

    days = build_multi_day_itinerary(*_flat(zones), daily_minutes_budget=480,
                                      food_pois=food, min_food_per_day=2)

    assert all(len(_meals_in(day)) >= 2 for day in days)


def test_no_meals_are_added_when_the_minimum_is_zero():
    """Control for the test above: without the guarantee these itineraries
    contain no food at all, which is exactly the bug #20 reported."""
    zones = {0: _sights_along_a_line(4), 1: _sights_along_a_line(4, lat=48.24)}
    food = [_food_poi(f"Cafe {i}", 48.20 + i * 0.006, 16.372) for i in range(8)]

    days = build_multi_day_itinerary(*_flat(zones), daily_minutes_budget=480,
                                      food_pois=food, min_food_per_day=0)

    assert all(not _meals_in(day) for day in days)


def test_meals_are_spread_across_the_day_not_stacked_at_the_start():
    """Two meals before 11am would satisfy a naive count but is not a day
    anyone would travel. They have to straddle the day's midpoint, and (#29)
    land close to MIN_MEAL_GAP_FRACTION's ~3h-on-a-full-day target, not just
    the old, easily-gamed ">= 2 hours"."""
    zones = {0: _sights_along_a_line(8, minutes=60)}
    food = [_food_poi(f"Cafe {i}", 48.20 + i * 0.005, 16.372) for i in range(8)]

    day = build_multi_day_itinerary(*_flat(zones), daily_minutes_budget=600,
                                     day_start_hour=9.0, food_pois=food,
                                     min_food_per_day=2)[0]

    meal_hours = sorted(slot["arrival"] for poi, slot in zip(day["route"], day["schedule"])
                        if poi.get("category") == "food")
    assert len(meal_hours) >= 2
    assert meal_hours[0] >= 11.0, "the first meal should not be breakfast-time"
    assert meal_hours[-1] - meal_hours[0] >= 2.5, "meals should be close to 3 hours apart"


# --- issue #29: meal placement was clustering meals together and, in the
# sparsest days, displacing every sightseeing stop to fit them ---

def test_meals_do_not_land_back_to_back():
    """The originally-reported #29 shape: a short, sparse day where cheapest-
    insertion used to converge both meals on the same end-of-day slot."""
    zones = {0: _sights_along_a_line(3)}
    food = [_food_poi(f"Cafe {i}", 48.20 + i * 0.005, 16.372) for i in range(8)]

    day = build_multi_day_itinerary(*_flat(zones), daily_minutes_budget=480,
                                     day_start_hour=9.0, food_pois=food,
                                     min_food_per_day=2)[0]

    meal_hours = sorted(slot["arrival"] for poi, slot in zip(day["route"], day["schedule"])
                        if poi.get("category") == "food")
    if len(meal_hours) >= 2:
        assert meal_hours[-1] - meal_hours[0] >= 1.0, \
            "two meals within an hour of each other is not two separate meals"


def test_a_sparse_day_keeps_at_least_one_sightseeing_stop():
    """The #29 comment's finding: a zone with barely more sightseeing content
    than meals could lose every non-food stop to meal displacement, leaving
    a day that is literally nothing but food. A day may drop to 1 meal
    instead (see test above / MIN_SIGHTSEEING_STOPS's docstring), but it may
    never lose its last sight to make room for a second one."""
    zones = {0: _sights_along_a_line(2)}
    food = [_food_poi(f"Cafe {i}", 48.20 + i * 0.005, 16.372) for i in range(8)]

    day = build_multi_day_itinerary(*_flat(zones), daily_minutes_budget=480,
                                     day_start_hour=9.0, food_pois=food,
                                     min_food_per_day=2)[0]

    non_food = [p for p in day["route"] if p.get("category") != "food"]
    assert len(non_food) >= 1, "a day must never be emptied down to food-only"


def test_meal_choice_prefers_a_venue_on_the_route_over_a_distant_one():
    """AC2: the meal has to be somewhere the traveler is already passing."""
    zones = {0: _sights_along_a_line(4)}
    on_route = _food_poi("Corner Bistro", 48.206, 16.3705)
    far_away = _food_poi("Distant Grill", 48.35, 16.60)

    day = build_multi_day_itinerary(*_flat(zones), daily_minutes_budget=480,
                                     food_pois=[far_away, on_route], min_food_per_day=1)[0]

    names = [p["name"] for p in _meals_in(day)]
    assert "Corner Bistro" in names
    assert "Distant Grill" not in names


def test_the_same_restaurant_is_not_booked_twice_in_one_trip():
    zones = {0: _sights_along_a_line(3), 1: _sights_along_a_line(3, lat=48.23),
             2: _sights_along_a_line(3, lat=48.26)}
    food = [_food_poi(f"Cafe {i}", 48.20 + i * 0.008, 16.372) for i in range(10)]

    days = build_multi_day_itinerary(*_flat(zones), daily_minutes_budget=480,
                                      food_pois=food, min_food_per_day=2)

    booked = [p["name"] for day in days for p in _meals_in(day)]
    assert len(booked) == len(set(booked))


def test_router_agent_feeds_every_day_from_the_citys_own_restaurants():
    idx = GraphIndex()
    agent = RouterAgent(idx)
    pois = [p for p in idx.city_pois(MAIN_CITY) if p.get("category") != "food"][:24]

    result = agent.run(MAIN_CITY, pois, n_days=3, daily_minutes_budget=480)

    assert len(result["itinerary"]) == 3
    for day in result["itinerary"]:
        # >= 1, not >= 2: on real (unpredictable, occasionally sparse) city
        # data, #29's sightseeing floor can legitimately trade the second
        # meal away on a short day. The invariant this test actually guards
        # -- meals are sourced from the graph at all -- only needs >= 1.
        assert len(_meals_in(day)) >= 1, f"day {day['day']} has no meals"


# --- issue #70: opening hours are a rule over days, not one open/close pair ---

MONDAY = datetime.date(2026, 9, 7)
TUESDAY = datetime.date(2026, 9, 8)


def _tagged_poi(name, tag, lat=48.20, lon=16.37, minutes=60):
    """A POI carrying a real OSM opening_hours tag. open_hour/close_hour are
    deliberately wide and wrong-for-some-days: that is exactly the coarse pair
    the tag has to override, and leaving them permissive proves the tag is what
    the decision is being made on."""
    return {"name": name, "lat": lat, "lon": lon, "avg_visit_minutes": minutes,
            "open_hour": 0, "close_hour": 24, "opening_hours_raw": tag}


def test_a_poi_shut_on_mondays_is_not_scheduled_on_a_monday():
    """The reported shape of #70. `Tu-Su 10:00-18:00; Mo off` collapsed to the
    pair (10, 18), which is indistinguishable from "open 10-18 every day" -- so
    57 POIs in the shipped catalogue, the Musée d'Orsay and the Catacombs among
    them, were schedulable on a day they are shut."""
    museum = _tagged_poi("Closed Mondays", "Tu-Su 10:00-18:00; Mo off")

    monday = optimize_day_route([museum], daily_minutes_budget=600,
                                 day_start_hour=9.0, day_date=MONDAY)
    tuesday = optimize_day_route([museum], daily_minutes_budget=600,
                                  day_start_hour=9.0, day_date=TUESDAY)

    assert monday["route"] == []
    assert [p["name"] for p in tuesday["route"]] == ["Closed Mondays"]
    assert tuesday["schedule"][0]["arrival"] == 10.0


def test_without_a_date_the_coarse_pair_still_decides():
    """The fallback has to stay intact: callers that pass no date -- and rows
    OSM never described -- keep the pre-#70 behaviour rather than losing their
    hours entirely."""
    museum = _tagged_poi("Closed Mondays", "Tu-Su 10:00-18:00; Mo off")
    museum["open_hour"], museum["close_hour"] = 10, 18

    result = optimize_day_route([museum], daily_minutes_budget=600, day_start_hour=9.0)

    assert [p["name"] for p in result["route"]] == ["Closed Mondays"]


def test_an_unparseable_tag_falls_back_instead_of_dropping_the_stop():
    """27 of the catalogue's 4,404 distinct tags are malformed OSM. A tag we
    cannot read is not evidence that the place is shut."""
    poi = _tagged_poi("Bad Tag", "Mar-Dim 10:00-17:00")     # French day names
    poi["open_hour"], poi["close_hour"] = 10, 17

    result = optimize_day_route([poi], daily_minutes_budget=600,
                                 day_start_hour=9.0, day_date=MONDAY)

    assert [p["name"] for p in result["route"]] == ["Bad Tag"]


def test_a_lunch_closure_is_respected_rather_than_averaged_over():
    """17% of the catalogue's tags close for lunch. The pair kept the first
    stretch and dropped the second, so an afternoon visit was priced against
    morning hours."""
    poi = _tagged_poi("Lunch Closer", "Mo-Su 09:00-12:00,14:00-18:00", minutes=60)

    result = optimize_day_route([poi], daily_minutes_budget=600,
                                 day_start_hour=12.5, day_date=MONDAY)

    # 12:30 is inside the closure, so the visit waits for the afternoon session
    # rather than starting immediately or being dropped.
    assert result["schedule"][0]["arrival"] == 14.0


def test_each_day_of_a_trip_is_resolved_against_its_own_date():
    """Day 2 is a different weekday from day 1, and the router has to know it."""
    zones = {0: [_tagged_poi("Mon only", "Mo 10:00-18:00")],
             1: [_tagged_poi("Tue only", "Tu 10:00-18:00", lat=48.24)]}

    days = build_multi_day_itinerary(*_flat(zones), daily_minutes_budget=600, day_start_hour=9.0,
                                      start_date=MONDAY)

    assert [p["name"] for p in days[0]["route"]] == ["Mon only"]
    assert [p["name"] for p in days[1]["route"]] == ["Tue only"]
    assert days[0]["date"] == MONDAY and days[1]["date"] == TUESDAY


def test_the_catalogue_carries_its_opening_hours_tags_through_the_graph():
    """The column has to survive into the graph the router actually reads --
    build_graph copies an explicit column list, so a new column is invisible
    until it is named there."""
    idx = GraphIndex()
    pois = [p for city in CITY_CODES for p in idx.city_pois(city)]
    tagged = [p for p in pois if (p.get("opening_hours_raw") or "").strip()]

    assert len(tagged) > 100, "the graph should carry OSM's opening_hours verbatim"
    # And the tags are the grammar, not a re-rendered pair.
    assert any(";" in p["opening_hours_raw"] or "," in p["opening_hours_raw"]
               for p in tagged)


# --- issue #61: closing times past midnight, and when a day should begin ---

def test_a_venue_open_past_midnight_is_not_treated_as_shutting_at_midnight():
    """close_hour < open_hour means the venue spans midnight. Clamping it to
    24.0 was documented as a known limitation when #59 made days long enough
    to reach 06:00; this is the follow-up. Without it a 01:00 arrival at a bar
    open until 02:00 is 'closed'."""
    club = {"name": "Matrix", "lat": 41.0, "lon": 29.0, "avg_visit_minutes": 60,
            "open_hour": 22, "close_hour": 7}

    # A day running 12:00-06:00, arriving well after midnight.
    result = optimize_day_route([club], day_start_hour=12.0, daily_minutes_budget=18 * 60)

    assert [p["name"] for p in result["route"]] == ["Matrix"]
    assert result["schedule"][0]["arrival"] == 22.0


def test_a_long_day_reaches_stops_a_shorter_one_cannot():
    """The clamp made extra hours worthless past midnight: a 15-hour and an
    18-hour day from 12:00 both stopped at 23:49, because anything later was
    considered shut whatever its stated closing time."""
    hub = {"name": "hub", "lat": 41.0, "lon": 29.0}
    bars = [{"name": f"Bar {i}", "lat": 41.0 + i * 0.002, "lon": 29.0,
             "avg_visit_minutes": 120, "open_hour": 18, "close_hour": 2,
             "category": "nightlife"} for i in range(5)]

    shorter = optimize_day_route(bars, start_hub=hub, day_start_hour=12.0,
                                  daily_minutes_budget=10 * 60)
    longer = optimize_day_route(bars, start_hub=hub, day_start_hour=12.0,
                                daily_minutes_budget=16 * 60)

    assert len(longer["route"]) > len(shorter["route"])
    assert max(s["arrival"] for s in longer["schedule"]) >= 24.0, \
        "the extra hours have to actually reach past midnight"


def test_day_start_defaults_to_the_archetypes_own_hour():
    assert start_hour_for("Nightlife Seeker") == DAY_START_HOURS["Nightlife Seeker"]
    assert start_hour_for("Nightlife Seeker") > start_hour_for("Culture Enthusiast"), \
        "a night out starts later than a museum day"
    # An unknown archetype must not crash; it gets the ordinary morning.
    assert 6.0 <= start_hour_for("Something New") <= 12.0
    # The traveler's own choice always wins.
    assert start_hour_for("Nightlife Seeker", override=8.0) == 8.0


def test_a_nightlife_trip_does_not_come_back_holding_one_bar():
    """The reported symptom. Nightlife is never scheduled before 18:00 (#59),
    so a day opening at 09:00 has nine hours with nothing schedulable in them
    and returns a single stop. Letting the archetype set the start is what
    makes the day usable, not more retrieval or longer days."""
    idx = GraphIndex()
    agent = RouterAgent(idx)
    pois = [p for p in idx.city_pois(MAIN_CITY) if p.get("category") == "nightlife"]
    if len(pois) < 6:
        pytest.skip(f"{MAIN_CITY} has too few nightlife POIs to test the shape")

    morning = agent.run(MAIN_CITY, pois, n_days=3, daily_minutes_budget=12 * 60,
                        day_start_hour=9.0, narrate=False)
    archetype_led = agent.run(MAIN_CITY, pois, n_days=3, daily_minutes_budget=12 * 60,
                              archetype="Nightlife Seeker", narrate=False)

    morning_stops = sum(len(d["route"]) for d in morning["itinerary"])
    led_stops = sum(len(d["route"]) for d in archetype_led["itinerary"])
    assert led_stops > morning_stops, \
        f"archetype start should beat a 09:00 one ({led_stops} vs {morning_stops})"
    assert all(len(d["route"]) >= 2 for d in archetype_led["itinerary"]), \
        "no day should come back holding a single stop"


def test_every_day_reports_where_its_time_went():
    idx = GraphIndex()
    agent = RouterAgent(idx)
    pois = idx.city_pois(MAIN_CITY)[:24]

    result = agent.run(MAIN_CITY, pois, n_days=2, daily_minutes_budget=12 * 60, narrate=False)

    for day in result["itinerary"]:
        # The span is what the clock did; active is what the traveler did with
        # it. The fill/rebalance passes rank days on active, so a breakdown
        # that didn't reconcile would mean they were ranking on nothing real.
        assert day["active_minutes"] + day["idle_minutes"] == day["total_minutes"]
        assert day["active_minutes"] <= day["total_minutes"]

def test_orchestrator_end_to_end():
    orch = RoamWiseOrchestrator()
    prefs = {"budget": 0.7, "culture": 0.3, "nature": 0.2, "nightlife": 0.9, "relax": 0.2, "adventure": 0.3}
    result = orch.plan_trip(prefs, n_days=2)
    assert result["archetype"] == "Nightlife Seeker"
    assert result["destination_id"] in orch.destinations.destination_id.values
    assert len(result["routing"]["itinerary"]) == 2
    assert result["final_plan"]


# --- issues #56 / #57: the final narrative must describe the itinerary and
# nothing else, and must not cost a generation per intermediate agent ---

def _routed_names(result):
    return {p["name"] for day in result["routing"]["itinerary"] for p in day["route"]}


def _retrieved_names(result):
    return {r["name"] for r in result["fusion_rag"]["results"] if r.get("name")}


def ungrounded_places(result) -> list[str]:
    """Retrieved-but-unrouted places the narrative names without cover.

    "Without cover" is the operative part. A stop's own grounded description
    may legitimately name a neighbour -- the Humboldt Forum's says it sits on
    Museum Island, so a narrative describing that stop says "Museum Island"
    while recommending nothing off-route. Flagging that would make the check
    fire on correct output. What must never happen is a place appearing that
    the model was never shown: that is either an invention or a leak from the
    candidate list issue #56 removed.
    """
    off_route = _retrieved_names(result) - _routed_names(result)
    shown = result["routing"]["facts"]
    return sorted(name for name in off_route
                  if name in result["final_plan"] and name not in shown)


@pytest.mark.parametrize("city", CITY_CODES)
def test_synthesis_prompt_offers_no_place_outside_the_itinerary(city):
    """Issue #56: retrieval returns candidates, and the router drops most of
    them on opening hours and the time budget. Handing both lists to the
    narrator made it recommend stops the route never contained.

    Under TemplateLLMClient `final_plan` is the prompt verbatim, so asserting
    on it asserts on exactly what a real model would be shown -- the property
    that actually has to hold, and one no local-model download is needed to
    check. Every city is swept because whether a candidate survives routing
    is a per-city accident of geography and opening hours.
    """
    orch = RoamWiseOrchestrator()
    prefs = {"budget": 0.6, "culture": 0.9, "nature": 0.2, "nightlife": 0.2, "relax": 0.3, "adventure": 0.2}
    result = orch.plan_trip(prefs, destination_id=city, n_days=3)

    off_route = _retrieved_names(result) - _routed_names(result)
    assert off_route, f"{city}: retrieval should surface candidates the router drops -- else this proves nothing"
    assert not ungrounded_places(result), \
        f"{city}: places not in the itinerary were offered to the narrator: {ungrounded_places(result)}"


def test_planning_spends_one_generation_per_user_visible_narrative():
    """Issue #57: a trip used to cost four sequential generations, two of
    which produced text no user ever sees -- the retrieval and routing
    paraphrases existed only to be pasted into the synthesis prompt. Only the
    forecast blurb and the final plan are rendered, so only those two may
    cost a model pass."""
    from roamwise.agents.llm_client import LLMClient

    class CountingLLM(LLMClient):
        def __init__(self):
            self.calls = []

        def complete(self, system: str, prompt: str) -> str:
            self.calls.append(system)
            return prompt

    llm = CountingLLM()
    orch = RoamWiseOrchestrator(llm=llm)
    prefs = {"budget": 0.6, "culture": 0.9, "nature": 0.2, "nightlife": 0.2, "relax": 0.3, "adventure": 0.2}
    orch.plan_trip(prefs, destination_id=MAIN_CITY, n_days=3)

    assert len(llm.calls) == 2, f"expected forecast + synthesis only, got {len(llm.calls)}: {llm.calls}"


def test_stop_descriptions_are_not_cut_at_an_abbreviation():
    """Splitting on the first ". " turned "F. W. Borchardt" into "F." and
    "Pariser Platz (transl. ..." into "Pariser Platz (transl.", so a stop's
    one-line description said nothing about it."""
    from roamwise.agents.router_agent import _summarize

    assert _summarize("F. W. Borchardt is a restaurant in Berlin, open since 1853. It moved twice.") \
        .startswith(" -- F. W. Borchardt is a restaurant")
    assert "Paris Square" in _summarize(
        "Pariser Platz (transl. Paris Square) is a square in central Berlin, by the "
        "Brandenburg Gate. It is named after the French capital.")
    assert _summarize(None) == ""
    assert _summarize("") == ""


def test_itinerary_facts_describe_every_routed_stop():
    """The narrator can only describe stops from what it is given, so dropping
    the retrieval context (#56) is only safe if the itinerary carries each
    stop's own description."""
    orch = RoamWiseOrchestrator()
    prefs = {"budget": 0.6, "culture": 0.9, "nature": 0.2, "nightlife": 0.2, "relax": 0.3, "adventure": 0.2}
    result = orch.plan_trip(prefs, destination_id=MAIN_CITY, n_days=2)

    facts = result["routing"]["facts"]
    for name in _routed_names(result):
        assert name in facts, f"{name} is routed but missing from the itinerary facts"


def test_comparative_analysis_shows_fusion_advantage():
    df = run_comparative_analysis(top_k=8)
    summary = summarize(df)
    assert summary.loc["fusion", "mean_archetype_precision"] >= summary.loc["standard", "mean_archetype_precision"]
    assert summary.loc["fusion", "mean_grounded_entity_rate"] == 1.0
    assert summary.loc["standard", "mean_grounded_entity_rate"] == 0.0

    # The headline on the Results tab is generated from these verdicts, so a
    # metric silently changing category would rewrite a user-facing claim.
    verdicts = paired_significance(df).set_index(["metric", "opponent"]).verdict
    assert verdicts[("archetype_precision", "hybrid")] == "better"
    # Grounded-entity rate is 1.0 by construction for anything that retrieves,
    # so it can separate Fusion from standard prompting but never from Hybrid.
    assert verdicts[("grounded_entity_rate", "hybrid")] == "identical"
    assert verdicts[("grounded_entity_rate", "standard")] == "better"


def test_every_test_query_has_an_answer():
    """A query whose gold set is empty scores nothing and silently shrinks the
    sample. One shipped that way: Lisbon has four beach POIs and none within
    reach of a hub, so 'beach spots reachable from the airport' was graded
    against an empty key for every configuration (issue #50)."""
    idx = GraphIndex()
    empty = [q.text for q in TEST_QUERIES if not gold_for(idx, q)]
    assert not empty, f"queries with no possible correct answer: {empty}"


def test_query_set_is_powered_and_balanced_by_difficulty():
    """The comparison was run on 18 queries and at that size a real effect is
    detected about a third of the time, so the set has a floor.

    The second half used to be about circularity -- the answer key was the
    retriever's own traversal, so a query naming its category graded the
    retriever against itself. That is fixed at the source (see
    `test_the_answer_key_is_not_one_the_retriever_can_produce`), and what the
    split now guards is difficulty: a query that names its category tells the
    router where to look, and a headline number must not be carried entirely
    by that easy half (#48).
    """
    assert len(TEST_QUERIES) >= 45
    assert {q.tier for q in TEST_QUERIES} == {"handwritten", "grid"}

    name_no_category = [q for q in TEST_QUERIES if dependence_level(q) != "subset"]
    assert len(name_no_category) >= 30

    # No archetype may own the set the way Culture Enthusiast owned 44% of it.
    counts = collections.Counter(q.archetype for q in TEST_QUERIES)
    assert max(counts.values()) / len(TEST_QUERIES) < 0.40
    assert set(counts) == set(CATEGORY_AFFINITY), "every archetype should be exercised"


def test_dependence_level_spots_the_queries_that_name_their_own_category():
    query = comparative_analysis.TestQuery
    # Names the key's category next to a transport word, so the graph router
    # dispatches straight at it -- the easiest shape of query there is.
    assert dependence_level(query(MAIN_CITY, "Culture Enthusiast", ("landmark",), True,
                                  "landmarks near a transport hub")) == "subset"
    # Same category, no transport constraint -- the router's filter is wider
    # than the key rather than nested inside it.
    assert dependence_level(query(MAIN_CITY, "Culture Enthusiast", ("landmark",), False,
                                  "the best landmarks to visit")) == "superset"
    # The router never reaches for the key's category at all.
    assert dependence_level(query(MAIN_CITY, "Culture Enthusiast", ("history",), False,
                                  "somewhere calm to spend a slow afternoon")) == "independent"


def test_the_answer_key_is_not_one_the_retriever_can_produce():
    """`recall_at_k`'s answer key used to be every catalogue POI of the queried
    category -- `GraphIndex.city_pois`, which is the traversal the graph
    retriever dispatches to. On a query naming its category, retrieval was a
    subset of the key by construction, so its recall measured that it agrees
    with itself rather than that it finds good answers (#48).

    The key is now gated on Wikivoyage, which is written by travellers and owes
    nothing to Wikidata sitelinks, OSM tagging or this project's graph. The
    property that matters is that neither set contains the other: the retriever
    can return plenty of category matches and still score zero.
    """
    idx = GraphIndex()
    graph = GraphSearchIndex(idx)
    city = city_with_category("museum")
    if city is None:
        pytest.skip("no city holds museums")

    query = comparative_analysis.TestQuery(
        city, "Culture Enthusiast", ("museum",), True,
        "museums within easy reach of a transport hub")
    assert dependence_level(query) == "subset", "this test needs the easiest query shape"

    gold = gold_for(idx, query)
    assert gold, "the key must not be empty"

    # The key is a strict subset of the category, so something outside this
    # project decided which members are in it.
    every_museum = {p["poi_id"] for p in idx.city_pois(city, category="museum")}
    assert gold < every_museum, \
        "the key is every POI of the category again -- the circularity is back"

    # And the retriever's own traversal is not contained in it.
    retrieved = {r["poi_id"] for r in graph.search(query.text, top_k=40,
                                                   destination_id=city,
                                                   archetype=query.archetype)}
    assert retrieved - gold, \
        "graph retrieval falls entirely inside the key -- it is grading itself"


def test_every_recommended_poi_is_a_poi_the_catalogue_still_holds():
    """The key is committed rather than recomputed, so it can go stale against
    a catalogue that dropped rows -- which #65 did, by 46."""
    catalogue = set(pd.read_csv(DATA_DIR / "poi.csv").poi_id)
    orphans = comparative_analysis.RECOMMENDED_POIS - catalogue
    assert not orphans, (
        f"{len(orphans)} POIs in retrieval_gold.csv are no longer in the catalogue; "
        "rebuild with `cd roamwise/pipeline && python retrieval_gold.py --write`")


def _synthetic_pairs(**per_config_values) -> pd.DataFrame:
    """Long-form results frame with one row per (query, config)."""
    rows = []
    for config, columns in per_config_values.items():
        for query_id in range(10):
            rows.append({"query_id": query_id, "config": config,
                         **{column: values[query_id] for column, values in columns.items()}})
    return pd.DataFrame(rows)


def test_paired_significance_reads_direction_and_ties():
    """Wilcoxon cannot rank an all-zero difference vector, and half the metrics
    are better when *lower*. Both used to be latent ways to either crash the
    Results tab or print a backwards verdict (issue #46)."""
    df = _synthetic_pairs(
        fusion={"archetype_precision": [0.9] * 10, "retrieval_ms": [20.0] * 10,
                "grounded_entity_rate": [1.0] * 10, "km_per_stop_day1": [1.0] * 10,
                "recall_at_k": [0.5] * 10},
        hybrid={"archetype_precision": [0.5] * 10, "retrieval_ms": [5.0] * 10,
                "grounded_entity_rate": [1.0] * 10, "km_per_stop_day1": [1.0] * 10,
                "recall_at_k": [0.5] * 10},
    )
    verdicts = paired_significance(df, champion="fusion").set_index("metric")

    assert verdicts.loc["archetype_precision", "verdict"] == "better"
    # Higher latency is worse even though the number went up -- the sign has to
    # be flipped for lower-is-better metrics or this reads as a win.
    assert verdicts.loc["retrieval_ms", "verdict"] == "worse"
    assert verdicts.loc["retrieval_ms", "mean_advantage"] < 0
    # Identical columns: no test is possible, and it must not raise.
    assert verdicts.loc["grounded_entity_rate", "verdict"] == "identical"
    assert pd.isna(verdicts.loc["grounded_entity_rate", "p_value"])


def test_paired_significance_calls_a_coin_flip_no_difference():
    """A metric that trades wins back and forth must not be reported as a lead
    just because its mean happens to land higher."""
    df = _synthetic_pairs(
        fusion={"km_per_stop_day1": [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]},
        hybrid={"km_per_stop_day1": [2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.1]},
    )
    verdict = paired_significance(df, champion="fusion").set_index("metric")
    assert verdict.loc["km_per_stop_day1", "verdict"] == "no difference"


def test_langgraph_orchestrator_matches_custom_orchestrator_interface():
    """Optional: skipped in CI, where the langgraph extra isn't installed
    (see requirements-langgraph.txt). Confirms the LangGraph rewrite exposes
    the same plan_trip() shape and honors a pinned destination via its
    conditional edge (select_destination is skipped)."""
    pytest.importorskip("langgraph")
    from roamwise.agents.orchestrator_langgraph import RoamWiseLangGraphOrchestrator

    orch = RoamWiseLangGraphOrchestrator()
    prefs = {"budget": 0.7, "culture": 0.3, "nature": 0.2, "nightlife": 0.9, "relax": 0.2, "adventure": 0.3}

    result = orch.plan_trip(prefs, n_days=2)
    assert result["archetype"] == "Nightlife Seeker"
    assert result["destination_id"] in orch.destinations.destination_id.values
    assert len(result["routing"]["itinerary"]) == 2
    assert result["final_plan"]

    pinned = orch.plan_trip(prefs, destination_id=MAIN_CITY, n_days=2)
    assert pinned["destination_id"] == MAIN_CITY


def test_local_llm_is_opt_in_and_falls_back_safely(monkeypatch):
    """Issue #54: constructing LocalHuggingFaceLLMClient can trigger a
    multi-gigabyte model download, so get_default_llm_client() must never
    reach for it just because mlx-lm happens to be importable -- only on the
    explicit ROAMWISE_LOCAL_LLM opt-in. Runs regardless of whether mlx-lm is
    actually installed, unlike the test below."""
    from roamwise.agents.llm_client import TemplateLLMClient, get_default_llm_client

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ROAMWISE_LOCAL_LLM", raising=False)
    assert isinstance(get_default_llm_client(), TemplateLLMClient)


def test_local_llm_client_produces_real_generation():
    """Optional: skipped unless mlx-lm is installed (see
    requirements-local-llm.txt) -- this actually loads the model and runs
    generation, so it also downloads several GB on first run."""
    pytest.importorskip("mlx_lm")
    from roamwise.agents.llm_client import LocalHuggingFaceLLMClient

    client = LocalHuggingFaceLLMClient()
    out = client.complete(system="Reply with one short sentence.", prompt="Name a city in France.")
    assert out.strip()


# app.py is only the router since the System logs screen landed (#41); the
# import chain that reaches the agent/retrieval modules now hangs off the page
# scripts, so both have to be loaded to prove the app can actually start.
STREAMLIT_MODULES = ["roamwise.app", "roamwise.views.itinerary", "roamwise.views.system_logs"]


@pytest.mark.parametrize("module", STREAMLIT_MODULES)
def test_streamlit_app_imports(module):
    """Guards the failure mode where the whole suite is green but the app does
    not start at all: these modules' import chain reaches every agent/retrieval
    module, so a half-migrated import path breaks here instead of at launch."""
    importlib.import_module(module)


def test_internal_modules_are_loaded_under_a_single_import_path():
    """Every internal module must be reachable as `roamwise.<pkg>.<mod>` only.
    A bare `<pkg>.<mod>` entry in sys.modules means the same file was loaded
    twice under two names, which silently splits module state and breaks
    isinstance/monkeypatch across the two copies."""
    for module in STREAMLIT_MODULES:
        importlib.import_module(module)

    internal_packages = {
        "agents", "models", "optimization", "retrieval",
        "knowledge_graph", "evaluation", "data", "views",
    }
    duplicated = sorted(
        name for name in sys.modules
        if name.split(".")[0] in internal_packages
    )
    assert not duplicated, f"modules loaded outside the roamwise.* path: {duplicated}"


# Every file the README/Dockerfile/launch.json tell a user to run directly.
# Importing them the normal way (as `roamwise.<pkg>.<mod>`) proves nothing about
# these commands, because that path already has the repo root on sys.path.
DOCUMENTED_ENTRY_POINT_SCRIPTS = [
    "app.py",
    "evaluation/comparative_analysis.py",
    "evaluation/forecasting_comparison.py",
]

# Load the file the way `python <script>` does -- under a throwaway module name,
# so any `if __name__ == "__main__":` body stays unexecuted and only the imports
# and module-level code run.
_IMPORT_PROBE = (
    "import importlib.util, sys\n"
    "spec = importlib.util.spec_from_file_location('_entry_point_probe', sys.argv[1])\n"
    "spec.loader.exec_module(importlib.util.module_from_spec(spec))\n"
)


@pytest.mark.parametrize("script", DOCUMENTED_ENTRY_POINT_SCRIPTS)
def test_documented_entry_point_scripts_resolve_roamwise_imports(script):
    """Run as a script, sys.path[0] is the script's own directory, so the repo
    root -- where the `roamwise` package lives -- never enters the path unless
    the script puts it there. This is the exact reason the app used to die with
    `ModuleNotFoundError: No module named 'roamwise'` while the suite was green,
    so reproduce that sys.path instead of relying on pytest's.

    Only the roamwise import path is asserted on: these scripts may still exit
    non-zero for unrelated reasons (a missing optional dependency, no network),
    and that is not what this test is guarding.
    """
    path = Path(__file__).resolve().parents[1] / script
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}

    # cwd == the script's directory makes the child's sys.path[0] ('') resolve
    # to that directory, matching how the interpreter launches a script.
    proc = subprocess.run(
        [sys.executable, "-c", _IMPORT_PROBE, str(path)],
        cwd=path.parent, env=env, capture_output=True, text=True,
    )

    assert "No module named 'roamwise" not in proc.stderr, (
        f"`python {script}` cannot import the roamwise package:\n{proc.stderr}"
    )


# --- issue #21: map framing ---

def _visible_spans(zoom: float, center_lat: float, width_px: int, height_px: int):
    """Degrees of latitude and longitude visible at a given Web Mercator zoom.

    Mirrors what MapLibre renders: the world is 512 * 2**zoom pixels wide, and
    latitude is compressed by cos(lat) relative to longitude.
    """
    world = 512 * 2 ** zoom
    return (height_px * 360.0 / world * math.cos(math.radians(center_lat)),
            width_px * 360.0 / world)


@pytest.mark.parametrize("city", CITY_CODES)
def test_map_view_frames_every_stop_without_wasting_the_canvas(city):
    """Both halves of the framing criterion at once: no stop may fall off the
    map, and the itinerary may not sit in a small island of empty canvas.

    The heuristic this replaced compared raw latitude and longitude degrees as
    if they were the same unit, so it mis-framed north-south days by the
    Mercator factor 1/cos(lat) and filled roughly a third of the viewport.
    """
    from roamwise.views.itinerary import _fit_view, _MAP_ASSUMED_WIDTH_PX, _MAP_HEIGHT_PX

    idx = GraphIndex()
    pois = idx.city_pois(city)[:12]
    lats = [p["lat"] for p in pois]
    lons = [p["lon"] for p in pois]

    zoom, center_lat, center_lon = _fit_view(lats, lons)
    # checked at the width the fit assumes, which is deliberately the narrowest
    # this column renders at -- a wider canvas only adds margin, never clips
    lat_span, lon_span = _visible_spans(zoom, center_lat, _MAP_ASSUMED_WIDTH_PX, _MAP_HEIGHT_PX)

    assert min(lats) >= center_lat - lat_span / 2, f"{city}: a stop falls off the south edge"
    assert max(lats) <= center_lat + lat_span / 2, f"{city}: a stop falls off the north edge"
    assert min(lons) >= center_lon - lon_span / 2, f"{city}: a stop falls off the west edge"
    assert max(lons) <= center_lon + lon_span / 2, f"{city}: a stop falls off the east edge"

    fill = max((max(lats) - min(lats)) / lat_span, (max(lons) - min(lons)) / lon_span)
    assert fill > 0.5, f"{city}: itinerary fills only {fill:.0%} of the map -- zoomed too far out"


def test_map_view_centres_on_the_bounding_box_not_the_mean():
    """One outlying stop used to drag the centre toward the cluster it was
    furthest from, because the old code averaged the coordinates."""
    from roamwise.views.itinerary import _fit_view

    clustered = [41.900, 41.901, 41.902, 41.903]
    outlier = [41.960]
    lats = clustered + outlier
    lons = [12.50] * len(lats)

    _, center_lat, _ = _fit_view(lats, lons)

    assert center_lat == pytest.approx((min(lats) + max(lats)) / 2)
    assert center_lat > sum(lats) / len(lats), "mean-centring would sit inside the cluster"


def test_map_view_survives_a_single_stop():
    """A one-stop day has no extent to fit against; it must clamp rather than
    divide by zero or zoom to infinity."""
    from roamwise.views.itinerary import _fit_view, _MAP_MAX_ZOOM

    zoom, center_lat, center_lon = _fit_view([41.9], [12.5])

    assert zoom == pytest.approx(_MAP_MAX_ZOOM)
    assert (center_lat, center_lon) == (41.9, 12.5)


# --- issue #32: street distances come from the repo, not a routing server ---

def _street_haversine(a, b):
    from roamwise.optimization.routing import haversine_km
    return haversine_km(a["lat"], a["lon"], b["lat"], b["lon"])


def test_street_matrix_answers_for_the_committed_catalogue():
    """Every point the router can be handed -- POIs, hubs, the city centre it
    anchors days at -- is in the precomputed matrix, so a trip pays nothing
    for real distances."""
    from roamwise.optimization.street_network import fetch_distance_duration_matrix

    pois = GraphIndex().city_pois(MAIN_CITY)[:12]
    result = fetch_distance_duration_matrix(pois, profile="foot")

    assert result is not None, f"no committed street network covers {MAIN_CITY}"
    km, minutes = result
    for i, a in enumerate(pois):
        assert km[i][i] == 0, "a point is not zero distance from itself"
        for j, b in enumerate(pois):
            # A route along real streets, plus the set-back from each endpoint
            # to its road, can never be shorter than the straight line.
            assert km[i][j] >= _street_haversine(a, b) - 1e-6, f"{i}->{j} beats the crow"
            assert minutes[i][j] == pytest.approx(km[i][j] / 4.5 * 60)


def test_street_matrix_is_not_just_the_straight_line():
    """The whole point of #32: streets bend. If the ratio were ~1.0 the matrix
    would be a slower way to compute haversine."""
    from roamwise.optimization.street_network import fetch_distance_duration_matrix

    pois = GraphIndex().city_pois(MAIN_CITY)[:25]
    km, _ = fetch_distance_duration_matrix(pois, profile="foot")
    ratios = [km[i][j] / _street_haversine(a, b)
              for i, a in enumerate(pois) for j, b in enumerate(pois)
              if _street_haversine(a, b) > 0.2]

    assert 1.05 < sum(ratios) / len(ratios) < 1.9, "implausible detour factor"


def test_real_routing_opens_no_socket(monkeypatch):
    """This is what #32 bought. Real routing used to depend on a public OSRM
    demo server, which is why it shipped default-off; a trip planned with the
    network unplugged must now still come back on real street distances."""
    import socket

    def no_network(*args, **kwargs):
        raise AssertionError("real routing must not touch the network (#32)")

    monkeypatch.setattr(socket, "socket", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)

    pois = GraphIndex().city_pois(MAIN_CITY)[:10]
    days = build_multi_day_itinerary(pois, 2, use_real_routing=True)

    assert any(d["used_real_routing"] for d in days), "fell back to haversine"


def test_street_routing_solves_points_the_matrix_does_not_hold():
    """A coordinate that is not in the catalogue -- a pinned spot, a POI added
    since the last build -- still gets a real street distance, from the graph
    committed next to the matrix."""
    from roamwise.optimization.street_network import fetch_distance_duration_matrix

    pois = [dict(p) for p in GraphIndex().city_pois(MAIN_CITY)[:4]]
    pois[0]["lat"] += 0.0007  # ~78m away: a real place, no matrix row

    result = fetch_distance_duration_matrix(pois, profile="foot")

    assert result is not None, "the graph layer should have answered"
    km, _ = result
    assert km[0][1] >= _street_haversine(pois[0], pois[1]) - 1e-6


def test_street_routing_declines_a_city_it_holds_no_network_for():
    """Vienna was in the old eight-city dataset and is not in this one. Half a
    matrix in street distances and half in straight lines would make a day's
    total mean nothing, so the whole request is declined."""
    from roamwise.optimization.street_network import fetch_distance_duration_matrix

    vienna = [{"lat": 48.2082, "lon": 16.3738}, {"lat": 48.2000, "lon": 16.3600}]

    assert fetch_distance_duration_matrix(vienna, profile="foot") is None



# --- issue #32 stage 2: RAPTOR over the GTFS timetable ---
#
# These build timetables by hand, small enough to work out on paper. That is
# deliberate: a transit router that is subtly wrong does not crash, it returns
# plausible journeys that nobody can actually make, and the committed Paris
# matrix cannot tell you which of its 143,641 numbers are the wrong ones.

_HOUR = 3600


def _at(hour, minute=0):
    return int(hour * _HOUR + minute * 60)


def _timetable(n_stops, patterns, transfers=()):
    from roamwise.optimization.raptor import TransitTable
    return TransitTable.build(n_stops, [(stops, times, times) for stops, times in patterns],
                              list(transfers))


def _arrivals(table, origin_stop, depart, rounds=5):
    import numpy as np
    from roamwise.optimization.raptor import earliest_arrivals
    return earliest_arrivals(table, np.array([[origin_stop]]), np.array([[0]]),
                             np.array([depart]), rounds=rounds)[0]


def test_raptor_rides_two_lines_through_a_change():
    line_a = [[_at(8, 0), _at(8, 10), _at(8, 20)], [_at(8, 30), _at(8, 40), _at(8, 50)]]
    line_b = [[_at(8, 25), _at(8, 40)], [_at(9, 0), _at(9, 15)]]
    table = _timetable(5, [([0, 1, 2], line_a), ([2, 3], line_b)],
                       transfers=[(3, 4, 300), (4, 3, 300)])

    arrivals = _arrivals(table, origin_stop=0, depart=_at(7, 55))

    assert arrivals[1] == _at(8, 10)
    assert arrivals[2] == _at(8, 20)
    assert arrivals[3] == _at(8, 40), "the 08:25 connection was reachable and should be taken"
    assert arrivals[4] == _at(8, 45), "the five-minute footpath out of stop 3 was not relaxed"


def test_raptor_will_not_promise_a_connection_you_cannot_make():
    """A minute of slack is not a connection. Two minutes is the floor
    (DEFAULT_CHANGE_SECONDS) and a matrix that ignores it reads as faster than
    the city is."""
    line_a = [[_at(8, 0), _at(8, 10), _at(8, 20)]]
    departs_one_minute_later = [[_at(8, 21), _at(8, 36)], [_at(9, 0), _at(9, 15)]]
    table = _timetable(4, [([0, 1, 2], line_a), ([2, 3], departs_one_minute_later)])

    arrivals = _arrivals(table, origin_stop=0, depart=_at(7, 55))

    assert arrivals[3] == _at(9, 15), "an 08:20 arrival cannot board at 08:21"


def test_raptor_keeps_service_that_runs_past_midnight():
    """GTFS writes 00:10 on a service day as 24:10:00. Clamped or dropped at
    midnight, every night bus disappears and late evenings look unreachable."""
    night_bus = [[_at(23, 50), _at(24, 10)]]

    arrivals = _arrivals(_timetable(2, [([0, 1], night_bus)]), 0, _at(23, 0))

    assert arrivals[1] == _at(24, 10)


def test_raptor_boards_the_express_that_overtakes_the_stopper():
    """RAPTOR boards the earliest *departing* trip and never reconsiders, which
    is only optimal if trips keep their order along the pattern. The Paris feed
    breaks that on 30 patterns, and the cost is a wrong answer, not a slow
    one."""
    stopper = [_at(8, 0), _at(9, 0)]
    express = [_at(8, 10), _at(8, 30)]
    table = _timetable(2, [([0, 1], [stopper, express])])

    assert table.n_patterns == 2, "the express should have been split into its own pattern"
    assert _arrivals(table, 0, _at(7, 0))[1] == _at(8, 30)


def test_raptor_solves_every_origin_together_the_same_as_one_at_a_time():
    """The whole matrix is one batched solve; if batching changed an answer,
    every committed number would be suspect."""
    import numpy as np
    from roamwise.optimization.raptor import earliest_arrivals

    line_a = [[_at(8, 0), _at(8, 10), _at(8, 20)], [_at(8, 30), _at(8, 40), _at(8, 50)]]
    line_b = [[_at(8, 25), _at(8, 40)], [_at(9, 0), _at(9, 15)]]
    table = _timetable(4, [([0, 1, 2], line_a), ([2, 3], line_b)])
    origins = [(0, _at(7, 55)), (2, _at(8, 0)), (1, _at(8, 0))]

    together = earliest_arrivals(table, np.array([[s] for s, _ in origins]),
                                 np.zeros((len(origins), 1), dtype=int),
                                 np.array([t for _, t in origins]))

    for row, (stop, depart) in enumerate(origins):
        assert (together[row] == _arrivals(table, stop, depart)).all()


def test_raptor_leaves_unreachable_stops_unreachable():
    """Nothing runs at 23:00 in this timetable, and the honest answer is
    'you cannot get there', not a number."""
    from roamwise.optimization.raptor import INFINITY

    line_a = [[_at(8, 0), _at(8, 10)]]

    assert _arrivals(_timetable(2, [([0, 1], line_a)]), 0, _at(23, 0))[1] >= INFINITY



# --- issue #32 stage 2: the committed Paris timetable, end to end ---


def _transit_points(*names):
    """Catalogue points by name, as the router hands them to the matrix."""
    pois = pd.read_csv(DATA_DIR / "poi.csv")
    hubs = pd.read_csv(DATA_DIR / "transport.csv")
    both = pd.concat([pois[["name", "lat", "lon"]], hubs[["name", "lat", "lon"]]])
    out = []
    for name in names:
        row = both[both.name.str.contains(name, case=False, na=False)].iloc[0]
        out.append({"lat": float(row.lat), "lon": float(row.lon), "name": row["name"]})
    return out


def test_transit_is_never_slower_than_walking():
    """The matrix keeps whichever is faster, so it can never advise three
    changes to cross a square. The tolerance is float32 storage, not
    modelling: the largest excess over the whole matrix is 0.0005 seconds."""
    from roamwise.optimization.street_network import fetch_distance_duration_matrix

    points = GraphIndex().city_pois(MAIN_CITY)[:20]
    _, transit = fetch_distance_duration_matrix(points, profile="transit")
    _, walking = fetch_distance_duration_matrix(points, profile="foot")

    for i in range(len(points)):
        for j in range(len(points)):
            assert transit[i][j] <= walking[i][j] + 1e-3, f"{i}->{j} rides when walking is faster"


def test_transit_actually_beats_walking_across_town():
    """And the other direction: if the timetable never won, the matrix would
    be an expensive way to store walking times."""
    from roamwise.optimization.street_network import fetch_distance_duration_matrix

    points = _transit_points("Eiffel Tower", "Sacré-Cœur", "Gare de Lyon")
    _, transit = fetch_distance_duration_matrix(points, profile="transit")
    _, walking = fetch_distance_duration_matrix(points, profile="foot")

    for i in range(len(points)):
        for j in range(len(points)):
            if i != j:
                assert transit[i][j] < walking[i][j] * 0.8, "the timetable should win these"


def test_the_airport_transfer_stops_being_a_walk():
    """The reason stage 2 exists. With only street distances, arriving at
    Charles de Gaulle and asking to go on foot spends the better part of a day
    walking in; the timetable answers in under an hour and a half."""
    from roamwise.optimization.street_network import fetch_distance_duration_matrix

    points = _transit_points("Charles de Gaulle", "Notre-Dame de Paris")
    _, walking = fetch_distance_duration_matrix(points, profile="foot")
    _, transit = fetch_distance_duration_matrix(points, profile="transit")

    assert walking[0][1] > 5 * 60, "the walk from the airport should be absurd, and it is"
    assert transit[0][1] < 90, f"RER B and a change is not {transit[0][1]:.0f} minutes"


def test_berlin_rides_in_from_its_airport():
    """Brandenburg's coordinate is the middle of the airfield, 1,225m from the
    nearest footway and outside a village. Measuring access over the street
    network from there put the traveller on rural buses and the journey at 240
    minutes; the airport's own platforms are 785m away in a straight line."""
    from roamwise.optimization.street_network import fetch_distance_duration_matrix

    points = _transit_points("Brandenburg Airport", "Brandenburg Gate")
    _, transit = fetch_distance_duration_matrix(points, profile="transit")

    assert 30 < transit[0][1] < 75, f"FEX and the S-Bahn do not take {transit[0][1]:.0f} minutes"


def test_transit_declines_a_city_with_no_timetable():
    """Both catalogue cities ship a timetable now. Somewhere that does not has
    no transit answer at all -- and a straight line at an average speed is not
    one, so the matrix returns nothing rather than inventing a journey on
    services it knows nothing about."""
    from roamwise.optimization.street_network import fetch_distance_duration_matrix

    vienna = [{"lat": 48.2082, "lon": 16.3738}, {"lat": 48.2000, "lon": 16.3600}]

    assert fetch_distance_duration_matrix(vienna, profile="transit") is None


def test_transit_is_offered_only_where_a_timetable_ships(monkeypatch):
    from roamwise.views.itinerary import _travel_mode_options
    import roamwise.optimization.street_network as street_network

    for city in CITY_CODES:
        assert "transit" in _travel_mode_options(city), f"{city} ships a timetable"
    assert "transit" not in _travel_mode_options(None), "no city pinned, no timetable"

    # The list is driven by what actually ships, not by a literal -- the
    # destination picker outlived its dataset once already (#65).
    monkeypatch.setattr(street_network, "available_cities", lambda profile="foot": [])
    assert "transit" not in _travel_mode_options(MAIN_CITY)
    assert list(_travel_mode_options(MAIN_CITY)) == ["walking", "driving", "hybrid"]


def test_transit_reads_the_timetable_even_with_real_routing_off():
    """Every other mode has an honest estimate to fall back on. Transit does
    not, so selecting it always reads the timetable rather than quietly
    costing the day at an average speed over straight lines."""
    from roamwise.optimization.routing import _build_distance_functions

    points = _transit_points("Eiffel Tower", "Sacré-Cœur")
    _, duration, used_real = _build_distance_functions(points, use_real_routing=False,
                                                       travel_mode="transit")

    assert used_real is True
    assert duration(points[0], points[1]) < 60


def test_transit_plans_a_whole_trip():
    from roamwise.agents.orchestrator import RoamWiseOrchestrator

    prefs = {"budget": 0.6, "culture": 0.9, "nature": 0.3, "nightlife": 0.3,
             "relax": 0.4, "adventure": 0.3}
    planned = RoamWiseOrchestrator().plan_trip(prefs, destination_id=MAIN_CITY, n_days=3,
                                               travel_mode="transit", use_real_routing=True)

    days = planned["routing"]["itinerary"]
    assert all(day["used_real_routing"] for day in days), "fell back off the timetable"
    assert any(day["route"] for day in days)
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



if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
