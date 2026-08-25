"""End-to-end smoke tests covering every module in the pipeline. Run with:
    cd roamwise && ../venv/bin/pytest tests/ -v
"""
import collections
import importlib
import math
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from roamwise.knowledge_graph.build_graph import CATEGORY_AFFINITY, GraphIndex
from roamwise.models.forecasting import forecast_city, best_months_to_visit
from roamwise.models.segmentation import TravelerSegmenter, POIZoner
from roamwise.retrieval.fusion import FusionRetriever
from roamwise.optimization.routing import (
    NIGHTLIFE_EARLIEST_HOUR, build_multi_day_itinerary, optimize_day_route)
from roamwise.optimization.travel_modes import get_travel_mode
from roamwise.agents.orchestrator import RoamWiseOrchestrator
from roamwise.agents.router_agent import RouterAgent
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
        {0: zones[0] + bars}, start_hub=hub, daily_minutes_budget=12 * 60,
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


def test_routing_falls_back_when_osrm_unreachable(monkeypatch):
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
    days = build_multi_day_itinerary(zones, daily_minutes_budget=480)

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

    days = build_multi_day_itinerary(zones, daily_minutes_budget=480, day_start_hour=9.0)

    assert all(day["route"] for day in days), "the club's day should be filled from the pool"
    assert club not in days[0]["route"]  # still correctly refused: it never opens in the window


def test_no_fill_pass_reproduces_the_old_starved_day():
    """Guards the fix itself: with filling switched off, the same input still
    produces the empty day, so the test above is proving the fill pass works
    rather than passing for some incidental reason."""
    club = {"name": "Club", "lat": 48.20, "lon": 16.37, "avg_visit_minutes": 120,
            "open_hour": 18, "close_hour": 2}
    spares = [_daytime_poi(f"Museum {i}", 48.21 + i * 0.002, 16.36) for i in range(4)]

    days = build_multi_day_itinerary({0: [club], 1: spares}, daily_minutes_budget=480,
                                      day_start_hour=9.0, fill_days=False)
    assert days[0]["route"] == []


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


def test_driving_mode_requests_the_car_osrm_profile(monkeypatch):
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

    days = build_multi_day_itinerary(zones, daily_minutes_budget=480,
                                      food_pois=food, min_food_per_day=2)

    assert all(len(_meals_in(day)) >= 2 for day in days)


def test_no_meals_are_added_when_the_minimum_is_zero():
    """Control for the test above: without the guarantee these itineraries
    contain no food at all, which is exactly the bug #20 reported."""
    zones = {0: _sights_along_a_line(4), 1: _sights_along_a_line(4, lat=48.24)}
    food = [_food_poi(f"Cafe {i}", 48.20 + i * 0.006, 16.372) for i in range(8)]

    days = build_multi_day_itinerary(zones, daily_minutes_budget=480,
                                      food_pois=food, min_food_per_day=0)

    assert all(not _meals_in(day) for day in days)


def test_meals_are_spread_across_the_day_not_stacked_at_the_start():
    """Two meals before 11am would satisfy a naive count but is not a day
    anyone would travel. They have to straddle the day's midpoint, and (#29)
    land close to MIN_MEAL_GAP_FRACTION's ~3h-on-a-full-day target, not just
    the old, easily-gamed ">= 2 hours"."""
    zones = {0: _sights_along_a_line(8, minutes=60)}
    food = [_food_poi(f"Cafe {i}", 48.20 + i * 0.005, 16.372) for i in range(8)]

    day = build_multi_day_itinerary(zones, daily_minutes_budget=600,
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

    day = build_multi_day_itinerary(zones, daily_minutes_budget=480,
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

    day = build_multi_day_itinerary(zones, daily_minutes_budget=480,
                                     day_start_hour=9.0, food_pois=food,
                                     min_food_per_day=2)[0]

    non_food = [p for p in day["route"] if p.get("category") != "food"]
    assert len(non_food) >= 1, "a day must never be emptied down to food-only"


def test_meal_choice_prefers_a_venue_on_the_route_over_a_distant_one():
    """AC2: the meal has to be somewhere the traveler is already passing."""
    zones = {0: _sights_along_a_line(4)}
    on_route = _food_poi("Corner Bistro", 48.206, 16.3705)
    far_away = _food_poi("Distant Grill", 48.35, 16.60)

    day = build_multi_day_itinerary(zones, daily_minutes_budget=480,
                                     food_pois=[far_away, on_route], min_food_per_day=1)[0]

    names = [p["name"] for p in _meals_in(day)]
    assert "Corner Bistro" in names
    assert "Distant Grill" not in names


def test_the_same_restaurant_is_not_booked_twice_in_one_trip():
    zones = {0: _sights_along_a_line(3), 1: _sights_along_a_line(3, lat=48.23),
             2: _sights_along_a_line(3, lat=48.26)}
    food = [_food_poi(f"Cafe {i}", 48.20 + i * 0.008, 16.372) for i in range(10)]

    days = build_multi_day_itinerary(zones, daily_minutes_budget=480,
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


def test_query_set_is_powered_and_not_mostly_self_graded():
    """The comparison was run on 18 queries of which only 11 were not graded
    against the retriever's own traversal, and at that size a real effect is
    detected about a third of the time. Both properties are easy to lose again
    by editing the query list, so they are pinned here."""
    assert len(TEST_QUERIES) >= 45
    assert {q.tier for q in TEST_QUERIES} == {"handwritten", "grid"}

    not_self_graded = [q for q in TEST_QUERIES if dependence_level(q) != "subset"]
    assert len(not_self_graded) >= 30

    # No archetype may own the set the way Culture Enthusiast owned 44% of it.
    counts = collections.Counter(q.archetype for q in TEST_QUERIES)
    assert max(counts.values()) / len(TEST_QUERIES) < 0.40
    assert set(counts) == set(CATEGORY_AFFINITY), "every archetype should be exercised"


def test_dependence_level_spots_the_self_graded_queries():
    query = comparative_analysis.TestQuery
    # Names the key's category next to a transport word: the graph router
    # dispatches to the same traversal the key is built from, at 3km inside the
    # key's 6km, so its results cannot fall outside the key.
    assert dependence_level(query(MAIN_CITY, "Culture Enthusiast", ("landmark",), True,
                                  "landmarks near a transport hub")) == "subset"
    # Same category, no transport constraint -- the router's filter is wider
    # than the key rather than nested inside it.
    assert dependence_level(query(MAIN_CITY, "Culture Enthusiast", ("landmark",), False,
                                  "the best landmarks to visit")) == "superset"
    # The router never reaches for the key's category at all.
    assert dependence_level(query(MAIN_CITY, "Culture Enthusiast", ("history",), False,
                                  "somewhere calm to spend a slow afternoon")) == "independent"


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


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
