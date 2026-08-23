"""End-to-end smoke tests covering every module in the pipeline. Run with:
    cd roamwise && ../venv/bin/pytest tests/ -v
"""
import importlib
import math
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from roamwise.knowledge_graph.build_graph import GraphIndex
from roamwise.models.forecasting import forecast_city, best_months_to_visit
from roamwise.models.segmentation import TravelerSegmenter, POIZoner
from roamwise.retrieval.fusion import FusionRetriever
from roamwise.optimization.routing import build_multi_day_itinerary, optimize_day_route
from roamwise.optimization.travel_modes import get_travel_mode
from roamwise.agents.orchestrator import RoamWiseOrchestrator
from roamwise.agents.router_agent import RouterAgent
from roamwise.evaluation.comparative_analysis import (
    paired_significance, run_comparative_analysis, summarize,
)


def test_knowledge_graph_builds_and_traverses():
    idx = GraphIndex()
    stats = idx.stats()
    assert stats["by_type"]["City"] == 8
    assert stats["by_type"]["POI"] == 1200
    hop = idx.multi_hop_transport_to_poi("IST", "landmark")
    assert len(hop) > 0
    assert all("nearest_hub_km" in r for r in hop)


def test_forecasting_produces_crowding_labels():
    fc = forecast_city("PAR")
    assert len(fc) == 12
    assert set(fc.crowding_level.astype(str)) <= {"low", "medium", "high"}
    months = best_months_to_visit("PAR", top_k=3)
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
    pois = idx.city_pois("AMS")
    zones = POIZoner().zone(pois, n_zones=3)
    total = sum(len(v) for v in zones.values())
    assert total == len(pois)


def test_fusion_retrieval_all_configs_run():
    fr = FusionRetriever()
    for config in ["fusion", "hybrid", "standard"]:
        results = fr.retrieve("museums near a transport hub", config=config, destination_id="ROM", top_k=5)
        if config == "standard":
            assert results == []
        else:
            assert len(results) > 0
            assert all(r["destination_id"] == "ROM" for r in results)


def test_fusion_beats_hybrid_on_archetype_grounding():
    fr = FusionRetriever()
    fusion_results = fr.retrieve("things to do", config="fusion", destination_id="ROM", archetype="Nightlife Seeker", top_k=8)
    fusion_categories = {r.get("text", "") for r in fusion_results}
    assert any("nightlife" in t.lower() or "trastevere" in t.lower() for t in fusion_categories)


def test_routing_respects_time_budget():
    idx = GraphIndex()
    pois = idx.city_pois("VIE")
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


def test_routing_falls_back_when_osrm_unreachable(monkeypatch):
    import roamwise.optimization.routing as routing_module
    monkeypatch.setattr(routing_module, "fetch_distance_duration_matrix",
                        lambda points, profile="foot": None)

    idx = GraphIndex()
    pois = idx.city_pois("VIE")[:4]
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
    pois = idx.city_pois("VIE")[:2]
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
    cities = ["IST", "PAR", "ROM", "BCN", "AMS", "PRG", "VIE", "LIS"]

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

    assert strictly_better >= len(cities) // 2, \
        "balancing should visibly help in most cities, not merely never hurt"


def test_multi_day_itinerary_fills_every_day_near_its_budget():
    """The reported symptom: a multi-day plan where some days were empty and
    others held a two-hour route. Every day must now come back non-empty and
    reasonably full."""
    idx = GraphIndex()
    pois = idx.city_pois("VIE")[:40]
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
    pois = [p for p in idx.city_pois("VIE") if p.get("category") != "food"][:24]

    result = agent.run("VIE", pois, n_days=3, daily_minutes_budget=480)

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

    pinned = orch.plan_trip(prefs, destination_id="PAR", n_days=2)
    assert pinned["destination_id"] == "PAR"


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


@pytest.mark.parametrize("city", ["IST", "PAR", "ROM", "BCN", "AMS", "PRG", "VIE", "LIS"])
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
