"""A single day's route: the time budget, the day's own clock (#59), how a
multi-day trip balances and fills (#19), and the meals every day gets
(#20, #29).
"""

import pytest

from roamwise.agents.orchestrator import RoamWiseOrchestrator
from roamwise.agents.router_agent import ICONIC_GUARANTEED, RouterAgent
from roamwise.knowledge_graph.build_graph import GraphIndex
from roamwise.optimization.routing import NIGHTLIFE_EARLIEST_HOUR, optimize_day_route
from roamwise.optimization.scoring import quality
from roamwise.optimization.toptw import ICONIC_QUALITY_THRESHOLD, build_multi_day_itinerary
from roamwise.optimization.travel_modes import get_travel_mode
from roamwise.tests.helpers import MAIN_CITY, _flat


@pytest.mark.slow
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


@pytest.mark.slow
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


def test_a_landmark_far_from_the_cluster_survives_the_solver():
    """Issue #122. One scalar `drop_penalty_m` prices every stop the same, so
    the stop furthest from the cluster is always the cheapest to drop however
    well known it is -- measured over 2 cities x 7 archetypes, the Eiffel Tower
    reached the router's shortlist in 3 and the plan in 0, and so did the
    Berlin Wall (`evaluation/iconic_coverage.py`).

    The pool here is that shape in miniature: eight ordinary places within a few
    hundred metres of each other, and one famous one 4 km away. With the
    multiplier off the day takes the cluster and leaves the landmark; with the
    shipped setting the landmark costs more to skip than the detour to reach it.

    Asserted as a pair rather than as "the landmark is in the plan", because the
    second half alone would also pass if the solver simply had time for
    everything -- what this is about is which of the two the solver gives up.
    """
    cluster = [dict(_daytime_poi(f"Ordinary {i}", 48.860 + i * 0.001, 2.340,
                                 minutes=40), popularity_score=1.0)
               for i in range(8)]
    # 8.8 km out: more than the flat 8 km a dropped stop costs, less than the
    # 12 km it costs once the multiplier applies. The window is what the test
    # is for, and it is narrow on purpose -- at 10 km even 1.5x leaves the
    # landmark behind, which is the honest limit of a penalty that is priced
    # in metres.
    landmark = dict(_daytime_poi("Famous Tower", 48.860, 2.220, minutes=40),
                    popularity_score=5.0)
    pois = cluster + [landmark]

    without = build_multi_day_itinerary(pois, 1, daily_minutes_budget=900,
                                        day_start_hour=9.0,
                                        iconic_drop_multiplier=1.0)
    with_penalty = build_multi_day_itinerary(pois, 1, daily_minutes_budget=900,
                                             day_start_hour=9.0)

    names = lambda days: {p["name"] for day in days for p in day["route"]}
    assert "Famous Tower" not in names(without), \
        "the flat penalty is what this test is about; if it now keeps the " \
        "landmark the fixture no longer reproduces #122"
    assert "Famous Tower" in names(with_penalty)


def test_a_landmark_stays_a_candidate_for_a_traveler_who_did_not_ask_for_one():
    """Issue #122, item 4. The drop penalty can only keep what the router is
    handed, and retrieval is archetype-driven: measured, the Eiffel Tower
    reached the pool for 3 of 7 archetypes and the Berlin Wall for 3 of 7,
    because a Nightlife Seeker's query asks for bars and gets them. So the city's
    best-known places are completed into the working set from the graph -- the
    same route `_food_pois` takes, and for the same reason: widening the query
    instead would change what the Fusion/Hybrid/standard comparison measures.

    Checked on the working set rather than on a plan, because what the guarantee
    promises is candidacy. Whether a guaranteed landmark is then *kept* is the
    drop penalty's job and is measured in `evaluation/iconic_coverage.py`.
    """
    idx = GraphIndex()
    router = RouterAgent(idx)
    iconic = router._iconic_pois(MAIN_CITY)
    assert len(iconic) == ICONIC_GUARANTEED

    # A working set the archetype's own preferences might plausibly produce:
    # the city's least-known POIs, holding none of the landmarks.
    iconic_ids = {p["poi_id"] for p in iconic}
    obscure = [p for p in idx.city_pois(MAIN_CITY)
               if p["poi_id"] not in iconic_ids][-20:]

    completed = router._with_iconic(MAIN_CITY, obscure)
    assert iconic_ids <= {p["poi_id"] for p in completed}

    # Idempotent, and never twice: `routing.py` compares POIs by `id(p)`, so a
    # landmark carried under two dicts would be schedulable twice.
    again = router._with_iconic(MAIN_CITY, completed)
    ids = [p["poi_id"] for p in again]
    assert len(ids) == len(set(ids))
    assert len(again) == len(completed)


def test_the_iconic_penalty_only_touches_the_top_of_the_pool():
    """The idea `toptw_scoring_ablation.py` rejected at #72 was reweighting
    *every* node with the selection score; this is a handful of POIs paying
    more. The threshold is what keeps those two different, so it is worth an
    assertion: at the shipped 0.99, a pool with one clear leader must leave
    everything else on the flat price.

    Checked through `scoring.quality`, the same normalisation the solver reads,
    rather than by counting solutions -- what a POI costs to drop is not
    observable in the returned itinerary."""
    pois = [{"name": f"P{i}", "popularity_score": score}
            for i, score in enumerate([5.0, 4.5, 4.0, 3.0, 1.0])]
    fame = quality(pois)
    above = [p["name"] for p, f in zip(pois, fame) if f >= ICONIC_QUALITY_THRESHOLD]
    assert above == ["P0"], f"expected one landmark above the threshold, got {above}"


# --- issue #19: day balance, budget filling, and travel modes ---

@pytest.mark.slow
def test_multi_day_itinerary_fills_every_day_near_its_budget():
    """The reported symptom: a multi-day plan where some days were empty and
    others held a two-hour route. Every day must now come back non-empty and
    reasonably full."""
    idx = GraphIndex()
    pois = idx.city_pois(MAIN_CITY)[:40]
    days = build_multi_day_itinerary(pois, 5, daily_minutes_budget=480)

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


@pytest.mark.slow
def test_the_same_restaurant_is_not_booked_twice_in_one_trip():
    zones = {0: _sights_along_a_line(3), 1: _sights_along_a_line(3, lat=48.23),
             2: _sights_along_a_line(3, lat=48.26)}
    food = [_food_poi(f"Cafe {i}", 48.20 + i * 0.008, 16.372) for i in range(10)]

    days = build_multi_day_itinerary(*_flat(zones), daily_minutes_budget=480,
                                      food_pois=food, min_food_per_day=2)

    booked = [p["name"] for day in days for p in _meals_in(day)]
    assert len(booked) == len(set(booked))


@pytest.mark.slow
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
