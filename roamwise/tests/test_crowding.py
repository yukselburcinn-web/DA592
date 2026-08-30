"""Crowding as a real signal rather than a stub (#71), and the router choosing
the hour a stop is visited at, not just the stop (#33).
"""

import datetime

import pandas as pd
import pytest

from roamwise.knowledge_graph.build_graph import GraphIndex
from roamwise.optimization.routing import FOOD_CATEGORY
from roamwise.optimization.toptw import build_multi_day_itinerary
from roamwise.tests.helpers import DATA_DIR, MAIN_CITY


# --- issue #71: the crowding factor stopped being a stub ---

def _crowd_poi(poi_id, category="museum", name=None):
    return {"poi_id": poi_id, "category": category, "name": name or poi_id,
            "lat": 48.86, "lon": 2.34, "popularity_score": 4.0}


def test_crowding_discount_separates_a_busy_poi_from_a_quiet_one():
    """The factor returned a constant 1.0 for the project's whole history,
    because the only demand data was one number per city per month. It now
    reads `data/crowding.csv`, and the thing that makes it worth having is
    that it *differs* between POIs."""
    from roamwise.optimization.scoring import busyness, crowding_discount, _crowding_tables

    typical = _crowding_tables()["typical"]
    assert len(typical) > 100, "crowding.csv should cover a useful slice of the catalogue"
    busiest = max(typical, key=typical.get)
    quietest = min(typical, key=typical.get)

    assert crowding_discount(_crowd_poi(busiest)) < crowding_discount(_crowd_poi(quietest))
    assert 0 < crowding_discount(_crowd_poi(busiest)) < 1
    assert busyness(_crowd_poi(busiest)) > busyness(_crowd_poi(quietest))


def test_an_unmeasured_poi_is_not_rewarded_for_being_unmeasured():
    """Scoring a POI with no reading at 1.0 would hand every place the
    enrichment missed a silent bonus over the places it measured: the busy
    ones get marked down and the unknown ones do not. The fallback is the
    category's own mean, so absence of data is not an advantage."""
    from roamwise.optimization.scoring import (CROWDING_STRENGTH, crowding_discount,
                                               _crowding_tables)

    tables = _crowding_tables()
    assert tables["by_category"], "a category fallback has to exist for this to hold"
    for category, mean in tables["by_category"].items():
        unknown = crowding_discount(_crowd_poi("NOT_IN_CATALOGUE", category))
        assert unknown < 1.0, f"{category}: an unmeasured POI kept its full score"
        assert unknown == pytest.approx(1.0 - CROWDING_STRENGTH * mean / 100, abs=1e-9)


def test_crowding_changes_which_stops_the_score_selects():
    """The regression that matters. A factor that is the same for every
    candidate cancels out of a normalised score and provably cannot change an
    itinerary -- which is what the old stub did, and why it returned 1.0 and
    said so rather than looking like personalization. This asserts the new one
    is not in that position: turning it off changes the selection."""
    from roamwise.optimization import scoring

    idx = GraphIndex()
    pois = idx.city_pois(MAIN_CITY)[:80]
    prefs = {d: 0.5 for d in scoring.PREFERENCE_DIMS}

    with_crowd = [p["poi_id"] for p in scoring.select_by_score(pois, prefs, 20)]
    saved = scoring.CROWDING_STRENGTH
    try:
        scoring.CROWDING_STRENGTH = 0.0          # her POI'ye 1.0, eski stub davranışı
        without = [p["poi_id"] for p in scoring.select_by_score(pois, prefs, 20)]
    finally:
        scoring.CROWDING_STRENGTH = saved

    assert with_crowd != without, \
        "crowding makes no difference to selection -- it is a stub again"


def test_the_hour_carries_more_signal_than_the_poi():
    """Why the hour was worth a second node (#33): a POI's own day swings far
    wider than POIs differ from each other, so a factor that reads only the
    per-POI half is reading the smaller signal. This pins the reason."""
    import statistics as st
    from roamwise.optimization.scoring import _crowding_tables

    tables = _crowding_tables()
    between = st.pstdev(list(tables["typical"].values()))

    swings = []
    for series in tables["hourly"].values():
        by_day = {}
        for (day, _hour), busy in series.items():
            by_day.setdefault(day, []).append(busy)
        swings += [max(v) - min(v) for v in by_day.values() if len(v) > 4]

    assert st.median(swings) > 3 * between, \
        "if the hour stopped dominating, the per-POI-only factor would be the wrong trade"


# --- issue #33: the router chooses the hour, not just the stop ---

def _hour_gap(days, start_date):
    """Exposure minus the same stops' own typical level: how much busier than
    its own average a stop is at the hour it was scheduled for. The subtraction
    is what makes this a measure of *timing* -- an itinerary that simply picked
    quieter POIs moves both terms and leaves the gap where it was."""
    from roamwise.optimization.scoring import busyness

    codes = ("Mo", "Tu", "We", "Th", "Fr", "Sa", "Su")
    gaps = []
    for day_i, day in enumerate(days):
        weekday = codes[(start_date + datetime.timedelta(days=day_i)).weekday()]
        for poi, slot in zip(day["route"], day["schedule"]):
            at_hour = busyness(poi, weekday, int(slot["arrival"]) % 24)
            typical = busyness(poi)
            if at_hour is not None and typical is not None:
                gaps.append(at_hour - typical)
    return (sum(gaps) / len(gaps) if gaps else None), len(gaps)


def _measured_pool(city, limit=40):
    """POIs the crowding series actually covers -- the only ones this factor
    can say anything about."""
    from roamwise.optimization.scoring import _crowding_tables

    hourly = _crowding_tables()["hourly"]
    pois = [p for p in GraphIndex().city_pois(city) if p.get("poi_id") in hourly]
    return pois[:limit]


def test_a_measured_poi_is_offered_its_quiet_hours_as_a_second_option():
    """The mechanism, and the reason it cannot take an itinerary away. A POI
    with a real dip in its day gets a second node pinned to that dip -- but the
    day-wide node stays, so every route that was reachable before still is, at
    the same price. An earlier design cut the day into slots instead, and a
    stop that fitted no slot lost its place in the day entirely."""
    from roamwise.optimization import toptw

    measured = _measured_pool(MAIN_CITY, limit=60)
    offered = [toptw._crowd_slots(p, "Fr", 9.0, 720, hour_aware=True)
               for p in measured]
    with_window = [o for o in offered if len(o) == 2]
    assert with_window, "no measured POI was offered a quiet window at all"
    for options in with_window:
        (window, quiet_level), (day_wide, day_level) = options
        assert window[0] == "q" and day_wide is None
        assert quiet_level < day_level, "the quiet window is not the quieter one"

    off = toptw._crowd_slots(measured[0], "Fr", 9.0, 720, hour_aware=False)
    assert off == [(None, None)], "hour_aware=False has to be today's router"


def test_every_crowding_row_says_where_it_came_from():
    """`crowding.csv` follows `poi.csv`'s `hours_source` / `price_source`
    pattern (#33): the file states its provenance instead of a README
    asserting it. The column is constant today and still earns its bytes --
    the issue's fallback source is a monthly pageviews series for the POIs
    this scrape never reached, and it has to land beside these rows rather
    than be told apart by which POI it is about."""
    from roamwise.knowledge_graph.build_graph import DATA_DIR

    crowding = pd.read_csv(DATA_DIR / "crowding.csv")
    assert "source" in crowding.columns
    assert crowding["source"].notna().all(), "a row with no source is an assertion"
    assert set(crowding["source"]) <= {"gmaps", "pageviews"}


def test_an_unmeasured_poi_is_not_the_cheapest_hour_in_the_day():
    """`expected_busyness` has to answer for a POI nobody measured, because a
    stop priced at nothing is the cheapest stop in the pool and 59% of the
    catalogue is unmeasured. Absence of data must not be an advantage -- the
    same rule the static discount already follows."""
    from roamwise.optimization.scoring import _crowding_tables, expected_busyness

    for category, mean in _crowding_tables()["by_category"].items():
        unmeasured = {"poi_id": "NOT_IN_CATALOGUE", "category": category}
        assert expected_busyness(unmeasured) == pytest.approx(mean)
        assert expected_busyness(unmeasured) > 0


def test_a_sitting_is_offered_a_quiet_hour_inside_its_own_band():
    """Issue #109. #33 left the sittings out on the grounds that a compulsory
    meal carries no choice of hour. Half of that is right: the sitting happens
    either way. The half that is wrong is where the whole residual sat -- a
    sitting is pinned to a four-hour band, not to a minute, and inside the
    dinner band a restaurant runs 36% busy at its quietest hour and 100% at
    its busiest."""
    from roamwise.optimization import toptw

    meals = [(k, target - toptw.MEAL_WINDOW_HOURS, target + toptw.MEAL_WINDOW_HOURS)
             for k, target in enumerate(toptw.meal_target_hours(9.0, 720, 2))]
    restaurants = [p for p in _measured_pool(MAIN_CITY, limit=400)
                   if p.get("category") == FOOD_CATEGORY]
    assert restaurants, "no measured restaurant to test with"

    offered = 0
    for poi in restaurants:
        slots = toptw._meal_slots(poi, "Fr", meals, hour_aware=True)
        sittings = [toptw._sitting_of(slot) for slot, _ in slots]
        for k, low, high in meals:
            assert sittings.count(k) >= 1, "a sitting lost its band-wide copy"
            for slot, level in slots:
                if isinstance(slot, tuple) and slot[1] == k:
                    assert low <= slot[2] and slot[3] <= high, \
                        "a quiet window escaped its own sitting band"
                    offered += 1
    assert offered, "no restaurant was offered a quiet hour at all"

    off = toptw._meal_slots(restaurants[0], "Fr", meals, hour_aware=False)
    assert [toptw._sitting_of(s) for s, _ in off] == [k for k, _, _ in meals]
    assert all(level is None for _, level in off), "hour_aware=False priced a meal"


def test_both_kinds_of_meal_copy_count_as_the_same_sitting():
    """The meal dimension counts sittings by node. A sitting now has two kinds
    of node -- the band and the quiet stretch inside it -- and if the counter
    told them apart, "one lunch per day" would quietly become "one lunch per
    day per kind of copy", which is exactly the two-lunches bug #20 and #29
    exist to prevent."""
    from roamwise.optimization.toptw import _sitting_of

    assert _sitting_of(0) == 0
    assert _sitting_of(("m", 0, 12.0, 14.0)) == 0
    assert _sitting_of(("m", 1, 18.0, 20.0)) == 1
    assert _sitting_of(("q", 9.0, 12.0)) is None   # a sight's window is not a sitting
    assert _sitting_of(None) is None


@pytest.mark.slow
def test_the_router_schedules_stops_at_quieter_hours_than_it_used_to():
    """Issue #33's acceptance criterion. Before this, stops landed 11.1 points
    busier than the places' own averages across the measured trips -- the
    router put people in at the busy hour, because the hours a day naturally
    fills are the hours everyone else fills too. The full sweep is
    `evaluation/crowding_hour_measurement.py`; this pins the direction."""
    from roamwise.optimization.toptw import build_multi_day_itinerary

    start_date = datetime.date(2026, 9, 25)   # a Friday, so the trip spans Fri-Sun
    pool = _measured_pool(MAIN_CITY, limit=40)
    kwargs = dict(n_days=3, daily_minutes_budget=720, day_start_hour=9.0,
                  start_date=start_date)
    before = build_multi_day_itinerary(pool, hour_aware=False, **kwargs)
    after = build_multi_day_itinerary(pool, hour_aware=True, **kwargs)

    gap_before, n_before = _hour_gap(before, start_date)
    gap_after, n_after = _hour_gap(after, start_date)
    assert n_before and n_after, "no measured stop was scheduled at all"
    assert gap_after < gap_before, (
        f"the hour is not being chosen: gap {gap_before:.1f} -> {gap_after:.1f}")

    # Quiet hours must not be bought by not going. The bound is loose because
    # this pool is deliberately harsher than a real one: every one of its 40
    # POIs is measured, so every one of them competes for the same quiet hours.
    # On the pool the app actually retrieves, the measured cost is 0.4% of the
    # stops (6.71 -> 6.68 per day) -- see evaluation/crowding_hour_measurement.py.
    stops_before = sum(len(d["route"]) for d in before)
    stops_after = sum(len(d["route"]) for d in after)
    assert stops_after >= 0.9 * stops_before, (
        f"quieter hours cost stops: {stops_before} -> {stops_after}")


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
