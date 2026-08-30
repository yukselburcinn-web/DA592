"""Opening hours as a rule over days rather than one open/close pair (#70),
and closing times that run past midnight (#61).
"""

import datetime

import pytest

from roamwise.agents.orchestrator import RoamWiseOrchestrator
from roamwise.agents.router_agent import DAY_START_HOURS, RouterAgent, start_hour_for
from roamwise.knowledge_graph.build_graph import GraphIndex
from roamwise.optimization.routing import optimize_day_route
from roamwise.optimization.toptw import build_multi_day_itinerary
from roamwise.tests.helpers import CITY_CODES, MAIN_CITY, MONDAY, _flat


# --- issue #70: opening hours are a rule over days, not one open/close pair ---

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


@pytest.mark.slow
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


@pytest.mark.slow
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

@pytest.mark.slow
def test_orchestrator_end_to_end():
    orch = RoamWiseOrchestrator()
    prefs = {"budget": 0.7, "culture": 0.3, "nature": 0.2, "nightlife": 0.9, "relax": 0.2, "adventure": 0.3}
    result = orch.plan_trip(prefs, n_days=2)
    assert result["archetype"] == "Nightlife Seeker"
    assert result["destination_id"] in orch.destinations.destination_id.values
    assert len(result["routing"]["itinerary"]) == 2
    assert result["final_plan"]
