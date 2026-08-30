"""Travel between stops without a routing server (#32): the committed street
network, RAPTOR over the GTFS timetable, the Paris timetable end to end, and
the route the map actually draws (#94).
"""

import math

import pandas as pd
import pytest

from roamwise.agents.orchestrator import RoamWiseOrchestrator
from roamwise.knowledge_graph.build_graph import GraphIndex
from roamwise.optimization.toptw import build_multi_day_itinerary
from roamwise.tests.helpers import CITY_CODES, DATA_DIR, MAIN_CITY


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
    """Brandenburg's coordinate used to be the middle of the airfield, 1,225m
    from the nearest footway and outside a village. Measuring access over the
    street network from there put the traveller on rural buses and the journey
    at 240 minutes, while the airport's own platforms sat 785m away and were
    never considered.

    The row now carries those platforms (#92), so this journey is measured over
    real footways rather than rescued by the straight-line fallback -- and the
    answer is the same 45 minutes the fallback happened to produce, which is
    the point: the number was right for the wrong reason."""
    from roamwise.optimization.street_network import fetch_distance_duration_matrix

    points = _transit_points("Brandenburg Airport", "Brandenburg Gate")
    _, transit = fetch_distance_duration_matrix(points, profile="transit")

    assert 30 < transit[0][1] < 75, f"FEX and the S-Bahn do not take {transit[0][1]:.0f} minutes"


def test_every_airport_hub_stands_where_the_footway_network_can_place_it():
    """#92. A hub's coordinate is where access to it gets measured from, and
    for an aerodrome OSM's `out center` is the middle of the airfield -- a
    point between the runways that no traveller occupies.

    Nothing downstream can tell that from a real coordinate. Berlin
    Brandenburg's centroid sat 1,225m from the nearest footway node, in a field
    outside Wassmannsdorf; access measured from there found two village bus
    stops and reported 240 minutes to the city centre against a true ~45.
    Charles de Gaulle's was 185m out. Both now sit on the airport's own
    station.

    150m is `build_transit_matrix.SNAP_TRUST_METRES`, the threshold past which
    that build stops trusting the network to say where a place is and falls
    back to a straight line with a detour factor. Inside it, access is measured
    over real footways -- which is the whole point of the fix, and the thing a
    silent regression here would undo.
    """
    from roamwise.optimization.street_network import load_city_network, snap

    hubs = pd.read_csv(DATA_DIR / "transport.csv")
    airports = hubs[hubs.type == "airport"]
    assert not airports.empty, "transport.csv should carry at least one airport"

    checked = 0
    for code, group in airports.groupby("destination_id"):
        net = load_city_network(code, "foot")
        if net is None:
            continue
        _, offsets = snap(net, group.lat.to_numpy(), group.lon.to_numpy(),
                          max_metres=None)
        for name, metres in zip(group.name, offsets):
            assert metres <= 150.0, (
                f"{name} sits {metres:.0f}m from the nearest footway node -- "
                f"the runway centroid again, or a terminal the network cannot reach")
            checked += 1
    assert checked, "no city with a committed walking network held an airport"


# --- issue #94: what the map draws is what the day costs ---


def _paris_day_points(*names):
    return _transit_points(*names)


def test_drawn_route_follows_the_street_network_it_was_priced_on():
    """#94. The map drew a straight line between stops whatever the mode said,
    so a day the panel reported as 6.85 km appeared as 3.17. The polyline now
    comes off the same graph the distance does, and the two have to agree --
    the issue's criterion is 10%, and the residual is osmnx's simplification
    chording curved streets, which shortens the drawing and never the
    distance."""
    from roamwise.optimization.street_network import fetch_distance_duration_matrix
    from roamwise.optimization.routing import route_geometry

    points = _paris_day_points("Eiffel Tower", "Louvre Museum", "Notre-Dame de Paris",
                               "Sacré-Cœur")
    for mode, profile in (("walking", "foot"), ("driving", "car")):
        km, _ = fetch_distance_duration_matrix(points, profile=profile)
        geometry = route_geometry(points, use_real_routing=True, travel_mode=mode)

        assert len(geometry) == len(points) - 1
        for i, (vertices, is_real_route) in enumerate(geometry):
            assert is_real_route, f"{mode} leg {i} should have real geometry"
            assert len(vertices) > 2, "a real path is more than its two endpoints"
            drawn = sum(_metres(a, b) for a, b in zip(vertices, vertices[1:])) / 1000.0
            priced = km[i][i + 1]
            assert abs(drawn - priced) / priced < 0.10, (
                f"{mode} leg {i}: drew {drawn:.2f} km for a {priced:.2f} km leg")


def _metres(a, b):
    lat = math.radians((a[0] + b[0]) / 2)
    return math.hypot((b[1] - a[1]) * math.cos(lat), b[0] - a[0]) * 111_320.0


def test_a_leg_with_no_real_geometry_is_drawn_straight_and_says_so():
    """Two cases where no path exists to draw, and neither may be dressed up as
    a route: the transit matrix stores minutes rather than the lines a journey
    used, and with real routing off the straight line *is* the model the day
    was costed with. Both come back flagged, which is what the map turns into
    a dashed line and a caption."""
    from roamwise.optimization.routing import route_geometry

    points = _paris_day_points("Eiffel Tower", "Louvre Museum", "Notre-Dame de Paris")

    for geometry, why in (
            (route_geometry(points, use_real_routing=True, travel_mode="transit"), "transit"),
            (route_geometry(points, use_real_routing=False, travel_mode="walking"), "no real routing")):
        assert len(geometry) == len(points) - 1, why
        for vertices, is_real_route in geometry:
            assert not is_real_route, f"{why} has no route geometry to draw"
            assert len(vertices) == 2, f"{why} is a straight segment, not a path"


def test_a_city_with_no_street_network_still_gets_a_drawable_line():
    """The drawing must never disappear. Vienna has no committed network, so
    there is nothing to trace -- and the answer is a straight segment marked
    as not-a-route, not an empty list the map would silently skip."""
    from roamwise.optimization.routing import route_geometry

    vienna = [{"lat": 48.2082, "lon": 16.3738}, {"lat": 48.2000, "lon": 16.3600}]
    geometry = route_geometry(vienna, use_real_routing=True, travel_mode="walking")

    assert [is_real for _, is_real in geometry] == [False]
    assert len(geometry[0][0]) == 2


@pytest.mark.slow
def test_a_day_carries_the_place_its_first_leg_is_measured_from():
    """`distance_km` counts the journey from where the day starts to its first
    stop, and that place is not in `route`. Without it the map is short by the
    whole first leg -- 1.2-1.7km on a Paris walking day -- however faithfully
    it traces the rest (#94)."""
    from roamwise.optimization.street_network import fetch_distance_duration_matrix

    pois = GraphIndex().city_pois(MAIN_CITY)[:40]
    centre = pd.read_csv(DATA_DIR / "destinations.csv")
    centre = centre[centre.destination_id == MAIN_CITY].iloc[0]
    hub = {"name": centre.city, "lat": float(centre.lat), "lon": float(centre.lon)}

    itinerary = build_multi_day_itinerary(pois, n_days=2, start_hub=hub,
                                          use_real_routing=True, travel_mode="walking")
    checked = 0
    for day in itinerary:
        if not day["route"]:
            continue
        origin = day["origin"]
        assert origin is not None and "lat" in origin and "lon" in origin
        assert (origin["lat"], origin["lon"]) == (hub["lat"], hub["lon"])

        drawn = [origin] + day["route"]
        km, _ = fetch_distance_duration_matrix(drawn, profile="foot")
        total = sum(km[i][i + 1] for i in range(len(drawn) - 1))
        assert abs(total - day["distance_km"]) < 0.05, (
            f"day {day['day']}: legs from the origin sum to {total:.2f}, "
            f"panel says {day['distance_km']}")
        checked += 1
    assert checked, "no day was routed, so nothing was checked"

    # And the other branch: with nowhere to start from, no first leg is charged
    # and there is correspondingly nothing extra to draw.
    for day in build_multi_day_itinerary(pois, n_days=2, use_real_routing=True,
                                          travel_mode="walking"):
        assert day["origin"] is None


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
    _, duration, used_real, _ = _build_distance_functions(points, use_real_routing=False,
                                                       travel_mode="transit")

    assert used_real is True
    assert duration(points[0], points[1]) < 60


@pytest.mark.slow
def test_transit_plans_a_whole_trip():
    from roamwise.agents.orchestrator import RETRIEVED_POIS_PER_DAY, RoamWiseOrchestrator

    prefs = {"budget": 0.6, "culture": 0.9, "nature": 0.3, "nightlife": 0.3,
             "relax": 0.4, "adventure": 0.3}
    planned = RoamWiseOrchestrator().plan_trip(prefs, destination_id=MAIN_CITY, n_days=3,
                                               travel_mode="transit", use_real_routing=True)

    days = planned["routing"]["itinerary"]
    assert all(day["used_real_routing"] for day in days), "fell back off the timetable"
    assert any(day["route"] for day in days)
