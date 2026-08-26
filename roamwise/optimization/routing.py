"""
Routing primitives: distance, duration, opening hours, and a single day's
route.

The trip itself is assembled in `optimization/toptw.py`, which solves
selection, day assignment and ordering as one model. This module holds what
that model is built out of, plus the single-day router it replaced:

  - `_build_distance_functions` turns a set of points into (distance_km,
    duration_min) callables, either the mode's haversine estimate or a real
    street-network matrix from OSRM (`use_real_routing=True`, issue #8).
    Opt-in and network-dependent -- see osrm_client.py's docstring for why,
    and every caller falls back to haversine automatically if the request
    fails.
  - `_opening_intervals` / `_next_open_hour` read a POI's hours as the
    grammar OSM writes them in, resolved against the day's actual date
    (issue #70). TOPTW turns these straight into time windows.
  - `optimize_day_route` routes *one* day over a fixed candidate set:
    nearest-neighbour construction, 2-opt, then a sequential pass that waits
    for closed venues and drops what doesn't fit. It no longer assembles
    trips -- the multi-day passes that used to live here were what issue #72
    replaced -- but it remains the honest single-day primitive, and its
    behaviour is what the TOPTW model is measured against.
  - Travel mode (`travel_mode=`, issue #19): legs are costed as walking,
    driving or a per-leg hybrid rather than always assuming a flat walking
    speed. See travel_modes.py.
"""
import datetime
import math

from opening_hours import OpeningHours

from roamwise.optimization.osrm_client import fetch_distance_duration_matrix
from roamwise.optimization.travel_modes import DEFAULT_MODE, HybridTravelMode, get_travel_mode

FOOD_CATEGORY = "food"
# When a day begins, unless the traveler says otherwise. A default, not a
# constraint -- it was effectively the latter until issue #59, because nothing
# above RouterAgent.run() could pass a different one.
DEFAULT_DAY_START_HOUR = 9.0
# Cheapest-insertion picks whichever open slot costs the least geographic
# detour, with no separate reward for landing close to its target time --
# and appending at the very end of the day is nearly always the cheapest
# detour, so both meals kept gravitating to the same end-of-day slot next to
# each other (#29). MIN_MEAL_GAP_FRACTION makes that impossible outright: no
# slot within this fraction of the day's own span of an already-placed meal
# is eligible, regardless of its detour cost. 0.375 -> 3h apart on a full
# 8h/480min day (#29's own suggested figure), scaling down on shorter days
# the same way _meal_target_hours' 0.40/0.85 split already does -- and
# staying under that split's own 0.45 gap keeps the two constraints from
# ever being mutually unsatisfiable by construction.
MIN_MEAL_GAP_FRACTION = 0.375
# A bar or club belongs at the end of a day, and that is a property of the
# *category*, not of the venue's stated hours: the 2-opt pass orders purely on
# geography, so a nightlife venue whose OSM hours happen to start early (Bijou
# Bar opens at 07:00, Le Select at 07:00, Kilkenny at 10:00) was legitimately
# "open" at 09:00 and got picked as the day's opening stop. Measured over 32
# days, 6 began with a bar and all 12 scheduled nightlife stops fell before
# 17:00 (issue #59). Reordering by category is the fix; correcting the hours
# is not, because a venue being open in the morning does not make a morning
# visit to it a plan.
NIGHTLIFE_CATEGORY = "nightlife"
# Earliest hour a nightlife stop may be scheduled at, whatever its stated
# opening time. 18:00 is where the catalogue's own well-sourced venues sit
# (17 of 41 are 18:00-02:00), so this makes the mis-sourced ones behave like
# the correctly-sourced majority rather than inventing a threshold.
NIGHTLIFE_EARLIEST_HOUR = 18.0
# Calendar days of opening intervals _next_open_hour searches. One, deliberately:
# a venue's *closing* time may run into tomorrow (a bar shutting at 02:00), but
# its *opening* time may not. Allowing tomorrow's opening too let the scheduler
# wait out the night to be first through the door -- a measured 18-hour day from
# 15:00 put a 22:18 stop and then a 07:00 one nine hours later, which is not an
# itinerary. A venue that has closed for the night is simply gone for this day.
_OPENING_HORIZON_DAYS = 1


def _tag_intervals(raw: str, day_date) -> list[tuple[float, float]] | None:
    """OPEN intervals from an OSM `opening_hours` tag, as hours from midnight of
    `day_date` -- the same past-midnight clock the coarse pair below uses, so
    26.0 is 02:00 the next morning.

    This is the point of issue #70. `open_hour`/`close_hour` are one pair of
    integers and the tag is a grammar: 94% of the catalogue's tags name a day of
    the week, 53% carry several rules, 17% close for lunch. Collapsing that to a
    pair made `Tu-Su 10:00-18:00; Mo off` indistinguishable from "open 10-18
    every day", so a museum shut on Mondays was schedulable on a Monday.

    Returns None when the tag cannot be parsed -- 27 of the catalogue's 4,404
    distinct tags are malformed OSM (`none`, French day names, `May 1-Aug: 31`)
    -- and the caller falls back to the coarse pair rather than dropping the
    stop."""
    midnight = datetime.datetime.combine(day_date, datetime.time())
    try:
        raw_intervals = OpeningHours(str(raw)).intervals(
            midnight, midnight + datetime.timedelta(days=_OPENING_HORIZON_DAYS + 1))
    except Exception:
        return None

    # `intervals()` returns closed stretches too. The state's repr is
    # "State.OPEN" but its str() is "open" -- comparing against the repr silently
    # matched nothing and made every tagged POI look permanently shut.
    hours = []
    for start, end, state, _comment in raw_intervals:
        if str(state).lower() != "open":
            continue
        hours.append(((start - midnight).total_seconds() / 3600,
                      (end - midnight).total_seconds() / 3600))
    hours.sort()

    # An interval that runs to midnight and one that starts there are the same
    # evening: 18:00-24:00 plus 24:00-02:00 is a bar open until 02:00, and
    # keeping them apart would make the second look like tomorrow's opening.
    merged: list[tuple[float, float]] = []
    for start, end in hours:
        if merged and start <= merged[-1][1] + 1e-9:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    # Whatever still begins tomorrow is tomorrow's opening, and waiting out a
    # night for it is not an itinerary -- same rule as the coarse pair's horizon.
    return [(s, e) for s, e in merged if s < 24]


def _opening_intervals(poi: dict, earliest_hour: float = None,
                        day_date=None) -> list[tuple[float, float]]:
    """The hours a POI is open, on the itinerary day's own clock, where 26.0
    means 02:00 the following morning.

    Prefers the verbatim OSM tag (`opening_hours_raw`, issue #70) resolved
    against the day's actual date, because only that knows which day of the week
    it is. Falls back to the coarse `open_hour`/`close_hour` pair when the
    catalogue has no tag for the row, when the tag is malformed, or when the
    caller has no date to resolve it against.

    In the coarse case, a venue whose close_hour is below its open_hour spans
    midnight, so its interval has to *end* past 24 rather than being clamped to
    it. Issue #59 flagged that clamp as a real limitation once days could run to
    06:00 and left it documented; #61 closed it. Measured on the #59 day model,
    the clamp is what made a 15-hour and an 18-hour day from 12:00 return
    identical plans -- both stopped at 23:49, because every 18:00-02:00 bar was
    considered shut after midnight.

    `earliest_hour` is NIGHTLIFE_EARLIEST_HOUR's category floor from #59."""
    intervals = None
    raw = poi.get("opening_hours_raw")
    if raw and day_date is not None:
        intervals = _tag_intervals(raw, day_date)

    if intervals is None:
        open_h = poi.get("open_hour", 0)
        close_h = poi.get("close_hour", 24)
        if close_h < open_h:
            close_h += 24
        intervals = [(open_h + 24 * d, close_h + 24 * d)
                     for d in range(_OPENING_HORIZON_DAYS)]

    if earliest_hour is not None:
        intervals = [(max(s, earliest_hour), e) for s, e in intervals]
    return [(s, e) for s, e in intervals if s < e]


def _next_open_hour(poi: dict, arrival: float, earliest_hour: float = None,
                     day_date=None) -> float | None:
    """When the traveler can actually walk in, having arrived at `arrival`:
    `arrival` itself if the doors are open, the next opening if they are not
    open yet, or None if the venue does not open again within the horizon."""
    for start, end in _opening_intervals(poi, earliest_hour, day_date):
        if arrival < end:
            return max(arrival, start)
    return None


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _haversine_distance_fn(a: dict, b: dict) -> float:
    return haversine_km(a["lat"], a["lon"], b["lat"], b["lon"])


def _build_distance_functions(points: list[dict], use_real_routing: bool, travel_mode=DEFAULT_MODE):
    """Returns (distance_km_fn, duration_min_fn, used_real_routing). Falls
    back to the mode's haversine estimate whenever real routing wasn't
    requested, or was requested but OSRM couldn't be reached -- callers never
    need to know which case they're in."""
    mode = get_travel_mode(travel_mode)

    def haversine_duration_fn(a, b):
        return mode.leg_minutes(_haversine_distance_fn(a, b))

    if not use_real_routing:
        return _haversine_distance_fn, haversine_duration_fn, False

    if isinstance(mode, HybridTravelMode):
        built = _build_hybrid_matrix_functions(points, mode)
    else:
        built = _build_single_profile_matrix_functions(points, mode)
    if built is None:
        return _haversine_distance_fn, haversine_duration_fn, False
    return (*built, True)


def _matrix_lookup(points: list[dict]):
    """POI dicts aren't hashable, so legs are indexed by object identity --
    every caller passes the same dict objects it built the matrix from."""
    return {id(p): i for i, p in enumerate(points)}


def _build_single_profile_matrix_functions(points: list[dict], mode):
    result = fetch_distance_duration_matrix(points, profile=mode.osrm_profile)
    if result is None:
        return None
    distances_km, durations_min = result
    index = _matrix_lookup(points)

    def matrix_distance_fn(a, b):
        return distances_km[index[id(a)]][index[id(b)]]

    def matrix_duration_fn(a, b):
        # OSRM reports pure travel time; the mode's per-stop overhead (parking
        # and the walk in from it, for driving) is real time the traveler
        # spends and has to be added on top of it.
        return durations_min[index[id(a)]][index[id(b)]] + mode.stop_overhead_min

    return matrix_distance_fn, matrix_duration_fn


def _build_hybrid_matrix_functions(points: list[dict], mode: HybridTravelMode):
    """Hybrid needs both profiles: the walking network decides how far a leg
    really is on foot, and legs past the threshold are then priced on the
    road network. If either request fails there is no honest way to mix the
    two, so the caller falls back to the haversine hybrid estimate."""
    foot = fetch_distance_duration_matrix(points, profile=mode.walk.osrm_profile)
    car = fetch_distance_duration_matrix(points, profile=mode.drive.osrm_profile)
    if foot is None or car is None:
        return None
    foot_km, foot_min = foot
    car_km, car_min = car
    index = _matrix_lookup(points)

    def _pick(a, b):
        i, j = index[id(a)], index[id(b)]
        if foot_km[i][j] <= mode.threshold_km:
            return foot_km[i][j], foot_min[i][j] + mode.walk.stop_overhead_min
        return car_km[i][j], car_min[i][j] + mode.drive.stop_overhead_min

    return (lambda a, b: _pick(a, b)[0]), (lambda a, b: _pick(a, b)[1])


def _route_length(route: list[dict], distance_fn) -> float:
    return sum(distance_fn(route[i], route[i + 1]) for i in range(len(route) - 1))


def _nearest_neighbor(pois: list[dict], distance_fn, start: dict = None) -> list[dict]:
    remaining = pois[:]
    route = []
    current = start or remaining.pop(0)
    if start is None:
        route.append(current)
    while remaining:
        nxt = min(remaining, key=lambda p: distance_fn(current, p))
        route.append(nxt)
        remaining.remove(nxt)
        current = nxt
    return route


def _two_opt(route: list[dict], distance_fn, start_hub: dict = None) -> list[dict]:
    """2-opt local search over a path, scored by the change a reversal makes
    rather than by re-measuring the whole route each time. Reversing
    best[i..j] only ever swaps the two edges at the segment's ends, so the
    delta is O(1) -- which is what makes it affordable to keep optimizing the
    longer days that issue #19's budget-filling now produces, instead of
    bailing out at the old 9-stop cap.

    `start_hub` is pinned at the front as a fixed depot rather than being
    left out: the day really does begin by travelling from the hotel to the
    first stop, so leaving that leg out of the objective let the search pick
    a first stop on the wrong side of town and call it optimal."""
    # Index 0 is held fixed: it is the depot when there is one, and otherwise
    # the day's chosen opening stop.
    best = ([start_hub] if start_hub else []) + list(route)
    n = len(best)
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                before = distance_fn(best[i - 1], best[i])
                after = distance_fn(best[i - 1], best[j])
                if j + 1 < n:  # the tail edge only exists if the segment isn't the route's end
                    before += distance_fn(best[j], best[j + 1])
                    after += distance_fn(best[i], best[j + 1])
                if after < before - 1e-9:
                    best[i:j + 1] = best[i:j + 1][::-1]
                    improved = True
    return best[1:] if start_hub else best


def _is_food(poi: dict) -> bool:
    return poi.get("category") == FOOD_CATEGORY


def _is_nightlife(poi: dict) -> bool:
    return poi.get("category") == NIGHTLIFE_CATEGORY


def _nightlife_last(route: list[dict]) -> list[dict]:
    """Move nightlife stops to the end of the day, keeping relative order.

    Applied after 2-opt rather than inside it: the geographic solve stays a
    clean TSP, and this expresses the one thing geography cannot know -- that
    a bar is where a day ends. It does cost distance, and that trade is the
    right way round; nobody saves a detour by going clubbing before breakfast.
    """
    evening = [p for p in route if _is_nightlife(p)]
    if not evening:
        return route
    return [p for p in route if not _is_nightlife(p)] + evening


def optimize_day_route(pois: list[dict], start_hub: dict = None, daily_minutes_budget: int = 480,
                        day_start_hour: float = 9.0, respect_opening_hours: bool = True,
                        use_real_routing: bool = False, distance_fn=None, duration_fn=None,
                        used_real_routing: bool = None, travel_mode=DEFAULT_MODE,
                        preserve_order: bool = False, day_date=None) -> dict:
    """Returns an ordered subset of `pois` that fits the time budget (travel +
    wait-for-opening + visit time), plus total distance/time, a per-stop
    schedule, and which distance source was actually used.

    travel_mode ("walking"/"driving"/"hybrid") sets how a leg is costed, so
    the same set of stops yields a fuller day when the traveler is driving
    than when they are on foot.

    `day_date` is the calendar date this day falls on, and it is what makes
    opening hours mean anything: without it a POI's hours can only be read as
    the coarse open/close pair, which cannot say "shut on Mondays" (issue #70).
    Left out, the coarse pair is used and the result is the pre-#70 behaviour.

    preserve_order keeps `pois` in the order given instead of re-solving the
    TSP. Meal placement (issue #20) needs it: a meal is deliberately put at
    the point in the day where the clock reaches lunch or dinner time, and
    re-optimizing on pure geography would immediately move it back to
    wherever it happens to be closest, which is how you end up with two
    lunches before 11am.

    distance_fn/duration_fn/used_real_routing let build_multi_day_itinerary
    share a single OSRM matrix fetch across every day of a trip; if omitted
    (e.g. calling this function standalone), one is built just for this
    day's points."""
    if not pois:
        return {"route": [], "distance_km": 0.0, "total_minutes": 0, "active_minutes": 0,
                "idle_minutes": 0, "schedule": [], "used_real_routing": False}

    if distance_fn is None or duration_fn is None:
        all_points = ([start_hub] if start_hub else []) + pois
        distance_fn, duration_fn, used_real_routing = _build_distance_functions(
            all_points, use_real_routing, travel_mode)

    if preserve_order:
        ordered = list(pois)
    else:
        ordered = _nearest_neighbor(pois, distance_fn, start=start_hub)
        ordered = _two_opt(ordered, distance_fn, start_hub=start_hub)
        ordered = _nightlife_last(ordered)

    kept, schedule, total_km, clock, prev = [], [], 0.0, day_start_hour, start_hub
    # Time actually spent travelling and inside venues, tracked alongside the
    # elapsed clock. The budget still means wall-clock day length (#59's "Time
    # out per day"), but "how full is this day?" is a different question from
    # "how late is it?", and answering it with the clock is what starved
    # evening days: see _fill_days_to_budget (#61).
    active = 0.0
    for poi in ordered:
        leg_km = distance_fn(prev, poi) if prev else 0.0
        leg_minutes = duration_fn(prev, poi) if prev else 0.0
        arrival = clock + leg_minutes / 60

        if respect_opening_hours:
            # A nightlife venue is not worth visiting the moment its doors
            # unlock. Several carry early OSM hours (07:00 for a bar), which
            # made them legitimately schedulable at breakfast time; the
            # category has an earliest *sensible* hour of its own, and it
            # overrides the stated one where that is earlier (issue #59). A
            # day too short to reach it simply doesn't get a nightlife stop,
            # which is the right answer rather than a 15:00 club visit.
            earliest = NIGHTLIFE_EARLIEST_HOUR if _is_nightlife(poi) else None
            # _next_open_hour keeps past-midnight closing times past midnight
            # instead of clamping them to 24:00 (#61); it returns None only
            # when the venue genuinely does not open again in reach.
            arrival = _next_open_hour(poi, arrival, earliest, day_date)
            if arrival is None:
                continue  # never open again in reach -- skip this stop

        visit_minutes = poi.get("avg_visit_minutes", 60)
        finish = arrival + visit_minutes / 60
        elapsed_minutes = (finish - day_start_hour) * 60
        if elapsed_minutes > daily_minutes_budget:
            continue

        clock = finish
        active += leg_minutes + visit_minutes
        total_km += leg_km
        kept.append(poi)
        schedule.append({"arrival": arrival, "finish": finish})
        prev = poi

    # Derived from each other rather than rounded independently, so the three
    # always reconcile and no caller can show a breakdown that doesn't add up.
    span_minutes = int(round((clock - day_start_hour) * 60))
    active_minutes = min(int(round(active)), span_minutes)
    return {
        "route": kept,
        "distance_km": round(total_km, 2),
        "total_minutes": span_minutes,
        "active_minutes": active_minutes,
        "idle_minutes": span_minutes - active_minutes,
        "schedule": schedule,
        "used_real_routing": used_real_routing,
    }


def _demo_city():
    import pandas as pd
    from pathlib import Path as _P
    d = _P(__file__).resolve().parents[1] / "data" / "destinations.csv"
    return pd.read_csv(d).destination_id.iloc[0]


if __name__ == "__main__":
    # One day over a city's whole catalogue, to see the primitive on its own.
    # For a real trip use: python -m roamwise.optimization.toptw
    from roamwise.knowledge_graph.build_graph import GraphIndex

    idx = GraphIndex()
    day = optimize_day_route(idx.city_pois(_demo_city())[:40], daily_minutes_budget=600)
    print(f"{len(day['route'])} stops, {day['distance_km']}km, "
          f"{day['active_minutes']}min busy of {day['total_minutes']}min out")
    for poi, slot in zip(day["route"], day["schedule"]):
        print(f"  {int(slot['arrival']):02d}:{int(slot['arrival'] % 1 * 60):02d}  {poi['name']}")
