"""
Routing optimization ("Generating geographically optimized, day-by-day
itineraries using algorithmic routing constraints").

Three stages:
  1. POIZoner (models/segmentation.py) splits a city's selected POIs into
     n_days geographic zones via KMeans -- one zone per day.
  2. Within each day's zone, `optimize_day_route` solves a small Euclidean
     TSP: nearest-neighbor construction + 2-opt local search. This is the
     RouterAgent's actual "tool call" -- exact for the small per-day POI
     counts this produces (typically 3-6 stops), which is what makes 2-opt
     sufficient instead of needing an ILP/OR-Tools solver.
  3. `build_multi_day_itinerary` then balances the days against each other
     (issue #19), because stages 1 and 2 together decide where each day goes
     but nothing decides how full it is -- see that function's docstring.

Constraints beyond pure geography:
  - Real street-network distance/duration via OSRM (`use_real_routing=True`),
    instead of the haversine straight-line estimate (issue #8). Opt-in and
    network-dependent -- see osrm_client.py's docstring for why, and every
    caller falls back to haversine automatically if the OSRM request fails.
  - POI opening hours (`respect_opening_hours=True`, the default): a stop
    that isn't open yet makes the router wait; a stop that's already closed
    for the day gets skipped. The 2-opt geographic ordering itself still
    ignores opening hours (a true time-window TSP is out of scope here) --
    opening hours are enforced as a second pass over the geographically
    optimized order, which is a real but bounded improvement, not full
    time-window vehicle routing.
  - Travel mode (`travel_mode=`, issue #19): legs are costed as walking,
    driving or a per-leg hybrid of the two rather than always assuming a flat
    walking speed. See travel_modes.py.
"""
import math

from roamwise.optimization.osrm_client import fetch_distance_duration_matrix
from roamwise.optimization.travel_modes import DEFAULT_MODE, HybridTravelMode, get_travel_mode

FOOD_CATEGORY = "food"
# When a day begins, unless the traveler says otherwise. A default, not a
# constraint -- it was effectively the latter until issue #59, because nothing
# above RouterAgent.run() could pass a different one.
DEFAULT_DAY_START_HOUR = 9.0
# Fraction of a meal's sit-down time to also set aside for getting to it.
MEAL_TRAVEL_ALLOWANCE = 0.25
# How many sights a meal may bump to fit into a day. The guarantee is "every
# day has meals", so a meal outranks the marginal last sight -- but only just:
# past this the day stops being a sightseeing itinerary. See #29: this cap
# existed to bound the damage, but on a sparse day (few sightseeing stops to
# begin with, e.g. a Nightlife Seeker's late-opening zone) two meals'
# worth of displacement can still empty a day out completely -- see
# MIN_SIGHTSEEING_STOPS below, which is the actual fix for that.
MAX_MEAL_DISPLACEMENTS = 2
# How far from its target time a meal may slide to find a venue the traveler
# is already passing, instead of detouring to one that is on time.
MEAL_WINDOW_HOURS = 2.0
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
# A day must keep at least this many non-food stops beyond its meals, or
# "2 meals/day" (#20) and "food-only day" can both be technically true of
# the same result (#29 comment). When keeping this floor and reaching the
# meal-count guarantee conflict, the floor wins -- a day with a real sight
# and only one meal is a more plausible travel day than an emptied-out one
# with two.
MIN_SIGHTSEEING_STOPS = 1
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


def _opening_intervals(poi: dict, earliest_hour: float = None) -> list[tuple[float, float]]:
    """The hours a POI is open, on the itinerary day's own clock, where 26.0
    means 02:00 the following morning.

    A venue whose close_hour is below its open_hour spans midnight, so its
    interval has to *end* past 24 rather than being clamped to it. Issue #59
    flagged that clamp as a real limitation once days could run to 06:00 and
    left it documented; this is that follow-up (#61). Measured on the #59 day
    model, it is what made a 15-hour and an 18-hour day from 12:00 return
    identical plans -- both stop at 23:49, because the next venue would arrive
    after midnight and every 18:00-02:00 bar was considered shut by then.

    `earliest_hour` is NIGHTLIFE_EARLIEST_HOUR's category floor from #59,
    applied per calendar day so it still holds for tomorrow's opening."""
    open_h = poi.get("open_hour", 0)
    close_h = poi.get("close_hour", 24)
    if close_h < open_h:
        close_h += 24
    intervals = []
    for day in range(_OPENING_HORIZON_DAYS):
        start, end = open_h + 24 * day, close_h + 24 * day
        if earliest_hour is not None:
            start = max(start, earliest_hour + 24 * day)
        if start < end:
            intervals.append((start, end))
    return intervals


def _next_open_hour(poi: dict, arrival: float, earliest_hour: float = None) -> float | None:
    """When the traveler can actually walk in, having arrived at `arrival`:
    `arrival` itself if the doors are open, the next opening if they are not
    open yet, or None if the venue does not open again within the horizon."""
    for start, end in _opening_intervals(poi, earliest_hour):
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
                        preserve_order: bool = False) -> dict:
    """Returns an ordered subset of `pois` that fits the time budget (travel +
    wait-for-opening + visit time), plus total distance/time, a per-stop
    schedule, and which distance source was actually used.

    travel_mode ("walking"/"driving"/"hybrid") sets how a leg is costed, so
    the same set of stops yields a fuller day when the traveler is driving
    than when they are on foot.

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
            arrival = _next_open_hour(poi, arrival, earliest)
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


def build_multi_day_itinerary(pois_by_zone: dict[int, list[dict]], start_hub: dict = None,
                               daily_minutes_budget: int = 480, day_start_hour: float = 9.0,
                               respect_opening_hours: bool = True, use_real_routing: bool = False,
                               travel_mode=DEFAULT_MODE, fill_days: bool = True,
                               food_pois: list[dict] = None, min_food_per_day: int = 0) -> list[dict]:
    """Zones in, one routed day per zone out.

    Geographic zoning alone decides *where* each day goes but not *how full*
    it is, and a zone whose POIs don't fit the day window (a nightlife venue
    that opens at 18:00 against a 09:00-17:00 day) used to leave that day
    empty while other zones had stops to spare. `fill_days` therefore runs
    two balancing passes afterwards: `_fill_days_to_budget` pools every POI
    no day actually used and feeds it to the emptiest day that can take it,
    and `_rebalance_days` then moves stops between days for the case where
    the pool is empty but one day is still far fuller than another. Together
    they keep each day near its budget instead of leaving a 5-day trip at a
    quarter of the time the traveler asked for. Set it False to get the raw
    one-zone-one-day behaviour.

    `min_food_per_day` (issue #20) guarantees each day that many meal stops,
    drawn from `food_pois` and placed at meal times. Their time is reserved
    *before* the sightseeing passes run, so meals get added to a day that has
    room for them rather than displacing sights the traveler was already
    promised."""
    # Fetch one OSRM matrix for every POI across every day of the trip, not
    # one per day -- the public demo server rate-limits back-to-back
    # requests, and one matrix covering the whole trip is also just the
    # architecturally correct amount of network I/O for this.
    all_pois = [p for zone in pois_by_zone.values() for p in zone]
    meal_pois = list(food_pois or []) if min_food_per_day > 0 else []
    all_points = ([start_hub] if start_hub else []) + all_pois + meal_pois
    distance_fn, duration_fn, used_real_routing = _build_distance_functions(
        all_points, use_real_routing, travel_mode)

    sightseeing_budget = daily_minutes_budget - _meal_time_reserve(
        meal_pois, min_food_per_day, daily_minutes_budget)

    def make_router(budget, preserve_order=False):
        def route(day_pois):
            return optimize_day_route(
                day_pois, start_hub=start_hub, daily_minutes_budget=budget,
                day_start_hour=day_start_hour, respect_opening_hours=respect_opening_hours,
                distance_fn=distance_fn, duration_fn=duration_fn,
                used_real_routing=used_real_routing, preserve_order=preserve_order,
            )
        return route

    route_day = make_router(sightseeing_budget)

    days = []
    for zone_id in sorted(pois_by_zone):
        days.append({"day": zone_id + 1, "_assigned": list(pois_by_zone[zone_id]),
                     **route_day(pois_by_zone[zone_id])})

    pool = []
    if fill_days:
        pool = _fill_days_to_budget(days, route_day, sightseeing_budget, distance_fn, start_hub)
        _rebalance_days(days, route_day, distance_fn, start_hub, pool)

    # Evening venues cannot be placed by the passes above, and not because the
    # day is full: those route against `sightseeing_budget`, which is the day
    # minus the meal reserve, so on a 12-hour day from 09:00 they stop at
    # ~18:30 -- the exact moment NIGHTLIFE_EARLIEST_HOUR lets a bar open. The
    # stop is squeezed out by time reserved for a meal it would sit after, and
    # a Nightlife Seeker's day (whose candidates are mostly bars) collapsed to
    # 32% utilisation with zero nightlife stops. So they get the same
    # treatment meals already get: a pass of their own, against the full
    # budget, over an order that is already settled.
    _ensure_evening_stops(days, pool, make_router(daily_minutes_budget, preserve_order=True),
                          distance_fn, start_hub)

    if meal_pois:
        # Meals are timed into an order that is already settled, so this pass
        # (and only this pass) routes with the full budget and without
        # re-solving the geography.
        _ensure_daily_meals(days, meal_pois, make_router(daily_minutes_budget, preserve_order=True),
                            distance_fn, min_food_per_day, daily_minutes_budget,
                            day_start_hour, start_hub)

    for day in days:
        day.pop("_assigned", None)
    return days


def _ensure_evening_stops(days: list[dict], pool: list[dict], route_day_ordered,
                           distance_fn, start_hub: dict = None) -> None:
    """Give each day one nightlife stop from the leftover pool, at the end.

    Only ever appends, and only from POIs no day used -- so a day that already
    ends at a bar is left alone, and nothing is taken away from another day to
    supply one. A day that cannot reach NIGHTLIFE_EARLIEST_HOUR within its
    budget simply keeps none, which is the honest answer rather than a 15:00
    club visit (see optimize_day_route's opening-hours branch).
    """
    available = [p for p in pool if _is_nightlife(p)]
    if not available:
        return

    for day in days:
        if any(_is_nightlife(p) for p in day["route"]):
            continue
        anchor = day["route"][-1] if day["route"] else start_hub
        candidates = (sorted(available, key=lambda p: distance_fn(anchor, p))
                      if anchor else list(available))
        for cand in candidates:
            attempt = route_day_ordered(day["route"] + [cand])
            if not any(_is_nightlife(p) for p in attempt["route"]):
                continue  # didn't fit the budget -- the day ends too early
            day.update(attempt)
            available = [p for p in available if id(p) != id(cand)]
            break


def _meal_time_reserve(food_pois: list[dict], min_food_per_day: int, daily_minutes_budget: int) -> int:
    """Minutes to hold back from sightseeing so the day's meals fit.

    A meal costs more than the time spent eating: the traveler also has to
    get there, and a reserve covering only the sit-down time left days
    finishing at 16:00 with 63 minutes of slack and a second meal needing 72,
    so it never fit. MEAL_TRAVEL_ALLOWANCE covers that leg.

    Capped at half the day: on a very short day it is better to return a
    thin itinerary with meals in it than to hand back a day that is nothing
    but lunch and dinner."""
    if not food_pois or min_food_per_day <= 0:
        return 0
    visits = sorted(p.get("avg_visit_minutes", 60) for p in food_pois)
    typical = visits[len(visits) // 2] * (1 + MEAL_TRAVEL_ALLOWANCE)
    return int(min(min_food_per_day * typical, daily_minutes_budget * 0.5))


def _meal_target_hours(day_start_hour: float, daily_minutes_budget: int, n_meals: int) -> list[float]:
    """Clock times to aim each meal at, as fractions of the day's own window.

    Fixed clock times (13:00, 19:00) would be wrong for a day that runs
    09:00-15:00 -- dinner falls outside it entirely and the meal is simply
    dropped. Fractions of the actual window degrade sensibly instead: on a
    full 12-hour day from 09:00 they land at 13:48 and 19:12, real lunch and
    dinner; on a short 6-hour day they become an early and a late lunch,
    which is the honest answer for someone only sightseeing until 15:00."""
    span = daily_minutes_budget / 60
    fractions = [0.40, 0.85] if n_meals <= 2 else [
        (i + 1) / (n_meals + 1) for i in range(n_meals)
    ]
    return [day_start_hour + f * span for f in fractions[:n_meals]]


def _is_food(poi: dict) -> bool:
    return poi.get("category") == FOOD_CATEGORY


def _open_at(poi: dict, hour: float) -> bool:
    """Whether the doors are open at exactly `hour`, on the same past-midnight
    clock the rest of the module uses -- so a 25.0 (01:00) meal target asks a
    kitchen that closes at 23:00 the right question."""
    return any(start <= hour < end for start, end in _opening_intervals(poi))


def _ensure_daily_meals(days: list[dict], food_pois: list[dict], route_day_ordered,
                         distance_fn, min_per_day: int, daily_minutes_budget: int,
                         day_start_hour: float, start_hub: dict = None) -> None:
    """Gives every day at least `min_per_day` food stops, placed at meal times.

    Retrieval is archetype-driven, so a Culture Enthusiast's itinerary came
    back with museums and no meals at all (issue #20). Food stops are
    therefore added here rather than by widening retrieval, which would
    distort the Fusion-vs-Hybrid-vs-standard comparison the evaluation rests
    on.

    Each meal is inserted at the point in the day where the clock is nearest
    that meal's target time, and among the food POIs open then, the one
    chosen is whichever adds the least detour to the surrounding legs
    (cheapest insertion) -- so the meal is somewhere the traveler is already
    walking past, not a cross-town round trip. Insertion order is then
    preserved when the day is re-timed, since re-solving the route on pure
    geography would undo the placement."""
    if not food_pois or min_per_day <= 0:
        return

    used = {id(p) for day in days for p in day["route"]}
    targets = _meal_target_hours(day_start_hour, daily_minutes_budget, min_per_day)
    min_gap_hours = MIN_MEAL_GAP_FRACTION * (daily_minutes_budget / 60)

    for day in days:
        # Keep going until the day is fed rather than making one pass per
        # target: a meal aimed at lunch can land late if that is where the
        # day's geography puts the traveler, and it would then sit close
        # enough to the dinner target to look like dinner was handled too.
        while _food_count(day) < min_per_day:
            open_targets = [t for t in targets if not _has_meal_near(day, t)]
            target = open_targets[0] if open_targets else _least_covered_target(day, targets)
            before = _food_count(day)
            _insert_meal_at(day, food_pois, used, target, distance_fn,
                            route_day_ordered, start_hub, min_gap_hours=min_gap_hours)
            if _food_count(day) == before:
                # Nothing placeable is left for this day at the required gap
                # / sightseeing floor -- don't spin, and don't force it: a day
                # left on 1 meal is the intended outcome here (#29), not a
                # bug to route around.
                break


def _food_count(day: dict) -> int:
    return _food_count_in(day["route"])


def _food_count_in(route: list[dict]) -> int:
    return sum(1 for p in route if _is_food(p))


def _least_covered_target(day: dict, targets: list[float]) -> float:
    """The meal slot furthest from anything the day already eats -- used when
    every slot looks nominally covered but the day is still short of its
    minimum, so the extra meal still lands in the emptiest part of the day
    instead of next to an existing one."""
    eaten = [s["arrival"] for p, s in zip(day["route"], day.get("schedule", [])) if _is_food(p)]
    if not eaten:
        return targets[0]
    return max(targets, key=lambda t: min(abs(t - e) for e in eaten))


def _meal_slots(route: list[dict], schedule: list[dict], target_hour: float,
                 start_hub: dict = None, existing_meal_times: tuple[float, ...] = (),
                 min_gap_hours: float = 0.0) -> list[int]:
    """Positions in the day where a meal could plausibly go: every seam whose
    clock is within MEAL_WINDOW_HOURS of the target and at least
    `min_gap_hours` from every meal already placed in this day, plus the end
    of the day as a last resort so a short day can still be fed -- but that
    fallback still has to respect the gap, or it recreates exactly the bug it
    exists to prevent: with no eligible seam, both meals used to fall back to
    the same end-of-day slot and land next to each other (#29)."""
    def clock_at(i: int) -> float:
        if i == 0:
            return schedule[0]["arrival"] if schedule else target_hour
        return schedule[i - 1]["finish"]

    def far_enough(i: int) -> bool:
        clock = clock_at(i)
        return all(abs(clock - t) >= min_gap_hours for t in existing_meal_times)

    seams = range(len(route) + 1)
    in_window = [i for i in seams if abs(clock_at(i) - target_hour) <= MEAL_WINDOW_HOURS]
    slots = [i for i in in_window if far_enough(i)]
    if slots:
        return slots
    last_resort = len(route)
    return [last_resort] if far_enough(last_resort) else []


def _meals_respect_gap(route: list[dict], schedule: list[dict], min_gap_hours: float) -> bool:
    """True if every pair of meals in the route's *actual, already-timed*
    schedule is at least min_gap_hours apart. Ground truth for the gap
    constraint -- see the call site in _insert_meal_at for why the seam
    estimate _meal_slots picks from isn't enough on its own."""
    times = sorted(s["arrival"] for p, s in zip(route, schedule) if _is_food(p))
    return all(times[i + 1] - times[i] >= min_gap_hours - 1e-9 for i in range(len(times) - 1))


def _has_meal_near(day: dict, target_hour: float, tolerance_hours: float = 1.25) -> bool:
    """True if the day already eats around `target_hour` -- so a food POI that
    retrieval happened to surface counts, instead of being served a second
    lunch next to it."""
    for poi, slot in zip(day["route"], day.get("schedule", [])):
        if _is_food(poi) and abs(slot["arrival"] - target_hour) <= tolerance_hours:
            return True
    return False


def _insert_meal_at(day: dict, food_pois: list[dict], used: set, target_hour: float,
                     distance_fn, route_day_ordered, start_hub: dict = None,
                     max_attempts: int = 5, min_gap_hours: float = 0.0) -> None:
    route, schedule = day["route"], day.get("schedule", [])
    candidates = [f for f in food_pois if id(f) not in used and _open_at(f, target_hour)]
    if not candidates:
        return

    existing_meal_times = tuple(s["arrival"] for p, s in zip(route, schedule) if _is_food(p))

    def detour(index, f):
        prev = route[index - 1] if index > 0 else start_hub
        nxt = route[index] if index < len(route) else None
        if prev is None:
            return distance_fn(f, nxt) if nxt else 0.0
        if nxt is None:
            return distance_fn(prev, f)
        return distance_fn(prev, f) + distance_fn(f, nxt) - distance_fn(prev, nxt)

    # Rather than pinning the meal to the single seam where the clock crosses
    # meal time, consider every seam within MEAL_WINDOW_HOURS of it and take
    # whichever (seam, venue) pair adds the least detour. Eating at 13:30
    # somewhere the traveler already walks past beats eating at 12:15 after a
    # kilometre round trip, and the timing stays plausible either way.
    # _meal_slots also excludes anything within min_gap_hours of a meal
    # already in the day, so cheapest-insertion can no longer converge both
    # meals on the same end-of-day slot (#29).
    options = [(detour(i, f), i, f)
               for i in _meal_slots(route, schedule, target_hour, start_hub,
                                     existing_meal_times, min_gap_hours)
               for f in candidates]
    if not options:
        return
    options.sort(key=lambda t: t[0])

    meals_before = _food_count(day)
    for _, index, cand in options[:max_attempts]:
        sequence = route[:index] + [cand] + route[index:]
        for _ in range(MAX_MEAL_DISPLACEMENTS + 1):
            attempt = route_day_ordered(sequence)
            # The day has to end up with *more* meals, not just with this one
            # in it: adding a midday meal pushes everything after it later,
            # and an already-placed evening meal can fall off the end of the
            # budget as it does, which would net out at no gain at all.
            if _food_count_in(attempt["route"]) > meals_before:
                # _meal_slots picked this seam from the *pre-insertion*
                # schedule's clock as an estimate; re-timing the actual
                # sequence (especially after a displacement above drops a
                # stop ahead of this meal) can pull its real arrival earlier
                # than that estimate, silently landing inside the gap this
                # was supposed to keep clear (#29). Check the real timing,
                # not the estimate that picked the seam.
                if _meals_respect_gap(attempt["route"], attempt.get("schedule", []), min_gap_hours):
                    day.update(attempt)
                    used.add(id(cand))
                    return
                break  # fits, but too close once actually timed -- try the
                       # next (seam, venue) option instead of burning
                       # displacements on a slot that was never going to work
            # It overran the day. Give up the last sight rather than the meal
            # -- a day that ends an hour early but is fed beats one that
            # sightsees straight through dinner. But not past
            # MIN_SIGHTSEEING_STOPS: that trade is only a good one while the
            # day still has a sight left to give up. Below the floor, this
            # meal doesn't fit here -- fall through to the next (seam,
            # venue) option, or leave the day on fewer meals rather than none
            # of the day at all (#29).
            non_food_count = sum(1 for p in sequence if not _is_food(p))
            droppable = next((i for i in range(len(sequence) - 1, -1, -1)
                              if id(sequence[i]) != id(cand) and not _is_food(sequence[i])), None) \
                if non_food_count > MIN_SIGHTSEEING_STOPS else None
            if droppable is None:
                break
            sequence = sequence[:droppable] + sequence[droppable + 1:]


def _fill_days_to_budget(days: list[dict], route_day, daily_minutes_budget: int,
                          distance_fn, start_hub: dict = None) -> list[dict]:
    """Moves unused POIs into whichever day has the most time left, in place,
    and returns whatever is still left over.

    Always serving the emptiest day first is what balances the trip: a day
    only gets a second stop once every other day has a first one. Candidates
    are tried nearest-first (measured from the day's last stop, or its start
    hub while the day is empty) so filling a day doesn't wreck its geography,
    and a candidate is only kept if re-routing the day actually places it --
    which is what enforces opening hours and the time budget here, rather
    than duplicating that logic.

    "Emptiest" is how much of the day the traveler is actually busy, not how
    much of the clock has passed. A day holding one 18:00 bar spans nine hours
    of which two are the visit, and ranking it by elapsed clock made it look
    like the fullest day in the trip -- so it was served last and stayed on one
    stop, which is the shape #61 reported. #59's _ensure_evening_stops worked
    around this for nightlife specifically; measuring the right thing fixes it
    for every category."""
    # POI dicts are compared by identity throughout: two different POIs can
    # hold equal values, and `list.remove`/`in` would silently confuse them.
    pool = []
    for day in days:
        routed = {id(p) for p in day["route"]}
        pool.extend(p for p in day["_assigned"] if id(p) not in routed)

    progress = True
    while pool and progress:
        progress = False
        # Emptiest day first; ties broken by day number so the result is stable.
        for day in sorted(days, key=lambda d: (d["active_minutes"], d["day"])):
            if day["active_minutes"] >= daily_minutes_budget:
                continue
            anchor = day["route"][-1] if day["route"] else start_hub
            candidates = sorted(pool, key=lambda p: distance_fn(anchor, p)) if anchor else list(pool)

            for cand in candidates:
                attempt = route_day(day["route"] + [cand])
                if len(attempt["route"]) <= len(day["route"]):
                    continue  # didn't fit: too late in the day, or closed by then
                kept_ids = {id(p) for p in attempt["route"]}
                # Re-routing can reorder the day and push a previously-kept
                # stop out; that stop goes back in the pool for another day
                # rather than disappearing from the trip.
                displaced = [p for p in day["route"] if id(p) not in kept_ids]
                pool = [p for p in pool if id(p) != id(cand)] + displaced
                day.update(attempt)
                progress = True
                break
            if progress:
                break  # re-pick the emptiest day now that this one grew
    return pool


def _rebalance_days(days: list[dict], route_day, distance_fn, start_hub: dict = None,
                     pool: list[dict] = None) -> None:
    """Evens out days by handing stops from the fullest day to the emptiest.

    Filling from the leftover pool can't help a day whose own zone held
    nothing routable while every other POI already fit elsewhere -- the pool
    is empty and the day stays blank next to a neighbour carrying five stops.
    So stops move between days too, accepted only when the move genuinely
    evens the pair out: the sum of the two days' squared *active* durations
    must strictly drop -- active for the same reason as the fill pass above,
    since waiting for a venue to open is not a full day. That sum is bounded below and strictly decreases on every
    accepted move, so this terminates rather than shuffling one stop back and
    forth forever."""
    pool = pool if pool is not None else []

    progress = True
    while progress:
        progress = False
        receiver = min(days, key=lambda d: (d["active_minutes"], d["day"]))
        donors = sorted((d for d in days if len(d["route"]) > 1 and d is not receiver),
                        key=lambda d: (-d["active_minutes"], d["day"]))

        for donor in donors:
            before = donor["active_minutes"] ** 2 + receiver["active_minutes"] ** 2
            anchor = receiver["route"][-1] if receiver["route"] else start_hub
            candidates = (sorted(donor["route"], key=lambda p: distance_fn(anchor, p))
                          if anchor else list(donor["route"]))

            for cand in candidates:
                grown = route_day(receiver["route"] + [cand])
                if len(grown["route"]) <= len(receiver["route"]):
                    continue  # the receiver can't actually host it (hours, budget)
                shrunk = route_day([p for p in donor["route"] if id(p) != id(cand)])
                if shrunk["active_minutes"] ** 2 + grown["active_minutes"] ** 2 >= before:
                    continue  # doesn't even the two days out; leave them alone

                kept_ids = {id(p) for p in grown["route"]} | {id(p) for p in shrunk["route"]}
                pool.extend(p for p in donor["route"] + receiver["route"]
                            if id(p) not in kept_ids)
                receiver.update(grown)
                donor.update(shrunk)
                progress = True
                break
            if progress:
                break


# Demo blocks below take their city from the catalogue rather than naming one:
# a hardcoded code prints nothing at all once that city stops shipping.
def _demo_city():
    import pandas as pd
    from pathlib import Path as _P
    d = _P(__file__).resolve().parents[1] / "data" / "destinations.csv"
    return pd.read_csv(d).destination_id.iloc[0]


if __name__ == "__main__":
    # Run from the repo root with: python -m roamwise.optimization.routing
    from roamwise.knowledge_graph.build_graph import GraphIndex
    from roamwise.models.segmentation import POIZoner

    idx = GraphIndex()
    pois = idx.city_pois(_demo_city())
    zoner = POIZoner()
    zones = zoner.zone(pois, n_zones=3)
    itinerary = build_multi_day_itinerary(zones, use_real_routing=True)
    for day in itinerary:
        names = [p["name"] for p in day["route"]]
        print(f"Day {day['day']}: {names}  ({day['distance_km']}km, "
              f"{day['active_minutes']}min busy of {day['total_minutes']}min out, "
              f"real_routing={day['used_real_routing']})")
