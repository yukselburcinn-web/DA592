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


def optimize_day_route(pois: list[dict], start_hub: dict = None, daily_minutes_budget: int = 480,
                        day_start_hour: float = 9.0, respect_opening_hours: bool = True,
                        use_real_routing: bool = False, distance_fn=None, duration_fn=None,
                        used_real_routing: bool = None, travel_mode=DEFAULT_MODE) -> dict:
    """Returns an ordered subset of `pois` that fits the time budget (travel +
    wait-for-opening + visit time), plus total distance/time and which
    distance source was actually used.

    travel_mode ("walking"/"driving"/"hybrid") sets how a leg is costed, so
    the same set of stops yields a fuller day when the traveler is driving
    than when they are on foot.

    distance_fn/duration_fn/used_real_routing let build_multi_day_itinerary
    share a single OSRM matrix fetch across every day of a trip; if omitted
    (e.g. calling this function standalone), one is built just for this
    day's points."""
    if not pois:
        return {"route": [], "distance_km": 0.0, "total_minutes": 0, "used_real_routing": False}

    if distance_fn is None or duration_fn is None:
        all_points = ([start_hub] if start_hub else []) + pois
        distance_fn, duration_fn, used_real_routing = _build_distance_functions(
            all_points, use_real_routing, travel_mode)

    ordered = _nearest_neighbor(pois, distance_fn, start=start_hub)
    ordered = _two_opt(ordered, distance_fn, start_hub=start_hub)

    kept, total_km, clock, prev = [], 0.0, day_start_hour, start_hub
    for poi in ordered:
        leg_km = distance_fn(prev, poi) if prev else 0.0
        leg_minutes = duration_fn(prev, poi) if prev else 0.0
        arrival = clock + leg_minutes / 60

        if respect_opening_hours:
            open_h = poi.get("open_hour", 0)
            close_h = poi.get("close_hour", 24)
            # Venues open past midnight (close_h < open_h, e.g. nightlife
            # 18:00-02:00) are treated as open through the end of the
            # itinerary's day window -- days here never run past ~23:00, so
            # the wraparound to the next calendar day never matters in
            # practice and modeling it exactly would need a real time-window
            # TSP.
            effective_close = 24.0 if close_h < open_h else close_h
            if arrival >= effective_close:
                continue  # closed for the rest of the day -- skip this stop
            if arrival < open_h:
                arrival = open_h  # wait for opening

        visit_minutes = poi.get("avg_visit_minutes", 60)
        finish = arrival + visit_minutes / 60
        elapsed_minutes = (finish - day_start_hour) * 60
        if elapsed_minutes > daily_minutes_budget:
            continue

        clock = finish
        total_km += leg_km
        kept.append(poi)
        prev = poi

    return {
        "route": kept,
        "distance_km": round(total_km, 2),
        "total_minutes": int((clock - day_start_hour) * 60),
        "used_real_routing": used_real_routing,
    }


def build_multi_day_itinerary(pois_by_zone: dict[int, list[dict]], start_hub: dict = None,
                               daily_minutes_budget: int = 480, day_start_hour: float = 9.0,
                               respect_opening_hours: bool = True, use_real_routing: bool = False,
                               travel_mode=DEFAULT_MODE, fill_days: bool = True) -> list[dict]:
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
    one-zone-one-day behaviour."""
    # Fetch one OSRM matrix for every POI across every day of the trip, not
    # one per day -- the public demo server rate-limits back-to-back
    # requests, and one matrix covering the whole trip is also just the
    # architecturally correct amount of network I/O for this.
    all_pois = [p for zone in pois_by_zone.values() for p in zone]
    all_points = ([start_hub] if start_hub else []) + all_pois
    distance_fn, duration_fn, used_real_routing = _build_distance_functions(
        all_points, use_real_routing, travel_mode)

    def route_day(day_pois):
        return optimize_day_route(
            day_pois, start_hub=start_hub, daily_minutes_budget=daily_minutes_budget,
            day_start_hour=day_start_hour, respect_opening_hours=respect_opening_hours,
            distance_fn=distance_fn, duration_fn=duration_fn, used_real_routing=used_real_routing,
        )

    days = []
    for zone_id in sorted(pois_by_zone):
        days.append({"day": zone_id + 1, "_assigned": list(pois_by_zone[zone_id]),
                     **route_day(pois_by_zone[zone_id])})

    if fill_days:
        pool = _fill_days_to_budget(days, route_day, daily_minutes_budget, distance_fn, start_hub)
        _rebalance_days(days, route_day, distance_fn, start_hub, pool)

    for day in days:
        day.pop("_assigned", None)
    return days


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
    than duplicating that logic."""
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
        for day in sorted(days, key=lambda d: (d["total_minutes"], d["day"])):
            if day["total_minutes"] >= daily_minutes_budget:
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
    evens the pair out: the sum of the two days' squared durations must
    strictly drop. That sum is bounded below and strictly decreases on every
    accepted move, so this terminates rather than shuffling one stop back and
    forth forever."""
    pool = pool if pool is not None else []

    progress = True
    while progress:
        progress = False
        receiver = min(days, key=lambda d: (d["total_minutes"], d["day"]))
        donors = sorted((d for d in days if len(d["route"]) > 1 and d is not receiver),
                        key=lambda d: (-d["total_minutes"], d["day"]))

        for donor in donors:
            before = donor["total_minutes"] ** 2 + receiver["total_minutes"] ** 2
            anchor = receiver["route"][-1] if receiver["route"] else start_hub
            candidates = (sorted(donor["route"], key=lambda p: distance_fn(anchor, p))
                          if anchor else list(donor["route"]))

            for cand in candidates:
                grown = route_day(receiver["route"] + [cand])
                if len(grown["route"]) <= len(receiver["route"]):
                    continue  # the receiver can't actually host it (hours, budget)
                shrunk = route_day([p for p in donor["route"] if id(p) != id(cand)])
                if shrunk["total_minutes"] ** 2 + grown["total_minutes"] ** 2 >= before:
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


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from knowledge_graph.build_graph import GraphIndex
    from models.segmentation import POIZoner

    idx = GraphIndex()
    pois = idx.city_pois("ROM")
    zoner = POIZoner()
    zones = zoner.zone(pois, n_zones=3)
    itinerary = build_multi_day_itinerary(zones, use_real_routing=True)
    for day in itinerary:
        names = [p["name"] for p in day["route"]]
        print(f"Day {day['day']}: {names}  ({day['distance_km']}km, {day['total_minutes']}min, "
              f"real_routing={day['used_real_routing']})")
