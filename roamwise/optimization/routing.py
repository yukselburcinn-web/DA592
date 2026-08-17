"""
Routing optimization ("Generating geographically optimized, day-by-day
itineraries using algorithmic routing constraints").

Two stages:
  1. POIZoner (models/segmentation.py) splits a city's selected POIs into
     n_days geographic zones via KMeans -- one zone per day.
  2. Within each day's zone, `optimize_day_route` solves a small Euclidean
     TSP: nearest-neighbor construction + 2-opt local search. This is the
     RouterAgent's actual "tool call" -- exact for the small per-day POI
     counts this produces (typically 3-6 stops), which is what makes 2-opt
     sufficient instead of needing an ILP/OR-Tools solver.

Two constraints beyond pure geography, both from issue #8:
  - Real street-network distance/duration via OSRM (`use_real_routing=True`),
    instead of haversine straight-line + a flat 4.5km/h walking-speed
    assumption. Opt-in and network-dependent -- see osrm_client.py's
    docstring for why, and every caller falls back to haversine automatically
    if the OSRM request fails.
  - POI opening hours (`respect_opening_hours=True`, the default): a stop
    that isn't open yet makes the router wait; a stop that's already closed
    for the day gets skipped. The 2-opt geographic ordering itself still
    ignores opening hours (a true time-window TSP is out of scope here) --
    opening hours are enforced as a second pass over the geographically
    optimized order, which is a real but bounded improvement, not full
    time-window vehicle routing.
"""
import math

from optimization.osrm_client import fetch_distance_duration_matrix


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _haversine_distance_fn(a: dict, b: dict) -> float:
    return haversine_km(a["lat"], a["lon"], b["lat"], b["lon"])


def _haversine_duration_fn(a: dict, b: dict) -> float:
    return (_haversine_distance_fn(a, b) / 4.5) * 60  # flat 4.5km/h walking-speed fallback


def _build_distance_functions(points: list[dict], use_real_routing: bool):
    """Returns (distance_km_fn, duration_min_fn, used_real_routing). Falls
    back to haversine whenever real routing wasn't requested, or was
    requested but OSRM couldn't be reached -- callers never need to know
    which case they're in."""
    if not use_real_routing:
        return _haversine_distance_fn, _haversine_duration_fn, False

    result = fetch_distance_duration_matrix(points)
    if result is None:
        return _haversine_distance_fn, _haversine_duration_fn, False

    distances_km, durations_min = result
    index = {id(p): i for i, p in enumerate(points)}

    def matrix_distance_fn(a, b):
        return distances_km[index[id(a)]][index[id(b)]]

    def matrix_duration_fn(a, b):
        return durations_min[index[id(a)]][index[id(b)]]

    return matrix_distance_fn, matrix_duration_fn, True


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


def _two_opt(route: list[dict], distance_fn) -> list[dict]:
    best = route[:]
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best) - 1):
            for j in range(i + 1, len(best)):
                if j - i == 1:
                    continue
                candidate = best[:i] + best[i:j][::-1] + best[j:]
                if _route_length(candidate, distance_fn) < _route_length(best, distance_fn) - 1e-9:
                    best = candidate
                    improved = True
    return best


def optimize_day_route(pois: list[dict], start_hub: dict = None, daily_minutes_budget: int = 480,
                        day_start_hour: float = 9.0, respect_opening_hours: bool = True,
                        use_real_routing: bool = False, distance_fn=None, duration_fn=None,
                        used_real_routing: bool = None) -> dict:
    """Returns an ordered subset of `pois` that fits the time budget (walk +
    wait-for-opening + visit time), plus total distance/time and which
    distance source was actually used.

    distance_fn/duration_fn/used_real_routing let build_multi_day_itinerary
    share a single OSRM matrix fetch across every day of a trip; if omitted
    (e.g. calling this function standalone), one is built just for this
    day's points."""
    if not pois:
        return {"route": [], "distance_km": 0.0, "total_minutes": 0, "used_real_routing": False}

    if distance_fn is None or duration_fn is None:
        all_points = ([start_hub] if start_hub else []) + pois
        distance_fn, duration_fn, used_real_routing = _build_distance_functions(all_points, use_real_routing)

    ordered = _nearest_neighbor(pois, distance_fn, start=start_hub)
    ordered = _two_opt(ordered, distance_fn) if len(ordered) <= 9 else ordered  # 2-opt is O(n^2) per pass; cap for safety

    kept, total_km, clock, prev = [], 0.0, day_start_hour, start_hub
    for poi in ordered:
        walk_km = distance_fn(prev, poi) if prev else 0.0
        walk_minutes = duration_fn(prev, poi) if prev else 0.0
        arrival = clock + walk_minutes / 60

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
        total_km += walk_km
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
                               respect_opening_hours: bool = True, use_real_routing: bool = False) -> list[dict]:
    # Fetch one OSRM matrix for every POI across every day of the trip, not
    # one per day -- the public demo server rate-limits back-to-back
    # requests, and one matrix covering the whole trip is also just the
    # architecturally correct amount of network I/O for this.
    all_pois = [p for zone in pois_by_zone.values() for p in zone]
    all_points = ([start_hub] if start_hub else []) + all_pois
    distance_fn, duration_fn, used_real_routing = _build_distance_functions(all_points, use_real_routing)

    days = []
    for zone_id in sorted(pois_by_zone):
        result = optimize_day_route(
            pois_by_zone[zone_id], start_hub=start_hub, daily_minutes_budget=daily_minutes_budget,
            day_start_hour=day_start_hour, respect_opening_hours=respect_opening_hours,
            distance_fn=distance_fn, duration_fn=duration_fn, used_real_routing=used_real_routing,
        )
        days.append({"day": zone_id + 1, **result})
    return days


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
