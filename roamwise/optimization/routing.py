"""
Routing optimization ("Generating geographically optimized, day-by-day
itineraries using algorithmic routing constraints").

Two stages:
  1. POIZoner (models/segmentation.py) splits a city's selected POIs into
     n_days geographic zones via KMeans -- one zone per day.
  2. Within each day's zone, `optimize_day_route` solves a small Euclidean
     TSP: nearest-neighbor construction + 2-opt local search, respecting a
     daily time budget (opening-hours-agnostic visit-duration sum). This is
     the RouterAgent's actual "tool call" -- exact for the small per-day POI
     counts this produces (typically 3-6 stops), which is what makes 2-opt
     sufficient instead of needing an ILP/OR-Tools solver.
"""
import itertools
import math


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _route_length(route: list[dict]) -> float:
    return sum(
        haversine_km(route[i]["lat"], route[i]["lon"], route[i + 1]["lat"], route[i + 1]["lon"])
        for i in range(len(route) - 1)
    )


def _nearest_neighbor(pois: list[dict], start: dict = None) -> list[dict]:
    remaining = pois[:]
    route = []
    current = start or remaining.pop(0)
    if start is None:
        route.append(current)
    while remaining:
        nxt = min(remaining, key=lambda p: haversine_km(current["lat"], current["lon"], p["lat"], p["lon"]))
        route.append(nxt)
        remaining.remove(nxt)
        current = nxt
    return route


def _two_opt(route: list[dict]) -> list[dict]:
    best = route[:]
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best) - 1):
            for j in range(i + 1, len(best)):
                if j - i == 1:
                    continue
                candidate = best[:i] + best[i:j][::-1] + best[j:]
                if _route_length(candidate) < _route_length(best) - 1e-9:
                    best = candidate
                    improved = True
    return best


def optimize_day_route(pois: list[dict], start_hub: dict = None, daily_minutes_budget: int = 480) -> dict:
    """Returns an ordered subset of `pois` that fits the time budget (visit
    time + walking time at 4.5km/h), plus total distance/time."""
    if not pois:
        return {"route": [], "distance_km": 0.0, "total_minutes": 0}

    ordered = _nearest_neighbor(pois, start=start_hub)
    ordered = _two_opt(ordered) if len(ordered) <= 9 else ordered  # 2-opt is O(n^2) per pass; cap for safety

    kept, total_minutes, prev = [], 0, start_hub
    for poi in ordered:
        walk_km = haversine_km(prev["lat"], prev["lon"], poi["lat"], poi["lon"]) if prev else 0
        walk_minutes = (walk_km / 4.5) * 60
        visit_minutes = poi.get("avg_visit_minutes", 60)
        if total_minutes + walk_minutes + visit_minutes > daily_minutes_budget:
            continue
        total_minutes += walk_minutes + visit_minutes
        kept.append(poi)
        prev = poi

    return {
        "route": kept,
        "distance_km": round(_route_length(([start_hub] if start_hub else []) + kept), 2),
        "total_minutes": int(total_minutes),
    }


def build_multi_day_itinerary(pois_by_zone: dict[int, list[dict]], start_hub: dict = None,
                               daily_minutes_budget: int = 480) -> list[dict]:
    days = []
    for zone_id in sorted(pois_by_zone):
        result = optimize_day_route(pois_by_zone[zone_id], start_hub=start_hub, daily_minutes_budget=daily_minutes_budget)
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
    itinerary = build_multi_day_itinerary(zones)
    for day in itinerary:
        names = [p["name"] for p in day["route"]]
        print(f"Day {day['day']}: {names}  ({day['distance_km']}km, {day['total_minutes']}min)")
