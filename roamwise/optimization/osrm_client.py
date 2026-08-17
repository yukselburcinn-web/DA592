"""
Optional real-street-network distance/duration client.

Issue #8 asked for real walking distances/times instead of the haversine
straight-line + flat 4.5km/h assumption. This uses OSRM's public foot-routing
demo server (routing.openstreetmap.de, run by the OpenStreetMap Foundation,
no API key required) rather than OpenRouteService: OpenRouteService needs a
signed-up API key we can't obtain on the user's behalf, while this endpoint
is open and matches the "no confidentiality constraints" spirit the original
proposal already applied to data sourcing.

This is opt-in (see optimize_day_route(use_real_routing=...)), not the
default: it requires network access, an external service can be slow/rate
limited/down, and the project's default posture (established by
agents/llm_client.py's offline TemplateLLMClient) is that nothing depends on
an external service unless the caller explicitly asks for it. Every caller
must treat `None` as "fall back to haversine" -- never assume the network
call succeeded.
"""
import requests

OSRM_FOOT_TABLE_URL = "https://routing.openstreetmap.de/routed-foot/table/v1/foot"
TIMEOUT_SECONDS = 6


def fetch_distance_duration_matrix(points: list[dict]):
    """points: list of {"lat": .., "lon": ..}, in the exact order callers will
    index into the result. Returns (distance_km_matrix, duration_min_matrix)
    as same-order nested lists, or None if the request fails for any reason
    (no network, timeout, rate limit, OSRM error response) -- callers must
    fall back to the haversine heuristic in that case, never raise."""
    if len(points) < 2:
        return None
    coords = ";".join(f"{p['lon']},{p['lat']}" for p in points)
    url = f"{OSRM_FOOT_TABLE_URL}/{coords}"
    try:
        resp = requests.get(url, params={"annotations": "distance,duration"}, timeout=TIMEOUT_SECONDS)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != "Ok":
            return None
        distances_km = [[d / 1000 for d in row] for row in data["distances"]]
        durations_min = [[d / 60 for d in row] for row in data["durations"]]
        return distances_km, durations_min
    except Exception:
        return None
