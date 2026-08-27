"""
Self-hosted street-network distances and times (issue #32).

This replaces `osrm_client.py`, which asked a public, unauthenticated OSRM
demo server (routing.openstreetmap.de) for every distance matrix. That server
has no uptime guarantee and rate-limits bursts, which is why real routing had
to ship opt-in and default-off: a feature that silently degrades to haversine
whenever someone else's machine is busy is not a feature you turn on for
everyone. Nothing here touches the network at run time, so that reason is
gone.

The distances come from the same place OSRM's do -- the OpenStreetMap street
network -- but they are computed here, from data committed to the repo, with
osmnx doing the download once at pipeline time (`pipeline/build_street_network.py`)
and never again. Measured against the OSRM demo server over 868 walking pairs
in Paris, this agrees to a mean 4.3% (median 2.1%); the haversine estimate it
replaces is 16.4% off the same reference.

There are two layers, and a caller gets whichever one can answer:

  1. **The precomputed matrix.** Every point RoamWise can route over is
     already known: POIs (`poi.csv`), transport hubs (`transport.csv`) and
     each city's centre (`destinations.csv`, which is what RouterAgent hands
     in as `start_hub`). Shortest paths between all of them are solved once,
     offline, and committed. At run time this is a dictionary lookup and an
     array slice -- no graph, no Dijkstra, no measurable cost, which is what
     makes real routing cheap enough to be worth defaulting on.

  2. **The network itself.** The pruned street graph is committed next to the
     matrix, so a point that is *not* in the catalogue -- a pinned coordinate,
     a POI added since the matrix was last built -- still gets a real street
     distance rather than falling back to a straight line. This costs about
     8s for 150 points in Paris and is why it is the second layer, not the
     first.

Every caller must still treat `None` as "fall back to haversine": a point
outside every city we hold a network for, or one that snaps to nothing within
`MAX_SNAP_M`, has no honest street answer and must not be given a fabricated
one.
"""
import math
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from roamwise.optimization.travel_modes import mode_for_network_profile

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "street_network"

# Which osmnx network the pipeline downloads for each routing profile, and
# whether legs on it are symmetric. Pedestrians ignore one-way restrictions,
# so the walking network is stored and solved undirected -- which also halves
# its edge count; a car's network is not, and a one-way street costs a
# different distance in each direction.
NETWORK_TYPE = {"foot": "walk", "car": "drive"}
DIRECTED = {"foot": False, "car": True}

# `transit` is a profile with no street graph behind it. Its file carries a
# timetable answer -- minutes solved by RAPTOR over GTFS
# (`pipeline/build_transit_matrix.py`, issue #32 stage 2) -- rather than a
# distance to divide by a speed, and there is nothing to fall back to for a
# point the matrix does not hold: you cannot Dijkstra your way onto a train.
# Only cities with a committed transit file offer the mode at all, which is
# why `available_cities` exists.
MATRIX_ONLY = {"transit"}

# A point is routed from the nearest node of the street network, and the walk
# from the point to that node is real distance the traveler covers. Most of
# the time it is trivial -- a venue's coordinate is its building rather than
# its door, and the median catalogue point snaps under 100m -- but a park or
# an airport is stored as the centroid of a large polygon, and its nearest
# *drivable* road can be a kilometre away (Berlin Brandenburg: 1,254m;
# Tempelhofer Feld: 863m). Those legs are not errors and must not be dropped,
# so the offset at both ends is added to every leg instead of ignored.
#
# The cap is what "we hold no network here" looks like: past it, the point is
# outside this network's coverage rather than merely set back from the road,
# and the haversine fallback is the honest answer.
MAX_SNAP_M = 2000.0

# Dijkstra returns a full row per source: one float64 per node of the graph.
# Paris' walking network has 317k nodes, so asking for 382 sources at once
# would allocate ~1GB for a result we immediately slice 382 columns out of.
# Sources are solved in chunks and sliced as we go.
_SOURCE_CHUNK = 32

_CACHE = {}


def point_key(lat: float, lon: float) -> str:
    """The identity of a catalogue point, to 6 decimal places (~11cm). Both
    the matrix builder and the lookup go through this, because a matrix keyed
    one way and read another is a bug that shows up as "real routing silently
    did nothing" rather than as a crash."""
    return f"{float(lat):.6f},{float(lon):.6f}"


def _equirect_metres(lons, lats, lat0_rad: float):
    """Lon/lat to metres for nearest-node search. A KD-tree needs a plane, and
    over a single city an equirectangular projection about the city's own
    latitude is accurate to well under the snapping tolerance -- these are
    30km-wide boxes, not continents."""
    x = np.asarray(lons, dtype=np.float64) * math.cos(lat0_rad)
    y = np.asarray(lats, dtype=np.float64)
    return np.column_stack([x, y]) * 111_320.0


def available_cities(profile: str = "foot") -> list[str]:
    if not DATA_DIR.is_dir():
        return []
    return sorted(p.name.split("_")[0] for p in DATA_DIR.glob(f"*_{profile}.npz"))


def load_city_network(city: str, profile: str):
    """Returns the cached handle for one city/profile, or None if we hold no
    network for it. `np.load` on a compressed .npz reads members on demand, so
    the matrix path never pays to decompress the street graph."""
    key = (city, profile)
    if key not in _CACHE:
        path = DATA_DIR / f"{city}_{profile}.npz"
        if not path.is_file():
            _CACHE[key] = None
        else:
            data = np.load(path, allow_pickle=False)
            _CACHE[key] = {
                "path": path, "data": data,
                "bbox": tuple(float(v) for v in data["bbox"]),
                "index": {k: i for i, k in enumerate(data["keys"].tolist())},
                "matrix": None,
                "minutes": None,
                "tree": None,
                "graph": None,
            }
    return _CACHE[key]


def _city_for_points(points: list[dict], profile: str):
    """The city whose network covers every point. Points carry no city id by
    the time they reach here -- they are plain {"lat","lon"} dicts -- and a
    trip never mixes cities, so geography answers this. A point outside every
    box we hold means the whole matrix is unanswerable: mixing street
    distances for some legs with haversine for others would make a day's
    total meaningless."""
    for city in available_cities(profile):
        net = load_city_network(city, profile)
        if net is None:
            continue
        west, south, east, north = net["bbox"]
        if all(west <= p["lon"] <= east and south <= p["lat"] <= north for p in points):
            return net
    return None


def _cached(net, slot: str, member: str):
    """One decompression per member per process. An .npz member is
    decompressed on every access, and these are read on every trip."""
    if net[slot] is None:
        if member not in net["data"]:
            return None
        net[slot] = np.asarray(net["data"][member], dtype=np.float64)
    return net[slot]


def _tree_for(net):
    if net["tree"] is None:
        xy = net["data"]["node_xy"]
        lat0 = math.radians(float(np.mean(xy[:, 1])))
        net["tree"] = (cKDTree(_equirect_metres(xy[:, 0], xy[:, 1], lat0)), lat0)
    return net["tree"]


def _csr_for(net):
    if net["graph"] is None:
        data = net["data"]
        uv, w = data["edge_uv"], data["edge_m"]
        n = len(data["node_xy"])
        net["graph"] = coo_matrix((w, (uv[:, 0], uv[:, 1])), shape=(n, n)).tocsr()
    return net["graph"]


def arrays_net(node_xy, edge_uv, edge_m) -> dict:
    """An ad-hoc network handle over flat arrays, for callers that hold a
    graph but no committed file yet -- `pipeline/build_street_network.py` is
    the only one."""
    return {"data": {"node_xy": node_xy, "edge_uv": edge_uv, "edge_m": edge_m},
            "matrix": None, "minutes": None, "tree": None, "graph": None}


def snap(net, lats, lons, max_metres=MAX_SNAP_M):
    """(nearest node index, offset in metres) per point, or None if any point
    lands further than `max_metres` from the network.

    `max_metres=None` snaps every point and lets the caller decide what to do
    with the far ones. A distance matrix cannot use that -- one bad point
    makes the whole matrix dishonest, so it takes the None -- but a caller
    snapping thousands of transit stops wants to drop the handful that sit
    inside a station concourse rather than lose the city."""
    tree, lat0 = _tree_for(net)
    offsets, nodes = tree.query(_equirect_metres(lons, lats, lat0))
    if max_metres is not None and len(offsets) and float(np.max(offsets)) > max_metres:
        return None
    return nodes, offsets


def matrix_over(net, lats, lons, directed: bool):
    """Shortest street-path distances in km between the given points. Both the
    committed matrix (built offline) and any run-time solve come through here,
    so the two cannot drift apart unnoticed."""
    snapped = snap(net, lats, lons)
    if snapped is None:
        return None
    nodes, offsets = snapped
    csr = _csr_for(net)
    out = np.empty((len(nodes), len(nodes)), dtype=np.float64)
    for start in range(0, len(nodes), _SOURCE_CHUNK):
        block = nodes[start:start + _SOURCE_CHUNK]
        out[start:start + len(block)] = dijkstra(csr, directed=directed, indices=block)[:, nodes]
    out /= 1000.0
    # The set-back from each endpoint to its road, at both ends of the leg.
    off_km = offsets / 1000.0
    out += off_km[:, None] + off_km[None, :]
    np.fill_diagonal(out, 0.0)
    # The graph is pruned to one connected component, so this cannot happen by
    # construction -- but an unreachable pair would otherwise reach the solver
    # as `inf` and quietly poison a whole day's arithmetic.
    if not np.isfinite(out).all():
        return None
    return out


def fetch_distance_duration_matrix(points: list[dict], profile: str = "foot"):
    """points: list of {"lat": .., "lon": ..}, in the exact order callers will
    index into the result. `profile` is "foot" or "car" (travel_modes.py), so a
    driving itinerary is priced on the road network rather than on footpaths.
    Returns (distance_km_matrix, duration_min_matrix) as same-order nested
    lists, or None if we hold no street network covering these points --
    callers must fall back to the haversine heuristic in that case, never
    raise.

    A `transit` file carries its own minutes -- a timetable is not a distance
    divided by a speed -- and those are returned as they were solved. For the
    street profiles, durations are the network distance at the travel mode's
    own calibrated speed, not a free-flow time read off OSM's `maxspeed` tags. That is
    deliberate: `driving`'s 25km/h is door-to-door urban driving in a historic
    centre, measured to be right for the kind of leg these itineraries
    contain, while OSM's posted limits describe an empty road. This issue is
    about replacing a straight line with the real street network; changing
    what a kilometre costs at the same time would hide one change inside the
    other."""
    if len(points) < 2:
        return None
    net = _city_for_points(points, profile)
    if net is None:
        return None

    keys = [point_key(p["lat"], p["lon"]) for p in points]
    rows = [net["index"].get(k) for k in keys]
    if all(r is not None for r in rows):
        take = np.asarray(rows)
        km = _cached(net, "matrix", "matrix_km")[np.ix_(take, take)]
        stored = _cached(net, "minutes", "matrix_min")
        minutes = None if stored is None else stored[np.ix_(take, take)]
    elif profile in MATRIX_ONLY:
        return None
    else:
        km = matrix_over(
            net, [p["lat"] for p in points], [p["lon"] for p in points],
            DIRECTED[profile])
        if km is None:
            return None
        minutes = None

    if minutes is None:
        minutes = km / mode_for_network_profile(profile).speed_kmh * 60.0
    return km.tolist(), minutes.tolist()
