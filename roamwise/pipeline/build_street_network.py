"""Build the committed street networks and distance matrices (issue #32).

RoamWise used to get real street distances from a public OSRM demo server.
That worked, and it made real routing permanently opt-in: the server has no
uptime guarantee and rate-limits bursts, so `use_real_routing=True` could
silently mean "haversine after all" on any given afternoon, and nobody wanted
that as a default. Everything else in this pipeline already answers that the
same way -- fetch once, commit the result, run offline -- and this script is
that answer applied to routing.

Two artefacts per city and profile, in one `.npz` (`np.load` reads members on
demand, so reading the matrix never decompresses the graph):

  * **the matrix** -- shortest street-path distances between every point the
    router can be handed: the city's POIs, its transport hubs, and its centre,
    which is what RouterAgent uses as each day's `start_hub`. This is the
    whole point: at run time a distance is a dictionary lookup, so the real
    street network costs nothing per trip.
  * **the graph** -- the pruned street network itself, so a point that is
    *not* in the catalogue (a pinned coordinate, a POI added after the last
    build) still gets a real distance instead of a straight line.

Three things are done to the network osmnx returns, and each one earns its
place. Paris' raw walking graph is 395MB of GraphML for 317k nodes:

  * **Parallel edges collapse to the shortest.** OSM models a dual
    carriageway as two ways; for "how far is it" the shorter one is the
    answer, and a MultiDiGraph is 2x the edges for no extra information.
  * **Only the largest connected component survives.** An unreachable pair
    comes back `inf` and poisons a day's arithmetic; pruning makes that
    impossible by construction rather than by a guard that might be missing.
  * **Only x, y and length are kept.** Names, OSM ids and edge geometry are
    what makes GraphML enormous, and nothing downstream reads them. Stored as
    flat arrays, Paris on foot is 3.9MB.

Walking networks are stored undirected -- pedestrians ignore one-way
restrictions, and it halves the edge count. Driving networks are not: a
one-way street genuinely costs a different distance in each direction.

    python build_street_network.py PAR BER --write
"""
import argparse
import math
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

from common import CACHE, CITIES, DATA, haversine_km

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from roamwise.optimization.street_network import (  # noqa: E402
    DIRECTED, MAX_SNAP_M, NETWORK_TYPE, arrays_net, matrix_over, point_key, snap)

OUT_DIR = DATA / "street_network"
# How far past the outermost catalogue point the network reaches. Two reasons,
# and the second is what set the number:
#
#   * A route between two points near the same edge of the box can legitimately
#     leave it -- the shortest path around a park is not always inside the box
#     drawn around its endpoints -- and `truncate_by_edge` keeps whole ways
#     that cross the boundary rather than cutting them mid-street.
#   * A point at a *corner* of the box needs enough network around it to reach
#     the rest of the city, or component pruning throws its streets away. At
#     2.0km Berlin Brandenburg's footways were a 788-node island: the corridor
#     linking them to Berlin ran outside the box, so the airport's own paths
#     were not the largest component and were dropped, leaving the nearest
#     node 1,865m away in a field. The airport is 18km out and sits in the
#     box's south-east corner, which is exactly where the margin has to do
#     this work (#92). It reconnects at 3.0km; 4.0 is that plus headroom, for
#     about 20% more nodes.
MARGIN_KM = 4.0

# How much further from the network a catalogue point may be pushed by
# component pruning before the build is wrong rather than merely lossy.
#
# This is the check that #92 went missing for want of. Dropping a component is
# normal and mostly right -- a car park loop with no way out is noise -- but
# when the component that goes is the one a catalogue point stands on, the
# point does not fail, it silently starts answering for somewhere else. BER's
# transit access was measured from a village 1.2km away and reported 240
# minutes to the centre against a true 45, and nothing in the build said so.
MAX_COMPONENT_LOSS_M = 100.0

ox.settings.use_cache = True
ox.settings.cache_folder = str(CACHE / "osmnx")


def catalogue_points(code: str) -> pd.DataFrame:
    """Every point the router can be asked to route over, in a stable order.

    The city centre is in here because RouterAgent anchors every day at it
    (`start_hub`), not because it is a place anyone visits -- leave it out and
    the first leg of every single day misses the matrix, which takes the whole
    trip onto the slow path for one point."""
    pois = pd.read_csv(DATA / "poi.csv")
    hubs = pd.read_csv(DATA / "transport.csv")
    dests = pd.read_csv(DATA / "destinations.csv")
    frames = [
        pois.loc[pois.destination_id == code, ["name", "lat", "lon"]].assign(kind="poi"),
        hubs.loc[hubs.destination_id == code, ["name", "lat", "lon"]].assign(kind="hub"),
        dests.loc[dests.destination_id == code, ["city", "lat", "lon"]]
             .rename(columns={"city": "name"}).assign(kind="centre"),
    ]
    out = pd.concat(frames, ignore_index=True)
    out["key"] = [point_key(a, b) for a, b in zip(out.lat, out.lon)]
    # Two POIs sharing a coordinate to 6dp would collide in the lookup; the
    # matrix row is identical for both, so keeping the first is lossless.
    return out.drop_duplicates("key").reset_index(drop=True)


def bbox_for(points: pd.DataFrame):
    """(west, south, east, north) around the catalogue, with MARGIN_KM added."""
    dlat = MARGIN_KM * 1000 / 111_320
    dlon = MARGIN_KM * 1000 / (111_320 * math.cos(math.radians(points.lat.mean())))
    return (points.lon.min() - dlon, points.lat.min() - dlat,
            points.lon.max() + dlon, points.lat.max() + dlat)


def prune(G, undirected: bool):
    """osmnx MultiDiGraph -> (node_xy, edge_uv, edge_m) over one component."""
    H = nx.Graph() if undirected else nx.DiGraph()
    for u, v, d in G.edges(data=True):
        length = float(d["length"])
        if H.has_edge(u, v):
            H[u][v]["length"] = min(H[u][v]["length"], length)
        else:
            H.add_edge(u, v, length=length)
    components = nx.connected_components(H) if undirected else nx.strongly_connected_components(H)
    keep = max(components, key=len)
    H = H.subgraph(keep)

    nodes = list(H.nodes)
    order = {n: i for i, n in enumerate(nodes)}
    node_xy = np.array([[G.nodes[n]["x"], G.nodes[n]["y"]] for n in nodes], dtype=np.float32)
    edge_uv = np.array([[order[u], order[v]] for u, v in H.edges], dtype=np.uint32)
    edge_m = np.array([d["length"] for _, _, d in H.edges(data=True)], dtype=np.float32)
    return node_xy, edge_uv, edge_m


def build(code: str, profile: str, write: bool):
    points = catalogue_points(code)
    box = bbox_for(points)
    print(f"\n{code}/{profile}: {len(points)} catalogue points, "
          f"bbox {box[0]:.4f},{box[1]:.4f},{box[2]:.4f},{box[3]:.4f}")

    # `retain_all=True` because the component decision belongs to `prune`, which
    # documents it and which the loss check below can see. osmnx's default
    # quietly drops everything outside the largest weakly connected component
    # before we ever look, and that is how BER's footways disappeared without
    # a line of output (#92).
    G = ox.graph_from_bbox(bbox=box, network_type=NETWORK_TYPE[profile],
                           simplify=True, retain_all=True, truncate_by_edge=True)
    print(f"  downloaded {G.number_of_nodes():,} nodes / {G.number_of_edges():,} edges")

    # Where each point sits relative to the *whole* download, to compare
    # against where it sits once one component is kept.
    full_xy = np.array([[d["x"], d["y"]] for _, d in G.nodes(data=True)], dtype=np.float32)
    empty_uv = np.empty((0, 2), dtype=np.uint32)
    empty_m = np.empty(0, dtype=np.float32)
    lats, lons = points.lat.to_numpy(), points.lon.to_numpy()
    _, full_offsets = snap(arrays_net(full_xy, empty_uv, empty_m), lats, lons,
                           max_metres=None)

    node_xy, edge_uv, edge_m = prune(G, undirected=not DIRECTED[profile])
    print(f"  pruned to  {len(node_xy):,} nodes / {len(edge_uv):,} edges")

    net = arrays_net(node_xy, edge_uv, edge_m)
    snapped = snap(net, lats, lons)
    if snapped is None:
        print(f"  !! a catalogue point is further than {MAX_SNAP_M:.0f}m from this "
              f"network -- widen the bbox or check the coordinate")
        return
    offsets = snapped[1]
    worst = points.name[int(np.argmax(offsets))]
    print(f"  snapped: median {np.median(offsets):.0f}m, mean {offsets.mean():.0f}m, "
          f"worst {offsets.max():.0f}m ({worst})")

    stranded = np.flatnonzero(offsets - full_offsets > MAX_COMPONENT_LOSS_M)
    if len(stranded):
        print(f"  !! {len(stranded)} point(s) lost their streets to component pruning "
              f"-- widen MARGIN_KM:")
        for i in stranded[np.argsort(-offsets[stranded])][:10]:
            print(f"       {points.name[int(i)][:44]:<46} "
                  f"{full_offsets[i]:>6.0f}m -> {offsets[i]:>6.0f}m")
        return

    km = matrix_over(net, lats, lons, DIRECTED[profile])
    if km is None:
        print("  !! a pair came back unreachable despite the component pruning")
        return
    straight = np.array([[haversine_km(a, b, c, d) for c, d in zip(points.lat, points.lon)]
                         for a, b in zip(points.lat, points.lon)])
    off_diagonal = straight > 0.05
    ratio = float(np.mean(km[off_diagonal] / straight[off_diagonal]))
    print(f"  matrix {km.shape[0]}x{km.shape[1]}, street/straight-line ratio {ratio:.3f}")

    if not write:
        print("  (dry run -- pass --write to save)")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{code}_{profile}.npz"
    np.savez_compressed(
        path, bbox=np.array(box, dtype=np.float64),
        keys=points.key.to_numpy().astype("U22"),
        matrix_km=km.astype(np.float32),
        node_xy=node_xy, edge_uv=edge_uv, edge_m=edge_m)
    print(f"  wrote {path.relative_to(DATA.parent)} ({path.stat().st_size / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cities", nargs="*", default=list(CITIES), help="city codes")
    ap.add_argument("--profiles", nargs="*", default=list(NETWORK_TYPE))
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    for code in (args.cities or list(CITIES)):
        for profile in args.profiles:
            build(code, profile, args.write)


if __name__ == "__main__":
    main()
