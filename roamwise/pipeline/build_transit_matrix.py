"""Build the committed transit travel-time matrix from GTFS (issue #32, stage 2).

Paris is the pilot. `transport.csv` and the catalogue give the places; the
IDFM feed gives the timetable; `optimization/raptor.py` turns one into travel
times over the other. Nothing here runs at request time -- what ships is a
matrix, the same way stage 1 ships street distances.

The feed is read straight out of its zip. Uncompressed it is 1.3GB, of which
`stop_times.txt` alone is 992MB and 11.4M rows, and none of that needs to
touch disk: one service day inside the city's bounding box is 2.0M rows and
99k trips, which is a few hundred MB of numpy.

**Which day.** A timetable is not a number, it is a schedule, and a matrix is.
This builds one representative weekday (`--date`, default the first Wednesday
the feed covers): school-term Wednesday service is the densest ordinary day,
and a Sunday matrix would understate every journey in the catalogue. What the
matrix cannot express is *when* -- see `--departures` below.

**Which departure.** Travel time by transit depends on when you leave: a
20-minute headway means the honest answer varies by 20 minutes depending on
your luck. RAPTOR is cheap enough to just ask repeatedly, so every pair is
solved from several departures across the day and the **median** is kept.
That is the time a traveller should plan around, rather than a best case that
assumes the platform clock was kind.

**Walking is part of the answer.** Each pair is also costed as a direct walk
on the committed street network (stage 1) and the faster of the two is kept,
so the matrix never advises three changes to cross a square.

Access, egress and transfer footpaths are real street distances from that same
network -- wherever the network can say where something *is*. It usually can:
14,130 of Paris' 14,147 stops and 6,386 of Berlin's 6,412 sit within 150m of a footway. Where it
cannot, the answer is not a worse walk but a walk from somewhere else, and
that is a different kind of wrong. Berlin Brandenburg's coordinate is the
middle of the airfield; its nearest footway node is 1,225m away in a field
outside Waßmannsdorf, and measuring access from there found two village bus
stops and routed the airport into the city in 240 minutes on rural buses,
while the airport's own platforms sat 785m away and were never considered.
So a place set far back from the network gets a straight line and a detour
factor instead, and platforms of one station are linked through the feed's own
`parent_station` -- the walk from an S-Bahn platform to the FEX is inside a
building no footway network describes.

    python build_transit_matrix.py PAR --write
"""
import argparse
import datetime
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import dijkstra

from common import CACHE, DATA

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from roamwise.optimization.raptor import (  # noqa: E402
    DEFAULT_CHANGE_SECONDS, DEFAULT_ROUNDS, INFINITY, TransitTable, earliest_arrivals)
from roamwise.optimization.street_network import (  # noqa: E402
    _csr_for, load_city_network, point_key, snap)
from build_street_network import catalogue_points  # noqa: E402

GTFS_URLS = {
    # Île-de-France Mobilités and VBB each publish one official feed for their
    # whole region, with no API key, which is what made these two the pilot
    # pair. Both are regional rather than municipal, so the city bounding box
    # does real work: Berlin's feed reaches Cottbus and Frankfurt (Oder).
    "PAR": "https://eu.ftp.opendatasoft.com/stif/GTFS/IDFM-gtfs.zip",
    "BER": "https://www.vbb.de/fileadmin/user_upload/VBB/Dokumente/API-Datensaetze/"
           "gtfs-mastscharf/GTFS.zip",
}
OUT_DIR = DATA / "street_network"
GTFS_DIR = CACHE / "gtfs"

# How far someone will walk to reach a stop, and between stops to change.
# 800m is about ten minutes and is the usual planning figure for "within
# walking distance of transit"; 400m keeps transfers to what a traveller
# would actually accept mid-journey rather than routing them across a
# district on foot to save a minute.
ACCESS_METRES = 800.0
TRANSFER_METRES = 400.0
WALK_SPEED_KMH = 4.5
# How close a place has to snap to the footway network before we believe the
# network knows where it is. Almost everything does -- the median catalogue
# point snaps under 25m in both cities -- but a place stored as the centroid
# of a large polygon does not, and for those the network answers a question
# about somewhere else. Berlin Brandenburg's coordinate is the middle of the
# airfield: its nearest footway node is 1,225m away in a field outside
# Waßmannsdorf, so network access found two village bus stops and routed the
# airport into the city in 240 minutes, on rural buses, while "Flughafen BER"
# sat 727m away in a straight line and was never considered.
#
# Past this, access is measured as a straight line with a detour factor
# instead. That is a worse model of walking and a far better model of the
# truth: the walk from the middle of an airport to its own station is inside
# the airport, and no footway network describes it.
SNAP_TRUST_METRES = 150.0
STRAIGHT_LINE_DETOUR = 1.3
# Walking between platforms of one station. GTFS models this with
# `parent_station`, and ignoring it is how an airport ends up unreachable:
# Flughafen BER's 21 platforms share a parent, they sit inside a terminal no
# footway network maps, and without the parent link arriving on one platform
# tells you nothing about the trains leaving from another. Paris tags every
# boarding stop with a parent; Berlin only 8% of them, but the ones that
# matter -- the stations -- are among them.
STATION_CHANGE_SECONDS = 120
# Departures sampled across an ordinary day. Hourly from 08:00 keeps the
# morning peak, the midday trough and the evening service all in the median.
DEPARTURE_HOURS = list(range(8, 21))
_SOURCE_CHUNK = 24


def gtfs_path(code: str) -> Path:
    return GTFS_DIR / f"{code}-gtfs.zip"


def service_ids_on(z: zipfile.ZipFile, day: datetime.date) -> set:
    """Services running on one date: the weekly pattern, plus the additions
    and minus the removals `calendar_dates.txt` records for that date. Feeds
    move a surprising amount of service through those exceptions -- this one
    carries 1,685 of them -- so reading only `calendar.txt` would quietly
    schedule trips that do not run."""
    key = day.strftime("%Y%m%d")
    weekday = day.strftime("%A").lower()
    with z.open("calendar.txt") as f:
        calendar = pd.read_csv(f, dtype=str)
    active = set(calendar.loc[(calendar[weekday] == "1")
                              & (calendar.start_date <= key)
                              & (calendar.end_date >= key), "service_id"])
    with z.open("calendar_dates.txt") as f:
        exceptions = pd.read_csv(f, dtype=str)
    today = exceptions[exceptions.date == key]
    active |= set(today.loc[today.exception_type == "1", "service_id"])
    active -= set(today.loc[today.exception_type == "2", "service_id"])
    return active


def busiest_weekday(z: zipfile.ZipFile, weekday: int = 2) -> datetime.date:
    """The Wednesday (weekday=2) inside the feed's span that runs the most
    trips. Feeds are republished constantly, so a literal date would rot -- but
    "the first Wednesday" is worse than it sounds: this feed's span opens three
    days before its own publication date and those days carry one service and
    5,027 trips against a normal Wednesday's 144,072. Picking the first one
    silently built a matrix of a city with almost no public transport. Asking
    which day actually runs service cannot make that mistake."""
    with z.open("calendar.txt") as f:
        calendar = pd.read_csv(f, dtype=str)
    with z.open("trips.txt") as f:
        per_service = pd.read_csv(f, dtype=str, usecols=["service_id", "trip_id"]) \
            .groupby("service_id").size()
    start = datetime.datetime.strptime(calendar.start_date.min(), "%Y%m%d").date()
    end = datetime.datetime.strptime(calendar.end_date.max(), "%Y%m%d").date()
    day = start + datetime.timedelta(days=(weekday - start.weekday()) % 7)
    best, best_trips = None, -1
    while day <= end:
        trips = int(per_service.reindex(list(service_ids_on(z, day))).fillna(0).sum())
        if trips > best_trips:
            best, best_trips = day, trips
        day += datetime.timedelta(days=7)
    if best is None:
        raise SystemExit(f"feed covers {start}..{end}, no weekday {weekday} inside it")
    print(f"service day {best} ({best:%A}), the fullest in {start}..{end}: {best_trips:,} trips")
    return best


def seconds_of_day(value: str) -> int:
    hours, minutes, secs = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + int(secs)


def read_timetable(code: str, day: datetime.date, bbox):
    """(stops dataframe, stop_times dataframe) for one service day inside the
    bounding box, read without unpacking the archive."""
    z = zipfile.ZipFile(gtfs_path(code))
    with z.open("stops.txt") as f:
        stops = pd.read_csv(f, dtype=str)
    stops["lat"] = pd.to_numeric(stops.stop_lat, errors="coerce")
    stops["lon"] = pd.to_numeric(stops.stop_lon, errors="coerce")
    west, south, east, north = bbox
    stops = stops[(stops.lon.between(west, east)) & (stops.lat.between(south, north))
                  & (stops.location_type == "0")].reset_index(drop=True)

    services = service_ids_on(z, day)
    with z.open("trips.txt") as f:
        trips = pd.read_csv(f, dtype=str, usecols=["route_id", "service_id", "trip_id"])
    running = set(trips.loc[trips.service_id.isin(services), "trip_id"])

    kept = []
    with z.open("stop_times.txt") as f:
        for chunk in pd.read_csv(f, dtype=str, chunksize=2_000_000,
                                 usecols=["trip_id", "arrival_time", "departure_time",
                                          "stop_id", "stop_sequence"]):
            kept.append(chunk[chunk.trip_id.isin(running)])
    times = pd.concat(kept, ignore_index=True)
    times = times[times.stop_id.isin(set(stops.stop_id))]
    times["dep"] = times.departure_time.map(seconds_of_day)
    times["arr"] = times.arrival_time.map(seconds_of_day)
    times["seq"] = times.stop_sequence.astype(int)
    # A trip that leaves the box and comes back keeps its real arrival times
    # at the stops inside it, so dropping the outside ones is safe -- the
    # vehicle still takes as long as it takes to come back round.
    return stops, times.sort_values(["trip_id", "seq"], kind="stable")


def build_patterns(stops: pd.DataFrame, times: pd.DataFrame):
    """Trips grouped by the exact stop sequence they call at, which is what
    RAPTOR scans. GTFS `route_id` will not do: trips on one route call at
    different stops (short workings, branches), and a binary search for "the
    next trip from this stop" is only meaningful within one sequence."""
    index = {stop_id: i for i, stop_id in enumerate(stops.stop_id)}
    by_trip = defaultdict(list)
    for trip_id, stop_id, dep, arr in zip(times.trip_id, times.stop_id, times.dep, times.arr):
        by_trip[trip_id].append((index[stop_id], dep, arr))

    grouped = defaultdict(list)
    for calls in by_trip.values():
        if len(calls) < 2:
            continue  # a single call cannot carry anyone anywhere
        grouped[tuple(c[0] for c in calls)].append(calls)

    patterns = []
    for sequence, trips in grouped.items():
        departures = np.array([[c[1] for c in trip] for trip in trips], dtype=np.int32)
        arrivals = np.array([[c[2] for c in trip] for trip in trips], dtype=np.int32)
        patterns.append((np.array(sequence, dtype=np.int32), departures, arrivals))
    return patterns, index


def haversine_metres(lat, lon, lats, lons):
    """One point against many, in metres."""
    radius = 6_371_000.0
    phi1, phi2 = np.radians(lat), np.radians(lats)
    a = (np.sin((phi2 - phi1) / 2) ** 2
         + np.cos(phi1) * np.cos(phi2) * np.sin(np.radians(lons - lon) / 2) ** 2)
    return 2 * radius * np.arcsin(np.sqrt(a))


def walk_seconds(metres: np.ndarray) -> np.ndarray:
    return (metres / (WALK_SPEED_KMH * 1000 / 3600)).astype(np.int32)


def bounded_walks(net, source_nodes, target_nodes, limit_m: float):
    """Walking distance in metres from each source node to every target node,
    over the committed street network, giving up past `limit_m`.

    The cutoff is doing real work: without it this is a 14,146 x 14,146
    all-pairs solve over a 350k-node graph. With it, each search stops after a
    few hundred metres and the whole thing is a matter of seconds."""
    csr = _csr_for(net)
    out = np.full((len(source_nodes), len(target_nodes)), np.inf)
    for start in range(0, len(source_nodes), _SOURCE_CHUNK):
        block = source_nodes[start:start + _SOURCE_CHUNK]
        distances = dijkstra(csr, directed=False, indices=block, limit=limit_m)
        out[start:start + len(block)] = distances[:, target_nodes]
    return out


def transfer_edges(net, stop_nodes, trusted, limit_m: float):
    """Footpaths between stops, as (from, to, seconds), over the street
    network -- only between stops the network can actually place. A stop
    sitting inside a terminal building snaps to whatever road passes outside,
    and a footpath measured from there is fiction; those stops connect through
    their station instead (see `station_edges`).

    Returned as edges rather than a matrix on purpose: 14,146 stops is a
    200-million-cell matrix that is almost entirely infinity, and RAPTOR only
    ever wants the neighbours."""
    csr = _csr_for(net)
    usable = np.flatnonzero(trusted)
    nodes = stop_nodes[usable]
    sources, targets, metres = [], [], []
    for start in range(0, len(nodes), _SOURCE_CHUNK):
        block = nodes[start:start + _SOURCE_CHUNK]
        distances = dijkstra(csr, directed=False, indices=block, limit=limit_m)[:, nodes]
        rows, columns = np.nonzero(np.isfinite(distances) & (distances > 0))
        sources.append(usable[rows + start])
        targets.append(usable[columns])
        metres.append(distances[rows, columns])
    return (np.concatenate(sources), np.concatenate(targets),
            walk_seconds(np.concatenate(metres)))


def station_edges(parents: np.ndarray):
    """Platform-to-platform links inside one station, as (from, to, seconds).

    This is the transfer a traveller makes without going outside: off the S9,
    up the escalator, onto the FEX. The feed says which platforms belong
    together and nothing else can -- they are metres apart in a building the
    street network does not describe."""
    sources, targets = [], []
    order = np.argsort(parents, kind="stable")
    grouped = parents[order]
    starts = np.flatnonzero(np.r_[True, grouped[1:] != grouped[:-1]])
    for start, end in zip(starts, np.r_[starts[1:], len(grouped)]):
        members = order[start:end]
        if len(members) < 2 or not grouped[start]:
            continue
        left = np.repeat(members, len(members))
        right = np.tile(members, len(members))
        keep = left != right
        sources.append(left[keep])
        targets.append(right[keep])
    if not sources:
        return (np.empty(0, dtype=int), np.empty(0, dtype=int), np.empty(0, dtype=np.int32))
    sources = np.concatenate(sources)
    return (sources, np.concatenate(targets),
            np.full(len(sources), STATION_CHANGE_SECONDS, dtype=np.int32))


def access_edges(net, point_nodes, point_offsets, points, stop_nodes, trusted_stops,
                 stop_points, limit_m: float):
    """(stops, seconds) each point can walk to, as a rectangular array padded
    with INFINITY -- the shape `earliest_arrivals` takes.

    Two ways of measuring, chosen per point by how far it snapped (see
    SNAP_TRUST_METRES). Points the network can place get real walking
    distances, plus the short walk from the place to its own footway. Points
    it cannot get a straight line and a detour factor, because a network path
    from the wrong starting node is not a worse estimate, it is an answer to a
    different question."""
    csr = _csr_for(net)
    trusted_points = point_offsets <= SNAP_TRUST_METRES
    per_point = []
    stop_lat, stop_lon = stop_points[:, 0], stop_points[:, 1]
    network_stops = np.flatnonzero(trusted_stops)

    for point in range(len(point_nodes)):
        # `reach` decides which stops are close enough to walk to; `cost` is
        # how long that walk takes. They are not the same number for a stop
        # measured as a straight line -- the detour factor is a model of the
        # walk, not of how far someone is willing to go -- and multiplying the
        # radius by it as well is how an airport's own platform, 785m away,
        # got pushed to 1,020m and out of reach.
        reach = haversine_metres(points[point, 0], points[point, 1], stop_lat, stop_lon)
        cost = reach * STRAIGHT_LINE_DETOUR
        if trusted_points[point] and len(network_stops):
            # Both ends placeable: use the streets, and charge the walk from
            # the place to its own footway on top.
            row = dijkstra(csr, directed=False, indices=[point_nodes[point]],
                           limit=limit_m)[0][stop_nodes[network_stops]]
            walked = row + point_offsets[point]
            found = np.isfinite(walked)
            placed = network_stops[found]
            reach[placed] = walked[found]
            cost[placed] = walked[found]
        near = np.flatnonzero(reach <= limit_m)
        per_point.append((near, walk_seconds(cost[near])))
    width = max((len(stops) for stops, _ in per_point), default=1) or 1
    stops_out = np.zeros((len(per_point), width), dtype=np.int32)
    seconds_out = np.full((len(per_point), width), INFINITY, dtype=np.int32)
    for i, (stops, seconds) in enumerate(per_point):
        stops_out[i, :len(stops)] = stops
        seconds_out[i, :len(seconds)] = seconds
    return stops_out, seconds_out


def egress_minutes(best: np.ndarray, access_stops, access_seconds, departures):
    """Journey time in minutes from every origin to every destination.

    Egress reuses the access footpaths: the walk from a stop to a place is the
    same walk as the one to it. `best` holds arrival at every stop, so a
    destination's journey time is the best of "arrive at a nearby stop, then
    walk"."""
    n_points = access_stops.shape[0]
    out = np.full((best.shape[0], n_points), np.inf)
    for destination in range(n_points):
        reachable = access_seconds[destination] < INFINITY
        if not reachable.any():
            continue
        stops = access_stops[destination][reachable]
        walk = access_seconds[destination][reachable]
        arrival = best[:, stops] + walk
        out[:, destination] = np.where(arrival.min(axis=1) >= INFINITY, np.inf,
                                       arrival.min(axis=1) - departures)
    return out / 60.0


def build(code: str, day: datetime.date, hours, write: bool):
    walk_net = load_city_network(code, "foot")
    if walk_net is None:
        raise SystemExit(f"no committed walking network for {code} -- run "
                         f"build_street_network.py {code} --write first")
    points = catalogue_points(code)
    bbox = walk_net["bbox"]
    print(f"{code}: {len(points)} catalogue points, service day {day} ({day:%A})")

    stops, times = read_timetable(code, day, bbox)
    print(f"  timetable: {len(stops):,} stops, {len(times):,} calls, "
          f"{times.trip_id.nunique():,} trips")

    patterns, _ = build_patterns(stops, times)
    print(f"  patterns:  {len(patterns):,} stop sequences")

    # Every stop stays. Most snap onto the footway network within a few
    # metres, and those get real walking transfers; the ones that do not are
    # inside station buildings the network does not model, and dropping them
    # was how an airport lost its own platforms and became unreachable.
    stop_nodes, stop_offsets = snap(walk_net, stops.lat.to_numpy(), stops.lon.to_numpy(),
                                    max_metres=None)
    trusted_stops = stop_offsets <= SNAP_TRUST_METRES
    print(f"  placeable: {int(trusted_stops.sum()):,} of {len(stops):,} stops sit within "
          f"{SNAP_TRUST_METRES:.0f}m of a footway")

    walk_from, walk_to, walk_secs = transfer_edges(walk_net, stop_nodes, trusted_stops,
                                                   TRANSFER_METRES)
    parents = stops.parent_station.fillna("").to_numpy()
    station_from, station_to, station_secs = station_edges(parents)
    print(f"  transfers: {len(walk_from):,} footpaths under {TRANSFER_METRES:.0f}m, "
          f"{len(station_from):,} platform links inside stations")
    transfers = list(zip(np.r_[walk_from, station_from].tolist(),
                         np.r_[walk_to, station_to].tolist(),
                         np.r_[walk_secs, station_secs].tolist()))

    table = TransitTable.build(len(stops), patterns, transfers)
    split = table.n_patterns - len(patterns)
    print(f"  table:     {table.n_patterns:,} scannable patterns "
          f"({split} split off as overtaking)")

    point_nodes, point_offsets = snap(walk_net, points.lat.to_numpy(),
                                      points.lon.to_numpy(), max_metres=None)
    point_coords = np.column_stack([points.lat.to_numpy(), points.lon.to_numpy()])
    stop_coords = np.column_stack([stops.lat.to_numpy(), stops.lon.to_numpy()])
    access_stops, access_seconds = access_edges(walk_net, point_nodes, point_offsets,
                                                point_coords, stop_nodes, trusted_stops,
                                                stop_coords, ACCESS_METRES)
    reachable = (access_seconds < INFINITY).sum(axis=1)
    off_network = int((point_offsets > SNAP_TRUST_METRES).sum())
    print(f"  access:    median {int(np.median(reachable))} stops within "
          f"{ACCESS_METRES:.0f}m, {int((reachable == 0).sum())} places with none, "
          f"{off_network} measured as straight lines (set back from the network)")

    samples = []
    for hour in hours:
        departures = np.full(len(points), hour * 3600, dtype=np.int32)
        best = earliest_arrivals(table, access_stops, access_seconds, departures,
                                 rounds=DEFAULT_ROUNDS,
                                 change_seconds=DEFAULT_CHANGE_SECONDS)
        samples.append(egress_minutes(best, access_stops, access_seconds, departures))
        print(f"  {hour:02d}:00 solved", flush=True)
    transit = np.median(np.stack(samples), axis=0)

    # Walking is the floor. A pair the timetable cannot beat -- two cafés on
    # one street, or anywhere the network simply does not reach -- keeps the
    # walk, so the matrix never proposes a bus ride across a square.
    walk_km = np.asarray(walk_net["data"]["matrix_km"], dtype=np.float64)
    walk_minutes = walk_km / WALK_SPEED_KMH * 60.0
    take_transit = transit < walk_minutes
    minutes = np.where(take_transit, transit, walk_minutes)
    np.fill_diagonal(minutes, 0.0)

    drive_net = load_city_network(code, "car")
    drive_km = (np.asarray(drive_net["data"]["matrix_km"], dtype=np.float64)
                if drive_net is not None else walk_km)
    # Ground actually covered: the road distance where a vehicle carries you,
    # the footway distance where you walk it.
    km = np.where(take_transit, drive_km, walk_km)
    np.fill_diagonal(km, 0.0)

    share = take_transit.sum() / take_transit.size
    faster = walk_minutes[take_transit] / minutes[take_transit]
    print(f"\n  transit beats walking on {share * 100:.0f}% of pairs, "
          f"by a median factor of {np.median(faster):.2f}x")
    print(f"  mean journey {minutes[~np.eye(len(points), dtype=bool)].mean():.0f} min "
          f"(walking everywhere: {walk_minutes[~np.eye(len(points), dtype=bool)].mean():.0f} min)")

    if not write:
        print("  (dry run -- pass --write to save)")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{code}_transit.npz"
    np.savez_compressed(
        path, bbox=np.array(bbox, dtype=np.float64),
        keys=points.key.to_numpy().astype("U22"),
        matrix_km=km.astype(np.float32),
        matrix_min=minutes.astype(np.float32),
        service_date=np.array(day.isoformat()),
        departure_hours=np.array(hours, dtype=np.int32))
    print(f"  wrote {path.relative_to(DATA.parent)} ({path.stat().st_size / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cities", nargs="*", default=list(GTFS_URLS))
    ap.add_argument("--date", help="service day, YYYY-MM-DD (default: the feed's first Wednesday)")
    ap.add_argument("--departures", nargs="*", type=int, default=DEPARTURE_HOURS)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    for code in (args.cities or list(GTFS_URLS)):
        if not gtfs_path(code).is_file():
            raise SystemExit(f"missing {gtfs_path(code)} -- download it from "
                             f"{GTFS_URLS.get(code, '(no feed registered)')}")
        with zipfile.ZipFile(gtfs_path(code)) as z:
            day = (datetime.date.fromisoformat(args.date) if args.date
                   else busiest_weekday(z))
        build(code, day, args.departures, args.write)


if __name__ == "__main__":
    main()
