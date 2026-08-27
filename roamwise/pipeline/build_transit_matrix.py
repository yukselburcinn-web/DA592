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
so the matrix never advises three changes to cross a square. Access, egress
and transfer footpaths are real street distances from the same network rather
than straight lines -- having built that in stage 1, using it here costs
nothing and a crow-flies transfer inside Châtelet would be a fiction.

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
    # Île-de-France Mobilités publishes one official feed for the whole
    # region, with no API key. Berlin's VBB equivalent is open too, which is
    # what makes these two cities the pilot pair.
    "PAR": "https://eu.ftp.opendatasoft.com/stif/GTFS/IDFM-gtfs.zip",
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


def transfer_edges(net, stop_nodes, limit_m: float):
    """Footpaths between stops, as (from, to, metres).

    Returned as edges rather than a matrix on purpose: 14,146 stops is a
    200-million-cell matrix that is almost entirely infinity, and RAPTOR only
    ever wants the neighbours."""
    csr = _csr_for(net)
    sources, targets, metres = [], [], []
    for start in range(0, len(stop_nodes), _SOURCE_CHUNK):
        block = stop_nodes[start:start + _SOURCE_CHUNK]
        distances = dijkstra(csr, directed=False, indices=block, limit=limit_m)[:, stop_nodes]
        rows, columns = np.nonzero(np.isfinite(distances) & (distances > 0))
        sources.append(rows + start)
        targets.append(columns)
        metres.append(distances[rows, columns])
    return (np.concatenate(sources), np.concatenate(targets), np.concatenate(metres))


def access_edges(net, point_nodes, stop_nodes, limit_m: float):
    """(stops, seconds) each point can walk to, as a rectangular array padded
    with INFINITY -- the shape `earliest_arrivals` takes."""
    csr = _csr_for(net)
    per_point = []
    for start in range(0, len(point_nodes), _SOURCE_CHUNK):
        block = point_nodes[start:start + _SOURCE_CHUNK]
        distances = dijkstra(csr, directed=False, indices=block, limit=limit_m)[:, stop_nodes]
        for row in distances:
            near = np.flatnonzero(np.isfinite(row))
            per_point.append((near, walk_seconds(row[near])))
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

    # Stops sit at street addresses, so most snap onto the footway network
    # within a few metres. The ones that do not are inside station buildings
    # the network does not model; they are dropped rather than attached to
    # whatever road happens to be nearest.
    stop_nodes, stop_offsets = snap(walk_net, stops.lat.to_numpy(), stops.lon.to_numpy(),
                                    max_metres=None)
    usable = stop_offsets <= ACCESS_METRES
    if not usable.all():
        print(f"  dropped {int((~usable).sum())} stops further than "
              f"{ACCESS_METRES:.0f}m from any footway")
    keep = np.flatnonzero(usable)
    renumber = np.full(len(stops), -1, dtype=np.int64)
    renumber[keep] = np.arange(len(keep))
    stop_nodes = stop_nodes[keep]
    patterns = [(renumber[seq], dep, arr) for seq, dep, arr in patterns
                if (renumber[seq] >= 0).all()]
    print(f"  usable:    {len(keep):,} stops, {len(patterns):,} patterns")

    sources, targets, metres = transfer_edges(walk_net, stop_nodes, TRANSFER_METRES)
    print(f"  transfers: {len(sources):,} footpaths under {TRANSFER_METRES:.0f}m")
    transfers = list(zip(sources.tolist(), targets.tolist(),
                         walk_seconds(metres).tolist()))

    table = TransitTable.build(len(keep), patterns, transfers)
    split = table.n_patterns - len(patterns)
    print(f"  table:     {table.n_patterns:,} scannable patterns "
          f"({split} split off as overtaking)")

    point_nodes, point_offsets = snap(walk_net, points.lat.to_numpy(),
                                      points.lon.to_numpy(), max_metres=None)
    access_stops, access_seconds = access_edges(walk_net, point_nodes, stop_nodes,
                                                ACCESS_METRES)
    reachable = (access_seconds < INFINITY).sum(axis=1)
    print(f"  access:    median {int(np.median(reachable))} stops within "
          f"{ACCESS_METRES:.0f}m, {int((reachable == 0).sum())} places with none")

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
