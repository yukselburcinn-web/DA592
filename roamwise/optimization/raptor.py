"""
RAPTOR: round-based public transit routing over a GTFS timetable (issue #32,
stage 2).

Why this exists rather than a routing server. The issue settled on
GTFS + OpenTripPlanner, and OTP2's transit router *is* RAPTOR -- but OTP is
built for passenger information, so its API answers "how do I get from A to
B". What this project needs is the one shape OTP2 no longer exposes: the
travel time from one place to *every* other place, because `toptw.py` solves
a trip against a distance/duration matrix. Asked pairwise, Paris' 379
catalogue points are 143,641 queries; RAPTOR answers each origin in a single
run, because filling in every stop is what the algorithm does on the way to
answering any one of them. So we run the algorithm directly.

RAPTOR (Delling, Pajor & Werneck, 2012) has no graph and no priority queue.
Round k holds the earliest arrival at every stop using at most k vehicles:

  1. Scan every route that touches a stop improved in round k-1. Walk it from
     the earliest such stop, riding the earliest boardable trip and improving
     the arrival at each later stop.
  2. Relax footpaths out of the stops just improved.

Rounds are transfers, so five rounds is "at most four changes" -- which is
also why the result is naturally Pareto-correct over (time, transfers)
without any dominance bookkeeping.

**Every origin is solved at once.** The scan of one route is the same
sequence of steps whatever the origin; only the times differ. So arrivals are
held as an (origins x stops) array and each step is one numpy operation over
all origins together. This is what makes a full matrix cost about the same as
a handful of individual queries.

Correctness notes, both of which are quiet-wrong-answer bugs if skipped:
  - **Overtaking trips.** RAPTOR boards the earliest *departing* trip and
    never reconsiders, which is optimal only if trips on a pattern keep their
    order the whole way. An express sharing a stop sequence with an
    all-stopper breaks that, and the cost is not a slow answer but a wrong
    one. Trips are split into non-overtaking blocks at build time
    (`_non_overtaking_blocks`), which is the preprocessing the algorithm
    assumes has already happened.
  - **Past-midnight service.** GTFS writes 00:40 on a service day as
    "24:40:00", and a night bus that is dropped or clamped at 24:00 silently
    makes late-evening trips unreachable. Times stay as seconds from noon
    minus 12h with no wraparound, so 24:40 is simply 88800.
"""
from dataclasses import dataclass

import numpy as np

# Time to change vehicles at a stop, when it is not the first boarding. The
# feed's own `min_transfer_time` covers specific stations; this is the floor
# everywhere else -- a platform change is never instant, and treating it as
# instant is how a matrix ends up promising connections nobody can make.
DEFAULT_CHANGE_SECONDS = 120
# Rounds. Five means up to four changes, which covers any journey inside a
# city; a sixth round measurably changed nothing on the Paris feed.
DEFAULT_ROUNDS = 5
INFINITY = np.int32(2 ** 30)


@dataclass
class TransitTable:
    """A GTFS timetable in the shape RAPTOR scans it.

    Patterns, not routes: GTFS `route_id` groups trips that may call at
    different stops, and RAPTOR needs "trips that visit exactly this sequence
    of stops" so one binary search can pick a trip. Built by
    `pipeline/build_transit_matrix.py`.

    pattern_stops/pattern_offsets  CSR: stop indices each pattern calls at.
    dep/arr                        per pattern, (trips x stops) second-of-day.
    order                          per pattern, argsort of dep down each stop
                                   column -- see the overtaking note above.
    stop_patterns/stop_offsets     CSR: (pattern, position) pairs per stop.
    transfer_*                     footpaths, grouped by destination stop so a
                                   whole relaxation is one reduceat. Only
                                   destinations with at least one incoming
                                   footpath appear -- see `_relax_transfers`.
    """
    n_stops: int
    pattern_stops: np.ndarray
    pattern_offsets: np.ndarray
    dep: list
    arr: list
    order: list
    stop_patterns: np.ndarray
    stop_positions: np.ndarray
    stop_offsets: np.ndarray
    transfer_from: np.ndarray
    transfer_to_offsets: np.ndarray
    transfer_targets: np.ndarray
    transfer_seconds: np.ndarray

    @property
    def n_patterns(self) -> int:
        return len(self.dep)

    @classmethod
    def build(cls, n_stops: int, patterns: list, transfers: list) -> "TransitTable":
        """patterns: [(stop_ids, departures, arrivals)] -- `stop_ids` the
        sequence a pattern calls at, the other two (trips x stops) arrays of
        seconds. transfers: [(from_stop, to_stop, seconds)], both directions
        given explicitly if both are walkable.

        Everything RAPTOR needs but a timetable does not carry is derived
        here, so the pipeline's GTFS reader and the tests build the same
        object through the same code."""
        pattern_stops, offsets = [], [0]
        dep, arr, order = [], [], []
        for stop_ids, departures, arrivals in patterns:
            stop_ids = np.asarray(stop_ids, dtype=np.int32)
            departures = np.asarray(departures, dtype=np.int32)
            arrivals = np.asarray(arrivals, dtype=np.int32)
            for block in _non_overtaking_blocks(departures, arrivals):
                pattern_stops.append(stop_ids)
                offsets.append(offsets[-1] + len(stop_ids))
                dep.append(departures[block])
                arr.append(arrivals[block])
                # Sorted per stop column rather than once by the first stop.
                # Within a block this is the same order at every stop, and
                # keeping it per column means a pattern that slipped through
                # the split still boards the right trip.
                order.append(np.argsort(departures[block], axis=0,
                                        kind="stable").astype(np.int32))

        flat_stops = (np.concatenate(pattern_stops) if pattern_stops
                      else np.empty(0, dtype=np.int32))
        owners, positions = [], []
        for index, stop_ids in enumerate(pattern_stops):
            owners.append(np.full(len(stop_ids), index, dtype=np.int32))
            positions.append(np.arange(len(stop_ids), dtype=np.int32))
        owners = np.concatenate(owners) if owners else np.empty(0, dtype=np.int32)
        positions = np.concatenate(positions) if positions else np.empty(0, dtype=np.int32)
        by_stop = np.argsort(flat_stops, kind="stable")
        stop_offsets = np.searchsorted(flat_stops[by_stop], np.arange(n_stops + 1))

        if transfers:
            edges = np.array(transfers, dtype=np.int64)
            by_target = np.argsort(edges[:, 1], kind="stable")
            edges = edges[by_target]
            targets, starts = np.unique(edges[:, 1], return_index=True)
            transfer_offsets = np.append(starts, len(edges)).astype(np.int64)
            transfer_from = edges[:, 0].astype(np.int32)
            transfer_seconds = edges[:, 2].astype(np.int32)
            transfer_targets = targets.astype(np.int32)
        else:
            transfer_from = np.empty(0, dtype=np.int32)
            transfer_seconds = np.empty(0, dtype=np.int32)
            transfer_targets = np.empty(0, dtype=np.int32)
            transfer_offsets = np.zeros(1, dtype=np.int64)

        return cls(n_stops=n_stops, pattern_stops=flat_stops,
                   pattern_offsets=np.array(offsets, dtype=np.int64),
                   dep=dep, arr=arr, order=order,
                   stop_patterns=owners[by_stop], stop_positions=positions[by_stop],
                   stop_offsets=stop_offsets.astype(np.int64),
                   transfer_from=transfer_from,
                   transfer_to_offsets=transfer_offsets,
                   transfer_targets=transfer_targets,
                   transfer_seconds=transfer_seconds)


def _non_overtaking_blocks(departures: np.ndarray, arrivals: np.ndarray) -> list:
    """Split a pattern's trips into groups where no trip overtakes another.

    RAPTOR boards the earliest *departing* trip and never reconsiders, which
    is optimal exactly when trips on a pattern keep their order all the way
    along. Feeds break that: an express and an all-stopper can share a stop
    sequence, and boarding the all-stopper because it left first then arrives
    an hour late. Classic RAPTOR assumes the timetable was preprocessed so
    this cannot happen, so this is that preprocessing -- each block is scanned
    as its own pattern, and the express wins on its own merits.

    Greedy and first-fit, which is enough: real feeds produce one block for
    almost every pattern and two for the handful that mix stopping patterns.
    """
    n_trips = len(departures)
    if n_trips <= 1:
        return [np.arange(n_trips, dtype=np.int32)]
    blocks = []
    for trip in np.argsort(departures[:, 0], kind="stable"):
        for block in blocks:
            last = block[-1]
            if ((departures[trip] >= departures[last]).all()
                    and (arrivals[trip] >= arrivals[last]).all()):
                block.append(trip)
                break
        else:
            blocks.append([trip])
    return [np.array(block, dtype=np.int32) for block in blocks]


def _relax_transfers(table: TransitTable, best: np.ndarray, marked: np.ndarray):
    """One footpath hop out of every stop, for every origin at once.

    Edges are grouped by destination so the scatter-minimum is a `reduceat`
    over a contiguous block per destination rather than 50,000 indexed
    updates -- the difference between a second and a minute per round."""
    if not len(table.transfer_from):
        return
    # The builder emits only destinations that actually have an incoming
    # footpath, so every group here is non-empty -- reduceat over an empty
    # group returns the element sitting at that offset rather than the
    # identity, and a stop would silently inherit a stranger's arrival time.
    candidate = best[:, table.transfer_from] + table.transfer_seconds
    reduced = np.minimum.reduceat(candidate, table.transfer_to_offsets[:-1], axis=1)
    targets = table.transfer_targets
    improved = reduced < best[:, targets]
    if improved.any():
        best[:, targets] = np.minimum(best[:, targets], reduced)
        marked[targets] |= improved.any(axis=0)


def earliest_arrivals(table: TransitTable, access_stops: np.ndarray,
                      access_seconds: np.ndarray, departure_seconds: np.ndarray,
                      rounds: int = DEFAULT_ROUNDS,
                      change_seconds: int = DEFAULT_CHANGE_SECONDS) -> np.ndarray:
    """Earliest arrival (seconds) at every stop, for every origin.

    access_stops/access_seconds: (origins x k) the stops each origin can walk
    to and how long that walk takes; INFINITY in `access_seconds` marks a
    padding slot, which is how origins with different numbers of reachable
    stops share one rectangular array.

    departure_seconds: (origins,) when each origin sets out.

    Returns (origins x stops), INFINITY where a stop is unreachable.
    """
    n_origins = len(departure_seconds)
    best = np.full((n_origins, table.n_stops), INFINITY, dtype=np.int32)
    rows = np.arange(n_origins)

    reachable = access_seconds < INFINITY
    for column in range(access_stops.shape[1]):
        live = reachable[:, column]
        if not live.any():
            continue
        stops = access_stops[live, column]
        arrive = departure_seconds[live] + access_seconds[live, column]
        current = best[rows[live], stops]
        best[rows[live], stops] = np.minimum(current, arrive)

    marked = np.zeros(table.n_stops, dtype=bool)
    marked[access_stops[reachable]] = True
    _relax_transfers(table, best, marked)

    for round_index in range(rounds):
        previous = best.copy()
        touched = np.zeros(table.n_stops, dtype=bool)
        # The first round boards from the walk in, which is not a vehicle
        # change: charging the transfer penalty there would push every
        # journey's first departure two minutes later than it really is.
        change = 0 if round_index == 0 else change_seconds
        for pattern in _queue(table, marked):
            _scan_pattern(table, pattern, best, previous, touched, change)
        if not touched.any():
            break
        marked = touched
        _relax_transfers(table, best, marked)
    return best


def _queue(table: TransitTable, marked: np.ndarray) -> np.ndarray:
    """Patterns that call at any stop improved last round. RAPTOR also tracks
    the earliest *position* to start scanning from; we scan from the first
    marked position per pattern, which is the same thing."""
    stops = np.flatnonzero(marked)
    if not len(stops):
        return np.empty(0, dtype=np.int64)
    pieces = [table.stop_patterns[table.stop_offsets[s]:table.stop_offsets[s + 1]]
              for s in stops]
    return np.unique(np.concatenate(pieces)) if pieces else np.empty(0, dtype=np.int64)


def _scan_pattern(table: TransitTable, pattern: int, best: np.ndarray,
                  previous: np.ndarray, touched: np.ndarray, change_seconds: int):
    """Ride one pattern from end to end, for every origin at once.

    Each origin carries its own "trip I am currently on" (-1 for none). At
    every stop we first let whoever is aboard improve their arrival, then let
    anyone board an earlier trip than the one they are on -- which is the
    hop-on-the-first-thing-that-comes rule that makes a single forward pass
    optimal."""
    start = table.pattern_offsets[pattern]
    stop_ids = table.pattern_stops[start:table.pattern_offsets[pattern + 1]]
    dep, arr, order = table.dep[pattern], table.arr[pattern], table.order[pattern]
    n_trips = dep.shape[0]
    n_origins = best.shape[0]
    riding = np.full(n_origins, -1, dtype=np.int32)
    rows = np.arange(n_origins)

    for position, stop in enumerate(stop_ids):
        aboard = riding >= 0
        if aboard.any():
            arrival = arr[riding[aboard], position]
            current = best[aboard, stop]
            better = arrival < current
            if better.any():
                index = rows[aboard][better]
                best[index, stop] = arrival[better]
                touched[stop] = True

        # Board here? Only from an arrival that already existed at the start
        # of this round -- boarding on the strength of this round's own
        # arrivals would let one round use two vehicles.
        ready = previous[:, stop]
        if position == len(stop_ids) - 1:
            continue
        ready = np.where(ready >= INFINITY, INFINITY, ready + change_seconds)
        candidate_slot = np.searchsorted(dep[order[:, position], position], ready)
        has_candidate = candidate_slot < n_trips
        if not has_candidate.any():
            continue
        candidate = np.where(has_candidate, order[np.minimum(candidate_slot, n_trips - 1),
                                                  position], -1)
        current_departure = np.where(riding >= 0, dep[riding, position], INFINITY)
        candidate_departure = np.where(candidate >= 0, dep[candidate, position], INFINITY)
        take = has_candidate & (candidate_departure < current_departure)
        riding[take] = candidate[take]
