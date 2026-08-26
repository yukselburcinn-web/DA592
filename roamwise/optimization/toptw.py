"""The router's optimizer: a Team Orienteering Problem with Time Windows.

The router used to solve a TSP -- order every candidate, minimise distance,
then drop whatever did not fit. That is the wrong question. A traveler's day
is time-limited and the catalogue is larger than the day, so the decision is
not *what order* but *which stops*: pick the subset worth visiting, assign it
to days, and sequence it, all against real opening hours. TOPTW is the
canonical formulation of exactly that, and the six special-case passes the
old router grew -- day filling, day rebalancing, meal insertion, evening
insertion, a nightlife hour floor and a nightlife-last reorder -- are all
constraints of this one model instead (issue #72).

Those passes were not merely redundant, they interacted. Each ran over a
route the previous one had settled, against the same budget, with no view of
the others: measured on the same candidate set, raising `min_food_per_day`
from 0 to 2 *removed* a nightlife stop, and the two-meal guarantee itself
held on only 28 of 72 measured days -- falling to 4 of 24 on eighteen-hour
days, because the longer the day the more passes competed over it. Under this
model it holds on all of them.

The model, as OR-Tools sees it:

  vehicles      one per day of the trip
  timeline      one absolute clock across the whole trip; day d owns the
                minutes [d*1440 + start, d*1440 + start + budget]
  nodes         one copy of each POI *per day*, pinned to that day's vehicle
                and carrying that day's opening window
  disjunction   a POI's copies, max cardinality 1 -- visit it on one of the
                days, or pay `drop_penalty_m` to skip it
  arc cost      distance in metres
  dimensions    Time (with waiting slack), Meals (exactly n per day),
                and one per category (the diversity cap)

The per-day copies are load-bearing, not tidiness. Modelled the other way --
one node per POI carrying the union of every day's windows -- nothing binds a
window to a vehicle until the routing is half-built, propagation collapses,
and a 71-POI pool does not find a non-empty route in twenty seconds. Pinned
copies bind each window to one vehicle up front, and the same pool solves in
under a second.

Scale: comfortable to roughly 120 POIs (about 360 nodes, ~2s). A full city
catalogue (371 POIs, 1113 nodes) did not solve in ten minutes, which is why
callers select candidates before handing them here -- see
`optimization.scoring.select_by_score`.
"""
import datetime

from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from ortools.util.optional_boolean_pb2 import BOOL_TRUE as _BOOL_TRUE

from roamwise.optimization.routing import (
    DEFAULT_DAY_START_HOUR, FOOD_CATEGORY, NIGHTLIFE_EARLIEST_HOUR,
    _build_distance_functions, _is_food, _is_nightlife, _opening_intervals)
from roamwise.optimization.travel_modes import DEFAULT_MODE

# What a stop is worth, in metres of extra walking, when nothing says
# otherwise. Swept over 72 days x two pool sizes: below 2000 the solver buys
# distance by dropping stops the old router kept, and from 8000 up the
# frontier is flat because the candidate pool rather than the penalty is the
# binding constraint. 8000 is the cheapest point on the flat part -- it takes
# every stop the pool can offer without paying for stops nobody asked for.
DEFAULT_DROP_PENALTY_M = 8000
# At most this many stops of one category in a day, the diversity factor from
# the issue's score formula. It cannot be a node weight -- "the third museum
# today" is a property of a partial route, not of a POI -- so it lives here,
# as the category quota a constraint model can express and a chain of passes
# cannot. Food is exempt: it carries its own exact-per-day contract.
#
# 4, swept over 72 days at two pool sizes against the pre-#72 router:
#
#   pool          cap 3            cap 4            cap 5           uncapped
#   24    6.08 / 0.862     6.69 / 0.906     7.15 / 1.006     7.36 / 1.084
#   72    8.85 / 0.536     9.06 / 0.471     9.28 / 0.481     9.32 / 0.474
#                          (stops per day / km per stop)
#
# The two pools disagree, and that is the whole reason this value is not
# obvious. On the retrieved pool the app asks for today (24 POIs, and for a
# Culture Enthusiast 19 of them museums) the cap is a hard constraint: 3 costs
# stops outright against the old router, and loosening it buys stops by
# spending distance. On a 72-POI pool there is enough variety that the cap
# stops binding past 4 -- km per stop is flat from there (0.471, 0.481, 0.474)
# and the extra stops are close to free.
#
# 4 is the value that is defensible on both. It is the km-per-stop optimum on
# the larger pool, and on the smaller one it still beats the old router on
# stops (+7.8%) where 3 does not (-2.0%). Going further trades away the
# diversity the cap exists for, and buys little the larger pool does not
# already give.
DEFAULT_MAX_SAME_CATEGORY = 4
# A fixed iteration budget rather than a wall-clock one, so the same input
# gives the same itinerary on any machine and under any load. Verified: three
# consecutive solves of the same case return byte-identical routes.
SOLUTION_LIMIT = 150
# Where each meal aims, as a fraction of the day's own window rather than a
# fixed clock time. Fixed times (13:00, 19:00) are wrong for a day running
# 09:00-15:00 -- dinner falls outside it and the meal is simply dropped.
# Fractions degrade sensibly: on a 12-hour day from 09:00 they land at 13:48
# and 19:12, real lunch and dinner; on a 6-hour day they become an early and a
# late lunch, the honest answer for someone out only until 15:00.
MEAL_DAY_FRACTIONS = (0.40, 0.85)
# How far from its target a meal may sit. Wide enough to reach a restaurant
# the traveler is already passing, narrow enough that two meals cannot both
# land at noon -- which a bare "two food stops a day" count would allow, and
# which is why that count alone is not what the issue asked for.
MEAL_WINDOW_HOURS = 2.0
# The hours people actually eat, and the least a meal may be from the next.
# Fractions of the day window alone are not enough: a Nightlife Seeker's day
# starts at 15:00, so on an 18-hour day the two fractions landed at 22:12 and
# *06:18* -- an hour with no restaurant open anywhere in the catalogue, which
# is why ten of the twelve Nightlife Seeker trips came back a meal short. The
# band is what keeps a meal at a mealtime; the gap is what stops two of them
# collapsing onto each other once the band has clamped them.
EATING_BAND = (11.5, 22.5)
MIN_MEAL_GAP_HOURS = 3.0
# What a missed sitting costs, as a multiple of one dropped stop. It has to
# outweigh the stops a day gives up to fit a meal in, and nothing more: set
# far above the rest of the objective (10,000,000 was the first attempt) the
# guided local search degenerates and returns an empty trip -- every stop
# dropped -- because one term dwarfs every gradient it could follow. Measured:
# at 100,000 the solver returns nothing, at 3x the drop penalty it fills the
# days and honours both sittings.
MEAL_SHORTFALL_MULTIPLE = 3
# A day the traveler asked for should not come back empty while there is still
# something to put in it. TOPTW on its own has no reason to spread: packing
# every stop into one day collects the same score for less distance, so a
# two-day trip offering four museums returned all four on day one and nothing
# on day two. This is the constraint form of what `_rebalance_days` used to do
# by hand, and it binds only when the pool is small -- on a real retrieved pool
# the time budget fills the days without it.
MIN_STOPS_PER_DAY = 1
_DEFAULT_VISIT_MINUTES = 60


def _visit_minutes(poi: dict) -> int:
    return int(round(poi.get("avg_visit_minutes", _DEFAULT_VISIT_MINUTES)))


def meal_target_hours(day_start_hour: float, daily_minutes_budget: int,
                       n_meals: int) -> list[float]:
    """Clock time each meal aims at: spread over the day, then pulled back to
    hours a kitchen is open.

    The fractions come first because they are what makes a short day work --
    a day running 09:00-15:00 has no dinner in it, and two sittings inside
    what it does have is the honest answer. The band comes second because the
    fractions alone do not know what a mealtime is: a day starting at 15:00
    put its second sitting after midnight. When clamping pushes two sittings
    within MIN_MEAL_GAP_HOURS of each other, they are respread evenly across
    whatever of the band the day covers, which is the case the band was
    needed for in the first place."""
    if n_meals < 1:
        return []
    span = daily_minutes_budget / 60
    fractions = (list(MEAL_DAY_FRACTIONS) if n_meals <= 2
                 else [(i + 1) / (n_meals + 1) for i in range(n_meals)])
    low = max(day_start_hour, EATING_BAND[0])
    high = min(day_start_hour + span, EATING_BAND[1])
    if low >= high:  # the day does not overlap any mealtime at all
        return [day_start_hour + f * span for f in fractions[:n_meals]]

    targets = [min(max(day_start_hour + f * span, low), high)
               for f in fractions[:n_meals]]
    too_close = any(b - a < MIN_MEAL_GAP_HOURS
                    for a, b in zip(targets, targets[1:]))
    if too_close:
        targets = [low + (i + 1) / (n_meals + 1) * (high - low)
                   for i in range(n_meals)]
    return targets


def _earliest_hour_for(poi: dict):
    """A bar is not worth visiting the moment its doors unlock. Several carry
    early OSM hours, so the category has an earliest *sensible* hour of its
    own (issue #59) -- here it is simply the lower bound of the time window
    rather than a separate reordering pass."""
    return NIGHTLIFE_EARLIEST_HOUR if _is_nightlife(poi) else None


def _day_windows(poi: dict, day: int, day_start_hour: float, budget_minutes: int,
                  start_date, respect_opening_hours: bool,
                  slot_hours: tuple[float, float] = None) -> list[tuple[int, int]]:
    """When this POI may be *arrived at* on this day, in absolute minutes.

    Two rules, the same two `optimize_day_route` applied one stop at a time:
    walk in before it closes, and finish inside the day's budget. A visit that
    runs past closing time is allowed, exactly as it was before -- the model is
    not held to a stricter standard than the router it replaces."""
    base = day * 1440 + int(round(day_start_hour * 60))
    day_end = base + budget_minutes
    if slot_hours is not None:  # a meal slot narrows the day to lunch or dinner
        base = max(base, day * 1440 + int(round(slot_hours[0] * 60)))
        day_end = min(day_end, day * 1440 + int(round(slot_hours[1] * 60)))
    visit = _visit_minutes(poi)
    if not respect_opening_hours:
        return [(base, day_end - visit)] if base <= day_end - visit else []

    day_date = None if start_date is None else start_date + datetime.timedelta(days=day)
    windows = []
    for open_hour, close_hour in _opening_intervals(poi, _earliest_hour_for(poi), day_date):
        low = max(base, day * 1440 + int(round(open_hour * 60)))
        high = min(day_end - visit, day * 1440 + int(round(close_hour * 60)) - 1)
        if low <= high:
            windows.append((low, high))
    return windows


def solve(pois: list[dict], n_days: int, start_hub: dict = None,
          daily_minutes_budget: int = 480, day_start_hour: float = 9.0,
          respect_opening_hours: bool = True, start_date=None,
          distance_fn=None, duration_fn=None, used_real_routing: bool = False,
          min_food_per_day: int = 0, drop_penalty_m: int = DEFAULT_DROP_PENALTY_M,
          max_same_category: int = DEFAULT_MAX_SAME_CATEGORY) -> list[dict]:
    """One model, every day. Returns one dict per day in the shape the rest of
    the app already reads -- route, distance_km, total/active/idle minutes,
    schedule -- so views and narration need no change.

    `pois` is the whole working set, meals included; which of them get visited,
    on which day, in which order and at what hour is what this decides.
    """
    days_out = [_empty_day(v, start_date) for v in range(n_days)]
    if not pois or n_days < 1:
        return days_out

    meals = [(k, low, high) for k, target in enumerate(
        meal_target_hours(day_start_hour, daily_minutes_budget, min_food_per_day))
        for low, high in [(target - MEAL_WINDOW_HOURS, target + MEAL_WINDOW_HOURS)]]

    # One node per (POI, day, slot). Sightseeing has a single slot spanning the
    # whole day; a meal POI gets one slot per sitting, each carrying that
    # sitting's window -- which is how "lunch and dinner" becomes a constraint
    # rather than a count. A bare count of two food stops a day is satisfied by
    # two lunches at 11:00 and 11:45, and that is what the old
    # `_ensure_daily_meals` pass existed to prevent.
    copies = []  # (poi_index, day, slot or None)
    for poi_i, poi in enumerate(pois):
        is_meal = min_food_per_day > 0 and _is_food(poi)
        for day in range(n_days):
            for slot in ([k for k, _, _ in meals] if is_meal else [None]):
                copies.append((poi_i, day, slot))

    depot = 0
    end_node = 1 + len(copies)
    n_nodes = end_node + 1
    hub = start_hub or pois[0]
    coords = [hub] + [pois[c[0]] for c in copies] + [hub]

    def entry_of(node: int):
        """(poi, day, slot) for a copy, or None for either depot."""
        return None if node in (depot, end_node) else (
            pois[copies[node - 1][0]], copies[node - 1][1], copies[node - 1][2])

    # Precomputed: the callbacks are hit hundreds of thousands of times during
    # local search, and recomputing haversine inside each one dominated the solve.
    dist_m = [[0] * n_nodes for _ in range(n_nodes)]
    time_m = [[0] * n_nodes for _ in range(n_nodes)]
    for a in range(n_nodes):
        for b in range(n_nodes):
            if a == b or a == end_node or b == end_node:
                continue
            dist_m[a][b] = int(round(distance_fn(coords[a], coords[b]) * 1000))
            time_m[a][b] = int(round(duration_fn(coords[a], coords[b])))

    manager = pywrapcp.RoutingIndexManager(n_nodes, n_days, [depot] * n_days,
                                            [end_node] * n_days)
    routing = pywrapcp.RoutingModel(manager)

    def distance_cb(i, j):
        return dist_m[manager.IndexToNode(i)][manager.IndexToNode(j)]

    def time_cb(i, j):
        a, b = manager.IndexToNode(i), manager.IndexToNode(j)
        entry = entry_of(a)
        return (_visit_minutes(entry[0]) if entry else 0) + time_m[a][b]

    routing.SetArcCostEvaluatorOfAllVehicles(routing.RegisterTransitCallback(distance_cb))
    time_idx = routing.RegisterTransitCallback(time_cb)

    # An empty vehicle is "unused" by default, and an unused vehicle's soft
    # cumul penalties are left out of the objective entirely -- so every
    # per-day floor below (a meal at each sitting, at least one stop) silently
    # stopped applying to exactly the days that needed it: the empty ones. It
    # is why raising those penalties changed nothing.
    for v in range(n_days):
        routing.SetVehicleUsedWhenEmpty(True, v)

    routing.AddDimension(time_idx, daily_minutes_budget, (n_days + 1) * 1440, False, "Time")
    time_dim = routing.GetDimensionOrDie("Time")
    for v in range(n_days):
        base = v * 1440 + int(round(day_start_hour * 60))
        time_dim.CumulVar(routing.Start(v)).SetRange(base, base)
        time_dim.CumulVar(routing.End(v)).SetRange(base, base + daily_minutes_budget)

    solver = routing.solver()
    slot_hours = {k: (low, high) for k, low, high in meals}
    by_poi: dict[int, list[int]] = {}
    for node, (poi_i, day, slot) in enumerate(copies, start=1):
        index = manager.NodeToIndex(node)
        by_poi.setdefault(poi_i, []).append(index)
        routing.VehicleVar(index).SetValues([-1, day])  # this copy is that day's
        windows = _day_windows(pois[poi_i], day, day_start_hour, daily_minutes_budget,
                               start_date, respect_opening_hours,
                               slot_hours.get(slot) if slot is not None else None)
        if not windows:
            solver.Add(routing.ActiveVar(index) == 0)  # shut, or shut at this sitting
            continue
        cumul = time_dim.CumulVar(index)
        cumul.SetRange(min(w[0] for w in windows), max(w[1] for w in windows))
        if len(windows) > 1:  # e.g. a lunchtime closure splits the day
            solver.Add(solver.Max([(cumul >= low) * (cumul <= high)
                                   for low, high in windows]) == 1)
        else:
            cumul.SetRange(*windows[0])

    # Visit each POI once across the whole trip, or pay to skip it. Spanning
    # every copy -- all days, all sittings -- is also what stops the same
    # restaurant being booked twice.
    for poi_i in sorted(by_poi):
        routing.AddDisjunction(by_poi[poi_i], max(int(drop_penalty_m), 1), 1)

    for k, _, _ in meals:
        # One sitting filled per day, as a constraint of the model rather than
        # a pass that runs afterwards and evicts what an earlier pass placed
        # (#20, #29). The upper bound matters as much as the lower one: without
        # it, under a pool where restaurants outnumber sights and cluster
        # tightly, the cheapest way to collect stops is a food crawl -- measured,
        # a day came back holding nine restaurants and nothing else.
        def meal_cb(i, _j, k=k):
            entry = entry_of(manager.IndexToNode(i))
            return 1 if entry and entry[2] == k else 0

        name = f"Meal_{k}"
        routing.AddDimension(routing.RegisterTransitCallback(meal_cb), 0, len(pois),
                             True, name)
        dim = routing.GetDimensionOrDie(name)
        for v in range(n_days):
            dim.CumulVar(routing.End(v)).SetMax(1)
            # Soft below, so a day with no restaurant open at that sitting stays
            # solvable rather than making the whole trip infeasible.
            dim.SetCumulVarSoftLowerBound(
                routing.End(v), 1, max(int(drop_penalty_m) * MEAL_SHORTFALL_MULTIPLE, 1))

    # Every day gets something, if anything can go in it.
    def stop_cb(i, _j):
        return 1 if entry_of(manager.IndexToNode(i)) else 0

    routing.AddDimension(routing.RegisterTransitCallback(stop_cb), 0, len(copies),
                         True, "Stops")
    stops_dim = routing.GetDimensionOrDie("Stops")
    for v in range(n_days):
        stops_dim.SetCumulVarSoftLowerBound(
            routing.End(v), MIN_STOPS_PER_DAY,
            max(int(drop_penalty_m) * MEAL_SHORTFALL_MULTIPLE, 1))

    if max_same_category:
        for category in sorted({p.get("category") for p in pois} - {FOOD_CATEGORY, None}):
            def category_cb(i, _j, category=category):
                entry = entry_of(manager.IndexToNode(i))
                return 1 if entry and entry[0].get("category") == category else 0

            routing.AddDimension(routing.RegisterTransitCallback(category_cb), 0,
                                 max_same_category, True, f"Cat_{category}")

    params = pywrapcp.DefaultRoutingSearchParameters()
    # PARALLEL_CHEAPEST_INSERTION, not the usual PATH_CHEAPEST_ARC. Every day
    # here carries per-vehicle quotas -- one stop per meal sitting, a cap per
    # category -- and PATH_CHEAPEST_ARC builds routes one vehicle at a time, so
    # it spends the reachable meal stops on the first day and dead-ends on the
    # rest. Measured on a 24-candidate Paris pool it returned [0, 0, 7] stops
    # across the three days; building every day at once returns [6, 7, 6] on
    # the same input. Guided local search could not repair it afterwards --
    # raising the iteration limit sevenfold changed nothing.
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION)
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    # Moving a POI from one day to another means deactivating that day's copy
    # and activating another day's -- two nodes, one decision. No default
    # local-search operator makes that move, so an unbalanced first solution
    # stayed unbalanced: four museums over two days came back [4, 0], the
    # empty day's penalty paid rather than fixed. These two operators swap an
    # active node for an inactive one, which is exactly the move the copies
    # made necessary.
    params.local_search_operators.use_swap_active = _BOOL_TRUE
    params.local_search_operators.use_extended_swap_active = _BOOL_TRUE
    params.solution_limit = SOLUTION_LIMIT
    solution = routing.SolveWithParameters(params)
    if solution is None:
        return days_out

    for v in range(n_days):
        route, schedule = [], []
        km, active, previous = 0.0, 0.0, start_hub
        index = routing.Start(v)
        while not routing.IsEnd(index):
            entry = entry_of(manager.IndexToNode(index))
            if entry:
                poi = entry[0]
                arrival = solution.Value(time_dim.CumulVar(index)) / 60 - v * 24
                visit = _visit_minutes(poi)
                if previous is not None:
                    km += distance_fn(previous, poi)
                    active += duration_fn(previous, poi)
                active += visit
                route.append(poi)
                schedule.append({"arrival": arrival, "finish": arrival + visit / 60})
                previous = poi
            index = solution.Value(routing.NextVar(index))
        days_out[v] = _finish_day(v, start_date, route, schedule, km, active,
                                  day_start_hour, used_real_routing)
    return days_out


def _empty_day(day: int, start_date) -> dict:
    return {"day": day + 1,
            "date": None if start_date is None else start_date + datetime.timedelta(days=day),
            "route": [], "distance_km": 0.0, "total_minutes": 0, "active_minutes": 0,
            "idle_minutes": 0, "schedule": [], "used_real_routing": False}


def _finish_day(day: int, start_date, route, schedule, km, active,
                day_start_hour, used_real_routing) -> dict:
    if not schedule:
        return _empty_day(day, start_date)
    # Derived from each other rather than rounded independently, so the three
    # always reconcile and no caller can show a breakdown that doesn't add up.
    span = int(round((max(s["finish"] for s in schedule) - day_start_hour) * 60))
    active_minutes = min(int(round(active)), span)
    return {
        "day": day + 1,
        "date": None if start_date is None else start_date + datetime.timedelta(days=day),
        "route": route,
        "distance_km": round(km, 2),
        "total_minutes": span,
        "active_minutes": active_minutes,
        "idle_minutes": span - active_minutes,
        "schedule": schedule,
        "used_real_routing": used_real_routing,
    }


def build_multi_day_itinerary(pois: list[dict], n_days: int, start_hub: dict = None,
                               daily_minutes_budget: int = 480,
                               day_start_hour: float = DEFAULT_DAY_START_HOUR,
                               respect_opening_hours: bool = True,
                               use_real_routing: bool = False, travel_mode=DEFAULT_MODE,
                               food_pois: list[dict] = None, min_food_per_day: int = 0,
                               start_date=None,
                               drop_penalty_m: int = DEFAULT_DROP_PENALTY_M,
                               max_same_category: int = DEFAULT_MAX_SAME_CATEGORY) -> list[dict]:
    """The router's entry point: candidates in, one routed day per trip day out.

    Note what this signature no longer takes. It used to be handed
    `pois_by_zone` -- KMeans had already decided which POIs belonged to which
    day, and everything after that could only shuffle stops between days that
    geography had already fixed. Day assignment is a decision the model makes
    now, jointly with selection and ordering, so the caller passes a flat pool
    and says how many days it has.

    `food_pois` are candidates for the day's meals. They are kept apart from
    the sightseeing pool for the same reason as before -- retrieval is
    preference-driven and a Culture Enthusiast's query surfaces no restaurants
    (issue #20) -- but they are now solved *with* everything else rather than
    inserted into a finished route.

    `start_date` is the trip's first day. Day N falls on `start_date + N-1`,
    and that date is what lets opening hours be read as the grammar they are
    rather than as a single open/close pair (issue #70).
    """
    meal_pool = list(food_pois or []) if min_food_per_day > 0 else []
    working_set = list(pois) + meal_pool
    if not working_set:
        return [_empty_day(v, start_date) for v in range(max(n_days, 0))]

    # One distance matrix for the whole trip rather than one per day: the
    # public OSRM demo server rate-limits back-to-back requests, and one
    # matrix covering the trip is also just the architecturally correct
    # amount of network I/O for this.
    points = ([start_hub] if start_hub else []) + working_set
    distance_fn, duration_fn, used_real_routing = _build_distance_functions(
        points, use_real_routing, travel_mode)

    return solve(working_set, n_days, start_hub=start_hub,
                 daily_minutes_budget=daily_minutes_budget,
                 day_start_hour=day_start_hour,
                 respect_opening_hours=respect_opening_hours, start_date=start_date,
                 distance_fn=distance_fn, duration_fn=duration_fn,
                 used_real_routing=used_real_routing,
                 min_food_per_day=min_food_per_day, drop_penalty_m=drop_penalty_m,
                 max_same_category=max_same_category)
