"""Issue #72's first task: does a real TOPTW solver buy the extra stops
*without* paying for them in km per stop?

The issue already measured that a greedy time-windowed insertion lifts
stops/day from 7.76 to 8.24 (+6%) but pushes km/stop from 1.04 to 1.16
(+12%) -- and km/stop is what the comparative analysis reports as
*itinerary coherence*, so that is a real cost, not an accounting detail.
The stated expectation was that "most of the +6% can be had without the km
penalty, because a real solver asks 'of all the orders that fit, which is
shortest?' rather than inserting greedily". That expectation was explicitly
NOT verified. This script verifies it.

What is held fixed, so the solver is the only variable:
  the city, the archetype, the retrieved candidate pool, the food pool, the
  start hub, the day start hour, the calendar dates, the travel mode, and
  the distance/duration functions (haversine, no network).

What differs between the two arms:
  baseline -- today's chain: KMeans zoning -> per-day 2-opt -> _fill_days_to_budget
              -> _rebalance_days -> _ensure_evening_stops -> _ensure_daily_meals.
  toptw    -- one OR-Tools model over the whole pool: days are vehicles,
              opening hours are time windows, and every stop is optional
              (a disjunction) with a drop penalty.

The drop penalty is the experiment's knob, and it is denominated in metres:
"a stop is worth taking if it costs less than P metres of extra walking".
Sweeping P traces the stops/km frontier, which is the actual answer to the
issue's open question -- a single operating point would not be.

Scoring is deliberately UNIFORM here (every stop worth the same P). The
issue's score function (preference match x quality x crowding x diversity)
is a separate acceptance criterion; folding it in now would change *which*
stops get picked for reasons unrelated to the geometry, and the count would
no longer be comparable to the baseline's. Geometry first, scoring second.

Run:  python evaluation/toptw_measurement.py
"""
import argparse
import datetime
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from roamwise.agents.fusion_rag_agent import FusionRAGAgent
from roamwise.agents.router_agent import (
    MIN_FOOD_PER_DAY, RouterAgent, start_hour_for)
from roamwise.knowledge_graph.build_graph import DATA_DIR, GraphIndex
from roamwise.optimization.routing import (
    FOOD_CATEGORY, NIGHTLIFE_EARLIEST_HOUR, _build_distance_functions,
    _is_food, _is_nightlife, _opening_intervals)
from roamwise.optimization.toptw import build_multi_day_itinerary
from roamwise.optimization.scoring import (
    PREFERENCE_DIMS, preference_match, quality, score_pois)
from roamwise.retrieval.query import archetype_query

HERE = Path(__file__).parent
RESULTS_CSV = HERE / "toptw_measurement_results.csv"
SUMMARY_CSV = HERE / "toptw_measurement_summary.csv"
# The pre-#72 router's numbers, kept as a file because the code that produced
# them no longer exists: the zoning + six-pass chain was deleted when the
# TOPTW router landed, so the baseline cannot be recomputed from this tree.
# Recover it with `git show <pre-#72-commit>:roamwise/optimization/routing.py`.
BASELINE_CSV = HERE / "toptw_baseline_pre72_results.csv"

CITIES = ["PAR", "BER"]
# Four archetypes spanning the catalogue's shapes: the issue reported its
# per-archetype gains for the first three, and Family Traveler stands in for
# a day whose candidates are neither late-opening nor spread out.
ARCHETYPES = ["Culture Enthusiast", "Nightlife Seeker",
              "Nature & Adventure", "Family Traveler"]
BUDGET_HOURS = [12, 15, 18]
N_DAYS = 3
# Fixed so opening hours resolve against real weekdays (#70) and both arms
# see the same ones. A Friday start, so the trip spans Fri/Sat/Sun.
START_DATE = datetime.date(2026, 9, 25)
TRAVEL_MODE = "walking"
MIN_RETRIEVED_POIS = 12
# How many POIs retrieval hands the router, in total for the trip. Two
# conditions, because the answer turned out to depend on it:
#   24 -- what the app actually asks for today (orchestrator.RETRIEVED_POIS_PER_DAY
#         = 8, times three days). At this size the pool is barely larger than
#         what a day can hold, so there is hardly anything to select *from*.
#   72 -- the size at which the baseline reproduces the numbers issue #72
#         reported (7.76 stops/day, 1.04 km/stop); measured here it gives
#         7.82 and 1.054. The issue's own method section assumes "40-60
#         candidates per day", so this is the condition its table was taken
#         under, and the one its claims have to be answered on.
TOP_K_TOTALS = [24, 72]

# What a stop is worth, in metres of extra walking. Calibrated on
# Paris/Culture Enthusiast/12h: below 2000 the solver buys distance by
# dropping stops the baseline keeps, and from 8000 up the frontier is flat
# because the retrieved pool, not the penalty, is the binding constraint.
# The range therefore spans "too cheap to bother" through saturation, which
# is what makes this a frontier rather than one arbitrary operating point.
DROP_PENALTIES_M = [1000, 2000, 4000, 8000, 16000]

# Deterministic: a fixed iteration budget, never a wall-clock limit, so the
# same input gives the same output on any machine (an acceptance criterion).
SOLUTION_LIMIT = 150
# The diversity factor's constraint form: at most this many stops of one
# category in a day (food excluded -- it carries its own exact-2 contract).
MAX_SAME_CATEGORY_PER_DAY = 4  # the router's default; see toptw.DEFAULT_MAX_SAME_CATEGORY
DEFAULT_VISIT_MINUTES = 60
_MEAL_SHORTFALL_PENALTY = 10_000_000


def _visit_minutes(poi: dict) -> int:
    return int(round(poi.get("avg_visit_minutes", DEFAULT_VISIT_MINUTES)))


def _earliest_for(poi: dict):
    return NIGHTLIFE_EARLIEST_HOUR if _is_nightlife(poi) else None


# --------------------------------------------------------------------------
# Shared metrics: both arms produce the same day shape, so one evaluator
# scores them and neither can be flattered by its own accounting.
# --------------------------------------------------------------------------

def _open_at_arrival(poi: dict, arrival_hour: float, day_date) -> bool:
    """Mirrors the router's own rule: a stop counts as legal if the traveler
    walks in while the doors are open. A visit that runs past closing is what
    today's code already allows, so the TOPTW arm is not held to a stricter
    standard than the thing it is being compared against."""
    for start, end in _opening_intervals(poi, _earliest_for(poi), day_date):
        if start <= arrival_hour < end:
            return True
    return False


def measure(days: list[dict], day_start_hour: float, budget_minutes: int,
            preferences: dict = None) -> dict:
    stops = sum(len(d["route"]) for d in days)
    food_stops = sum(1 for d in days for p in d["route"] if _is_food(p))
    km = sum(d["distance_km"] for d in days)
    closed = 0
    overrun = 0
    meal_ok = 0
    for day in days:
        for poi, slot in zip(day["route"], day["schedule"]):
            if not _open_at_arrival(poi, slot["arrival"], day.get("date")):
                closed += 1
        if day["schedule"]:
            finish = max(s["finish"] for s in day["schedule"])
            if (finish - day_start_hour) * 60 > budget_minutes + 1e-6:
                overrun += 1
        if sum(1 for p in day["route"] if _is_food(p)) >= MIN_FOOD_PER_DAY:
            meal_ok += 1
    # What got picked, not just how far apart it was. Sightseeing only: the
    # meal stops are fixed at two a day in both arms, so including them would
    # dilute every arm identically and hide the difference.
    sights = [p for d in days for p in d["route"] if not _is_food(p)]
    fame = quality(sights)
    pref = ([preference_match(preferences, p.get("category")) for p in sights]
            if preferences else [])
    distinct = [len({p.get("category") for p in d["route"] if not _is_food(p)})
                for d in days]
    return {
        "mean_pref_match": round(sum(pref) / len(pref), 4) if pref else 0.0,
        "mean_quality": round(sum(fame) / len(fame), 4) if fame else 0.0,
        "categories_per_day": round(sum(distinct) / len(days), 3) if days else 0.0,
        "stops": stops,
        "sight_stops": stops - food_stops,
        "food_stops": food_stops,
        "km": round(km, 3),
        "stops_per_day": stops / len(days) if days else 0.0,
        "sight_per_day": (stops - food_stops) / len(days) if days else 0.0,
        "km_per_stop": (km / stops) if stops else 0.0,
        "closed_violations": closed,
        "budget_overruns": overrun,
        "days_with_meals": meal_ok,
        "n_days": len(days),
    }


# --------------------------------------------------------------------------
# The TOPTW arm
# --------------------------------------------------------------------------

def solve_toptw(pool: list[dict], start_hub: dict, n_days: int, budget_minutes: int,
                day_start_hour: float, start_date, distance_fn, duration_fn,
                drop_penalty_m: int, scores: list[float] = None,
                max_same_category: int = None) -> list[dict]:
    """One model, all days. Every POI gets one *copy per day*, each copy
    pinned to that day's vehicle and carrying that day's opening window; a
    disjunction over a POI's copies (max cardinality 1) then says "visit this
    place on one of these days, or pay to skip it". That is what makes
    selection, day assignment and ordering a single decision instead of six
    passes that cannot see each other.

    The copies matter for more than tidiness. Modelled the other way -- one
    node per POI carrying the union of all three days' windows -- the solver
    has nothing tying a window to a vehicle until the routing is already
    half-built, propagation collapses, and a 71-POI pool does not find a
    non-empty route in 20 seconds. Pinned copies bound each window to one
    vehicle up front, and the same pool solves in well under a second."""
    n_pool = len(pool)
    depot = 0
    end_node = 1 + n_pool * n_days

    def copy_index(poi_i: int, day: int) -> int:
        return 1 + poi_i * n_days + day

    def poi_of(node: int):
        """(poi, day) for a copy node, or None for either depot."""
        if node in (depot, end_node):
            return None
        offset = node - 1
        return pool[offset // n_days], offset % n_days

    n_nodes = end_node + 1
    coords = [start_hub] + [pool[i] for i in range(n_pool) for _ in range(n_days)] + [start_hub]

    # Precomputed: the callbacks are hit hundreds of thousands of times during
    # local search, and recomputing haversine inside each one dominated the solve.
    dist_m = [[0] * n_nodes for _ in range(n_nodes)]
    time_m = [[0] * n_nodes for _ in range(n_nodes)]
    for a in range(n_nodes):
        for b in range(n_nodes):
            if a == end_node or b == end_node or a == b:
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
        entry = poi_of(a)
        service = _visit_minutes(entry[0]) if entry else 0
        return service + time_m[a][b]

    routing.SetArcCostEvaluatorOfAllVehicles(
        routing.RegisterTransitCallback(distance_cb))
    time_idx = routing.RegisterTransitCallback(time_cb)

    horizon = (n_days + 1) * 1440
    routing.AddDimension(time_idx, budget_minutes, horizon, False, "Time")
    time_dim = routing.GetDimensionOrDie("Time")

    def slot(day: int) -> tuple[int, int]:
        base = day * 1440 + int(round(day_start_hour * 60))
        return base, base + budget_minutes

    for v in range(n_days):
        lo, hi = slot(v)
        time_dim.CumulVar(routing.Start(v)).SetRange(lo, lo)
        time_dim.CumulVar(routing.End(v)).SetRange(lo, hi)

    solver = routing.solver()
    for poi_i, poi in enumerate(pool):
        visit = _visit_minutes(poi)
        copies = []
        for v in range(n_days):
            index = manager.NodeToIndex(copy_index(poi_i, v))
            copies.append(index)
            # This copy exists only for day v.
            routing.VehicleVar(index).SetValues([-1, v])
            lo, hi = slot(v)
            day_date = None if start_date is None else start_date + datetime.timedelta(days=v)
            windows = []
            for open_h, close_h in _opening_intervals(poi, _earliest_for(poi), day_date):
                w_lo = max(lo, v * 1440 + int(round(open_h * 60)))
                # Arrive before closing, and finish inside the day: the same
                # two rules optimize_day_route applies one stop at a time.
                w_hi = min(hi - visit, v * 1440 + int(round(close_h * 60)) - 1)
                if w_lo <= w_hi:
                    windows.append((w_lo, w_hi))
            if not windows:
                # Shut all day: this copy can never be served.
                solver.Add(routing.ActiveVar(index) == 0)
                continue
            cumul = time_dim.CumulVar(index)
            cumul.SetRange(min(w[0] for w in windows), max(w[1] for w in windows))
            if len(windows) > 1:  # e.g. a lunchtime closure splits the day
                solver.Add(solver.Max([(cumul >= w_lo) * (cumul <= w_hi)
                                       for w_lo, w_hi in windows]) == 1)
        # Visit this POI on at most one day, or pay to skip it. With a score
        # the price of skipping is what the stop is worth to this traveler,
        # which is the whole mechanism by which preferences reach the route;
        # without one every stop costs the same to skip and the model can
        # only reason about geometry and time.
        penalty = drop_penalty_m if scores is None else int(round(drop_penalty_m * scores[poi_i]))
        routing.AddDisjunction(copies, max(penalty, 1), 1)

    # Two meals a day, as a constraint of the model rather than a pass that
    # runs afterwards and evicts what an earlier pass placed (#20, #29). Soft,
    # so a day with no reachable restaurant stays solvable instead of making
    # the whole trip infeasible.
    def food_cb(i, j):
        entry = poi_of(manager.IndexToNode(i))
        return 1 if entry and _is_food(entry[0]) else 0

    routing.AddDimension(routing.RegisterTransitCallback(food_cb), 0,
                         n_pool, True, "Meals")
    meals = routing.GetDimensionOrDie("Meals")
    for v in range(n_days):
        # Exactly the food contract the baseline carries, not a looser one.
        # The baseline keeps food out of the geographic zoning entirely and
        # inserts at most MIN_FOOD_PER_DAY meals per day, so an uncapped
        # TOPTW is not solving the same problem: measured on Paris/Culture
        # Enthusiast, uncapped it returned 23 stops at 0.33 km/stop by
        # filling a day with nine restaurants -- the catalogue holds 47 food
        # POIs against 24 retrieved sights, and they cluster tightly, so
        # under a uniform score a food crawl is simply the cheapest way to
        # collect points. Capped, the extra stops have to come from the
        # sightseeing pool, which is the decision this issue is about.
        meals.CumulVar(routing.End(v)).SetMax(MIN_FOOD_PER_DAY)
        meals.SetCumulVarSoftLowerBound(routing.End(v), MIN_FOOD_PER_DAY,
                                        _MEAL_SHORTFALL_PENALTY)

    if max_same_category is not None:
        # The issue's diversity penalty, expressed where it can actually be
        # evaluated. "The third museum today is worth less than the first" is
        # a fact about a partial route, so it cannot be a node weight; as a
        # per-day count cap it is exactly the kind of category quota the
        # issue notes a constraint model buys you. Food is left out -- it
        # already carries its own exact-2-per-day contract above.
        categories = {p.get("category") for p in pool} - {FOOD_CATEGORY, None}
        for category in sorted(categories):
            def cat_cb(i, _j, category=category):
                entry = poi_of(manager.IndexToNode(i))
                return 1 if entry and entry[0].get("category") == category else 0
            name = f"Cat_{category}"
            routing.AddDimension(routing.RegisterTransitCallback(cat_cb), 0,
                                 max_same_category, True, name)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    params.solution_limit = SOLUTION_LIMIT
    solution = routing.SolveWithParameters(params)
    if solution is None:
        return []

    days = []
    for v in range(n_days):
        route, schedule, km, prev = [], [], 0.0, start_hub
        index = routing.Start(v)
        while not routing.IsEnd(index):
            entry = poi_of(manager.IndexToNode(index))
            if entry:
                poi = entry[0]
                arrival = solution.Value(time_dim.CumulVar(index)) / 60 - v * 24
                km += distance_fn(prev, poi)
                route.append(poi)
                schedule.append({"arrival": arrival,
                                 "finish": arrival + _visit_minutes(poi) / 60})
                prev = poi
            index = solution.Value(routing.NextVar(index))
        days.append({
            "day": v + 1,
            "date": None if start_date is None else start_date + datetime.timedelta(days=v),
            "route": route,
            "distance_km": round(km, 2),
            "schedule": schedule,
        })
    return days


# --------------------------------------------------------------------------
# Case setup: retrieval is run once per (city, archetype) and both arms are
# handed the same objects, so "same geography" is enforced by construction.
# --------------------------------------------------------------------------

def build_pools(graph: GraphIndex, rag: FusionRAGAgent, city: str, archetype: str,
                top_k_total: int):
    top_k = max(MIN_RETRIEVED_POIS, top_k_total)
    out = rag.run(archetype_query(archetype), destination_id=city, archetype=archetype,
                  config="fusion", top_k=top_k, narrate=False)
    candidates = [graph.g.nodes[r["poi_id"]] | {"poi_id": r["poi_id"]}
                  for r in out["results"] if r.get("type") == "poi"]
    sightseeing = [p for p in candidates if p.get("category") != FOOD_CATEGORY]
    already = {p.get("poi_id"): p for p in candidates
               if p.get("category") == FOOD_CATEGORY}
    food = [already.get(p.get("poi_id"), p)
            for p in graph.city_pois(city, category=FOOD_CATEGORY)]
    node = graph.g.nodes[city]
    hub = {"lat": node["lat"], "lon": node["lon"], "name": node["name"]}
    return sightseeing, food, hub


def archetype_preferences() -> dict[str, dict]:
    """The mean survey vector for each archetype, used as that archetype's
    representative traveler. It is fed to the score as a *vector*: nothing
    downstream reads the label it came from. The narrower claim -- that two
    travelers sharing a label now get different itineraries -- is not
    measurable from these means, and is checked separately by
    `same_label_travelers`."""
    survey = pd.read_csv(DATA_DIR / "user_survey.csv")
    return {a: survey[survey.archetype == a][PREFERENCE_DIMS].mean().to_dict()
            for a in survey.archetype.unique()}


def run_measurement() -> pd.DataFrame:
    graph = GraphIndex()
    rag = FusionRAGAgent()
    router = RouterAgent(graph)
    prefs_by_archetype = archetype_preferences()
    rows = []
    for top_k in TOP_K_TOTALS:
      for city in CITIES:
        for archetype in ARCHETYPES:
            sightseeing, food, hub = build_pools(graph, rag, city, archetype, top_k)
            day_start = start_hour_for(archetype)
            for hours in BUDGET_HOURS:
                budget = hours * 60
                common = dict(top_k=top_k, city=city, archetype=archetype,
                              budget_hours=hours,
                              n_sightseeing=len(sightseeing), n_food=len(food))

                prefs = prefs_by_archetype[archetype]
                # The shipped router: score-selected candidates, uniform TOPTW.
                shipped = router.run(city, sightseeing + food, n_days=N_DAYS,
                                     daily_minutes_budget=budget, archetype=archetype,
                                     narrate=False, start_date=START_DATE,
                                     preferences=prefs)["itinerary"]
                rows.append({**common, "arm": "shipped", "drop_penalty_m": None,
                             **measure(shipped, day_start, budget, prefs)})

                pool = list(sightseeing) + list(food)
                distance_fn, duration_fn, _ = _build_distance_functions(
                    [hub] + pool, use_real_routing=False, travel_mode=TRAVEL_MODE)
                scores = score_pois(pool, prefs)
                for penalty in DROP_PENALTIES_M:
                    for arm, kwargs in (
                            ("toptw", {}),
                            ("scored", {"scores": scores,
                                        "max_same_category": MAX_SAME_CATEGORY_PER_DAY})):
                        days = solve_toptw(pool, hub, N_DAYS, budget, day_start,
                                           START_DATE, distance_fn, duration_fn,
                                           penalty, **kwargs)
                        rows.append({**common, "arm": arm, "drop_penalty_m": penalty,
                                     **measure(days, day_start, budget, prefs)})
                print(f"  top_k={top_k} {city} / {archetype} / {hours}h done", flush=True)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per-day metrics are averaged over trips, but km/stop is recomputed from
    the totals rather than averaged over per-trip ratios -- averaging ratios
    would weight a 2-stop day the same as a 12-stop one."""
    grouped = df.groupby(["top_k", "arm", "drop_penalty_m"], dropna=False)
    out = grouped.agg(
        trips=("stops", "size"),
        stops=("stops", "sum"),
        sight_stops=("sight_stops", "sum"),
        food_stops=("food_stops", "sum"),
        km=("km", "sum"),
        days=("n_days", "sum"),
        closed_violations=("closed_violations", "sum"),
        budget_overruns=("budget_overruns", "sum"),
        days_with_meals=("days_with_meals", "sum"),
        mean_pref_match=("mean_pref_match", "mean"),
        mean_quality=("mean_quality", "mean"),
        categories_per_day=("categories_per_day", "mean"),
    ).reset_index()
    for col in ("mean_pref_match", "mean_quality", "categories_per_day"):
        out[col] = out[col].round(3)
    out["stops_per_day"] = (out["stops"] / out["days"]).round(3)
    out["sight_per_day"] = (out["sight_stops"] / out["days"]).round(3)
    out["km_per_stop"] = (out["km"] / out["stops"]).round(3)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(RESULTS_CSV))
    args = parser.parse_args()

    df = run_measurement()
    df.to_csv(args.out, index=False)
    summary = summarize(df)
    summary.to_csv(SUMMARY_CSV, index=False)

    print("\n=== TOPTW measurement (issue #72) ===")
    print(f"{len(CITIES)} cities x {len(ARCHETYPES)} archetypes x {BUDGET_HOURS} hours "
          f"x {N_DAYS} days = {len(df[df.arm == 'shipped']) // len(TOP_K_TOTALS) * N_DAYS} "
          f"days per pool size\n")
    for top_k in TOP_K_TOTALS:
        block = summary[summary.top_k == top_k]
        base = block[block.arm == "shipped"].iloc[0]
        print(f"--- retrieved pool: {top_k} POIs for the trip ---")
        print(f"{'arm':<10} {'P(m)':>6} {'stops/day':>10} {'':<9} {'km/stop':>9} {'':<9} "
              f"{'pref':>6} {'qual':>6} {'cat/day':>8} {'closed':>7} {'meal days':>10}")
        for _, r in block.iterrows():
            p = "-" if pd.isna(r.drop_penalty_m) else f"{int(r.drop_penalty_m)}"
            d_stops = "" if r.arm == "shipped" else f"({r.stops_per_day / base.stops_per_day - 1:+.1%})"
            d_km = "" if r.arm == "shipped" else f"({r.km_per_stop / base.km_per_stop - 1:+.1%})"
            print(f"{r.arm:<10} {p:>6} {r.stops_per_day:>10.2f} {d_stops:<9} "
                  f"{r.km_per_stop:>9.3f} {d_km:<9} {r.mean_pref_match:>6.3f} "
                  f"{r.mean_quality:>6.3f} {r.categories_per_day:>8.2f} "
                  f"{int(r.closed_violations):>7} {int(r.days_with_meals):>6}/{int(r.days)}")
        print()
    print(f"\nwrote {args.out}\nwrote {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
