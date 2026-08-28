"""Issue #77: the router reports stops, and a stop count has no ceiling.

`comparative_analysis.py` reports what retrieval collected *against what was
collectable* -- recall's structural maximum at `top_k=8` is 0.512, not 1.0,
and REPORT says so. The router reports stops and km/stop against nothing at
all, so "9 stops" cannot be read as good or bad. This script supplies the
missing denominator.

THE CEILING USED HERE is the first of the three the issue lists: the same
TOPTW model, the same pool, the same opening windows, category quota and
meal contract -- with distance and travel time set to zero. It answers "how
many stops would fit if getting between them were free", which is the
optimistic direction: a tighter ceiling (the same model given a much longer
search) would sit lower, so the true ratio is *higher* than what this prints.

It is a proxy in a second sense too, and the third arm below makes that
visible rather than leaving it as a caveat: the relaxed arm is itself a
heuristic solve under the same fixed SOLUTION_LIMIT, so it is not an exact
bound and can lose a stop or two on a harder search. Read the ratio to the
nearest percent, not the nearest stop.

WHAT IT MEASURED (48 configs: 2 cities x 4 archetypes x 3 day budgets x 2
pool sizes, 3-day trips):

    shipped 1178 stops / ceiling 1308 = 90.1%
    median 90.9%, range 77.8% - 100%, mean 2.71 stops missed per trip

    pool 24   93.0%       at the ceiling: 9 of 48 configs, all 9 at top_k=24
    pool 72   88.0%

THE THIRD ARM, and why the split it was added for does not exist. The issue
asked to separate the shortlist's cost from the geometry's, because the two
arms above compare a trip solved from `RouterAgent`'s shortlist against a
ceiling solved over the whole pool. So a third arm solves the *same
shortlist* with travel free, and the difference between the two ceilings
should be what the shortlist costs.

It is nothing. The two ceilings are identical in 35 of 48 configs, and in
the other 13 they differ by at most two stops in *both* directions -- the
shortlist ceiling comes out higher in 7 of them. That direction is
impossible for a true bound: the shortlist is a subset of the pool, so an
exact solve over the pool can never do worse. Both arms are heuristic solves
under the same fixed SOLUTION_LIMIT, and the larger pool is the harder
search, so the pool arm drops stops the shortlist arm keeps. The wobble is
this measurement's resolution, and the shortlist's cost is smaller than it:
-4 stops across 48 configs.

The conclusion is therefore stronger than a split would have been.
Effectively all of the ~10% gap is the geometry -- what it costs to get
between the stops. The shortlist, which is where the traveler's sliders
reach the plan and the only place they do (issue #80), costs no measurable
stops at all. Personalization is not being paid for in trip length.

TWO OLDER READINGS, both still standing. The narrow one: the router-side
misreading is far smaller than the retrieval one this issue was modelled on.
Retrieval read a 0.512-capped number against 1.0; the router's stop count is
already at 90% of an optimistic bound, so reading it against nothing
overstates the shortfall by about 10%. Reporting the ratio is still worth
doing -- two numbers on one screen should be read the same way -- but it
does not reveal a large hidden loss, because there isn't one.

The broader one: **the binding constraint is the candidate pool, not the
solver.** All nine configs that sit exactly at the ceiling are at
`top_k=24`, and the ratio drops as the pool grows (93.0% -> 88.0%). Where
the solver takes everything offered, the stops that never appear were never
candidates. That is not an argument for a bigger pool: issue #80 measured
stops flat as the pool grows from 40 to 371.

Run:  python evaluation/toptw_ceiling.py
"""
import argparse
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd

from roamwise.agents.fusion_rag_agent import FusionRAGAgent
from roamwise.agents.router_agent import (
    MIN_FOOD_PER_DAY, RouterAgent, start_hour_for)
from roamwise.knowledge_graph.build_graph import GraphIndex
from roamwise.optimization.toptw import DEFAULT_DROP_PENALTY_M
from roamwise.evaluation.toptw_measurement import (
    ARCHETYPES, BUDGET_HOURS, CITIES, MAX_SAME_CATEGORY_PER_DAY, N_DAYS,
    START_DATE, TOP_K_TOTALS, archetype_preferences, build_pools, measure,
    solve_toptw)

HERE = Path(__file__).parent
RESULTS_CSV = HERE / "toptw_ceiling.csv"


def _free_travel():
    """Distance and duration callbacks that cost nothing.

    Relaxing both is deliberate. Zeroing only the arc cost would still spend
    the day's minutes on travel, and the ceiling would then answer "how many
    stops fit if walking were free but still took time", which is not a
    quantity anyone wants. With both at zero the day's budget buys visits and
    nothing else, and the only constraints left are the ones the issue names:
    the clock and the opening hours."""
    return (lambda a, b: 0.0), (lambda a, b: 0.0)


def run_ceiling(cities=None, top_ks=None) -> pd.DataFrame:
    graph = GraphIndex()
    rag = FusionRAGAgent()
    router = RouterAgent(graph)
    prefs_by_archetype = archetype_preferences()

    rows = []
    for top_k in (top_ks or TOP_K_TOTALS):
        for city in (cities or CITIES):
            for archetype in ARCHETYPES:
                sightseeing, food, hub = build_pools(graph, rag, city, archetype, top_k)
                day_start = start_hour_for(archetype)
                prefs = prefs_by_archetype[archetype]
                pool = list(sightseeing) + list(food)

                for hours in BUDGET_HOURS:
                    budget = hours * 60

                    shipped = router.run(
                        city, sightseeing + food, n_days=N_DAYS,
                        daily_minutes_budget=budget, archetype=archetype,
                        narrate=False, start_date=START_DATE,
                        preferences=prefs)["itinerary"]
                    shipped_m = measure(shipped, day_start, budget, prefs)

                    free_distance, free_duration = _free_travel()
                    ceiling = solve_toptw(
                        pool, hub, N_DAYS, budget, day_start, START_DATE,
                        free_distance, free_duration, DEFAULT_DROP_PENALTY_M,
                        max_same_category=MAX_SAME_CATEGORY_PER_DAY)
                    ceiling_m = measure(ceiling, day_start, budget, prefs)

                    # The third arm: the same shortlist the shipped trip was
                    # solved from, with travel free. Without it the 8% gap is
                    # one number covering two different things, and the issue
                    # asks for them apart before anyone reads the ratio as
                    # solver inefficiency. Built through the router's own
                    # `_select` rather than a copy of its rules, so the arm
                    # cannot drift from what `run()` actually shortlists.
                    short_sights, short_food = router._select(
                        sightseeing, router._food_pois(city, sightseeing + food),
                        N_DAYS, prefs, MIN_FOOD_PER_DAY)
                    shortlist = list(short_sights) + list(short_food)
                    short_ceiling = solve_toptw(
                        shortlist, hub, N_DAYS, budget, day_start, START_DATE,
                        free_distance, free_duration, DEFAULT_DROP_PENALTY_M,
                        max_same_category=MAX_SAME_CATEGORY_PER_DAY)
                    short_ceiling_m = measure(short_ceiling, day_start, budget, prefs)

                    rows.append({
                        "top_k": top_k, "city": city, "archetype": archetype,
                        "budget_hours": hours, "pool": len(pool),
                        "shortlist": len(shortlist),
                        "shipped_stops": shipped_m["stops"],
                        "shortlist_ceiling_stops": short_ceiling_m["stops"],
                        "ceiling_stops": ceiling_m["stops"],
                        "ratio": (shipped_m["stops"] / ceiling_m["stops"]
                                  if ceiling_m["stops"] else None),
                        "ratio_vs_shortlist": (
                            shipped_m["stops"] / short_ceiling_m["stops"]
                            if short_ceiling_m["stops"] else None),
                        "shipped_km": round(shipped_m["km"], 2),
                    })
                    r = rows[-1]
                    print(f"  top_k={top_k} {city} {archetype[:18]:18s} {hours}h  "
                          f"shipped={r['shipped_stops']:>3d} "
                          f"shortlist_ceiling={r['shortlist_ceiling_stops']:>3d} "
                          f"ceiling={r['ceiling_stops']:>3d} "
                          f"ratio={r['ratio']:.1%} "
                          f"(vs shortlist {r['ratio_vs_shortlist']:.1%})", flush=True)
    return pd.DataFrame(rows)


def _report_split(df: pd.DataFrame) -> None:
    """How much of the gap is the shortlist's and how much is the geometry's.

    The honest answer is that the shortlist's share is not measurable here,
    and the evidence for that is a sign error the data itself commits: the
    shortlist is a subset of the pool, so a *true* ceiling over the pool can
    never sit below one over the shortlist -- yet it does, in some configs.
    Both arms are heuristic solves under the same fixed SOLUTION_LIMIT, and
    the larger pool is the harder search, so the pool arm loses stops the
    shortlist arm keeps. That wobble is the resolution of this measurement,
    and the shortlist's cost is smaller than it.

    So the number NOT printed here is a percentage split. Printing "the
    shortlist is -3% of the gap" would dress noise as a finding."""
    delta = df.shortlist_ceiling_stops - df.ceiling_stops
    identical, above, below = (delta == 0).sum(), (delta > 0).sum(), (delta < 0).sum()
    print(f"\nshortlist ceiling vs pool ceiling: identical in {identical} of {len(df)} "
          f"configs, higher in {above}, lower in {below}, never by more than "
          f"{int(delta.abs().max())} stops")
    print(f"  a subset cannot beat its superset under an exact bound, so the "
          f"{above} configs above are the relaxed solve showing its own heuristic "
          f"limit -- that spread is this measurement's resolution")
    print(f"  the shortlist's cost in stops is below it: "
          f"{int(df.ceiling_stops.sum() - df.shortlist_ceiling_stops.sum()):+d} stops over "
          f"{len(df)} configs. Effectively all of the gap is the geometry.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cities", default=None,
                        help="comma-separated subset, e.g. PAR")
    parser.add_argument("--top-k", type=int, default=None,
                        help="a single pool size instead of both")
    parser.add_argument("--out", default=str(RESULTS_CSV))
    args = parser.parse_args()

    df = run_ceiling(
        cities=args.cities.split(",") if args.cities else None,
        top_ks=[args.top_k] if args.top_k else None)
    df.to_csv(args.out, index=False)

    shipped, short_ceiling = df.shipped_stops.sum(), df.shortlist_ceiling_stops.sum()
    ceiling = df.ceiling_stops.sum()
    total = shipped / ceiling
    print(f"\nshipped {shipped} stops / ceiling {ceiling} = {total:.1%}")
    # The split the issue asks for: how much of the gap is the shortlist's and
    # how much is the geometry's. The first is the price of personalization
    # (`select_by_score` is the only place the sliders reach the itinerary),
    # the second is what a better solver could win back.
    print(f"against the shortlist alone: {shipped}/{short_ceiling} = "
          f"{shipped / short_ceiling:.1%}")
    # The split the issue asks for -- except the answer is that there is
    # nothing to split. See _report_split.
    _report_split(df)
    print(f"median {df.ratio.median():.1%}, "
          f"range {df.ratio.min():.1%} - {df.ratio.max():.1%}, "
          f"{(df.ceiling_stops - df.shipped_stops).mean():.2f} stops missed per trip")
    print(f"at the ceiling: {(df.ratio >= 0.999).sum()} of {len(df)} configs, "
          f"{((df.ratio >= 0.999) & (df.top_k == TOP_K_TOTALS[0])).sum()} of them "
          f"at top_k={TOP_K_TOTALS[0]}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
