"""Issue #77: the router reports stops, and a stop count has no ceiling.

`comparative_analysis.py` reports what retrieval collected *against what was
collectable* -- recall's structural maximum at `top_k=8` is 0.512, not 1.0,
and REPORT says so. The router reports stops and km/stop against nothing at
all, so "9 stops" cannot be read as good or bad. This script supplies the
missing denominator.

THE CEILING USED HERE is the first of the three the issue lists: the same
TOPTW model, the same pool, the same opening windows, category quota and
meal contract -- with distance and travel time set to zero. It answers "how
many stops would fit if getting between them were free", which is a real
upper bound and an optimistic one. It is a proxy, not the optimum: a tighter
ceiling (the same model given a much longer search) would sit lower, so the
true ratio is *higher* than what this prints.

WHAT IT MEASURED (48 configs: 2 cities x 4 archetypes x 3 day budgets x 2
pool sizes, 3-day trips):

    shipped 1130 stops / ceiling 1228 = 92.0%
    median 92.4%, range 80.8% - 100%, mean 2.04 stops missed per trip

    pool 24   93.2%      12h budget  90.6%      Family Traveler     89.3%
    pool 72   91.1%      15h budget  92.1%      Nightlife Seeker    91.4%
                         18h budget  93.2%      Culture Enthusiast  93.4%
                                                Nature & Adventure  94.1%

Two readings, and the second matters more than the first.

The narrow one: the router-side misreading is far smaller than the retrieval
one this issue was modelled on. Retrieval read a 0.512-capped number against
1.0; the router's stop count is already at 92% of an optimistic bound, so
reading it against nothing overstates the shortfall by about 8%. Reporting
the ratio is still worth doing -- two numbers on one screen should be read
the same way -- but it will not reveal a large hidden loss, because there
isn't one.

The broader one: **the binding constraint is the candidate pool, not the
solver.** Nine of the ten configs that sit exactly at the ceiling are at
`top_k=24`, and the ratio drops as the pool grows (93.2% -> 91.1%). Where
the solver takes everything offered, the stops that never appear were never
candidates. That is the same conclusion `toptw.py` reached from the other
direction ("the candidate pool rather than the penalty is the binding
constraint") and it points at `RETRIEVED_POIS_PER_DAY`, which REPORT section
5 flags as a knob nobody has deliberately chosen.

WHAT THIS DOES NOT SEPARATE. The ceiling arm solves over the whole pool; the
shipped arm solves the shortlist `RouterAgent` scores its way down to. So the
8% covers the shortlist's cost as well as the geometry's, and this script
cannot say how the two divide. Splitting them needs a third arm -- the same
shortlist, zero distance -- which is worth adding before anyone reads the
ratio as solver inefficiency.

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
from roamwise.agents.router_agent import RouterAgent, start_hour_for
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

                    rows.append({
                        "top_k": top_k, "city": city, "archetype": archetype,
                        "budget_hours": hours, "pool": len(pool),
                        "shipped_stops": shipped_m["stops"],
                        "ceiling_stops": ceiling_m["stops"],
                        "ratio": (shipped_m["stops"] / ceiling_m["stops"]
                                  if ceiling_m["stops"] else None),
                        "shipped_km": round(shipped_m["km"], 2),
                    })
                    r = rows[-1]
                    print(f"  top_k={top_k} {city} {archetype[:18]:18s} {hours}h  "
                          f"shipped={r['shipped_stops']:>3d} "
                          f"ceiling={r['ceiling_stops']:>3d} "
                          f"ratio={r['ratio']:.1%}", flush=True)
    return pd.DataFrame(rows)


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

    total = df.shipped_stops.sum() / df.ceiling_stops.sum()
    print(f"\nshipped {df.shipped_stops.sum()} stops / "
          f"ceiling {df.ceiling_stops.sum()} = {total:.1%}")
    print(f"median {df.ratio.median():.1%}, "
          f"range {df.ratio.min():.1%} - {df.ratio.max():.1%}, "
          f"{(df.ceiling_stops - df.shipped_stops).mean():.2f} stops missed per trip")
    print(f"at the ceiling: {(df.ratio >= 0.999).sum()} of {len(df)} configs, "
          f"{((df.ratio >= 0.999) & (df.top_k == TOP_K_TOTALS[0])).sum()} of them "
          f"at top_k={TOP_K_TOTALS[0]}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
