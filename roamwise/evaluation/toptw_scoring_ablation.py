"""Where the score from issue #72 actually belongs: picking the candidates,
or weighting the solver?

`toptw_measurement.py` answered the issue's first question (a real TOPTW
solver buys stops *and* coherence at once). Adding the score function on top
of it then made both worse, which is the opposite of what a personalization
term is supposed to do. This script isolates why, by crossing the two places
the score can act:

  pool     retrieval    -- today's path: `archetype_query` turns the archetype
                          label into a search string and retrieval returns the
                          candidates.
           score        -- the catalogue ranked by score_pois directly, same
                          number of candidates, no archetype label involved.
  weights  uniform      -- every stop costs the same to skip.
           scored       -- skipping a stop costs what it is worth to this
                          traveler (plus the per-day category cap).

Read `mean_pref_match` knowing it is the scored arm's own objective, not an
independent judge. That makes a small improvement there damning rather than
reassuring: a term that barely moves the quantity it is directly maximising,
while giving up stops and distance, is not paying for itself.

Run:  python evaluation/toptw_scoring_ablation.py
"""
import datetime
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd

from roamwise.agents.fusion_rag_agent import FusionRAGAgent
from roamwise.agents.router_agent import start_hour_for
from roamwise.knowledge_graph.build_graph import DATA_DIR, GraphIndex
from roamwise.optimization.routing import FOOD_CATEGORY, _build_distance_functions
from roamwise.optimization.scoring import PREFERENCE_DIMS, score_pois
from roamwise.evaluation.toptw_measurement import (
    ARCHETYPES, CITIES, MAX_SAME_CATEGORY_PER_DAY, N_DAYS, START_DATE,
    TRAVEL_MODE, build_pools, measure, solve_toptw)

HERE = Path(__file__).parent
RESULTS_CSV = HERE / "toptw_scoring_ablation.csv"
TOP_K = 72
BUDGET_HOURS = 12
DROP_PENALTY_M = 8000


def run() -> pd.DataFrame:
    graph, rag = GraphIndex(), FusionRAGAgent()
    survey = pd.read_csv(DATA_DIR / "user_survey.csv")
    rows = []
    for city in CITIES:
        node = graph.g.nodes[city]
        hub = {"lat": node["lat"], "lon": node["lon"], "name": node["name"]}
        catalogue = graph.city_pois(city)
        all_food = [p for p in catalogue if p.get("category") == FOOD_CATEGORY]
        all_sights = [p for p in catalogue if p.get("category") != FOOD_CATEGORY]
        for archetype in ARCHETYPES:
            prefs = survey[survey.archetype == archetype][PREFERENCE_DIMS].mean().to_dict()
            day_start = start_hour_for(archetype)
            retrieved, food, _ = build_pools(graph, rag, city, archetype, TOP_K)
            # Same candidate count, so the two pools differ in *what* they
            # hold, never in how much -- otherwise a bigger pool would win on
            # size alone and say nothing about the selector.
            ranked = [p for _, p in sorted(zip(score_pois(all_sights, prefs), all_sights),
                                           key=lambda t: -t[0])][:len(retrieved)]
            for pool_name, sights, meal_pool in (("retrieval", retrieved, food),
                                                  ("score", ranked, all_food)):
                pool = list(sights) + list(meal_pool)
                distance_fn, duration_fn, _, _ = _build_distance_functions(
                    [hub] + pool, use_real_routing=False, travel_mode=TRAVEL_MODE)
                scores = score_pois(pool, prefs)
                for weights, kwargs in (("uniform", {}),
                                        ("scored", {"scores": scores,
                                                    "max_same_category": MAX_SAME_CATEGORY_PER_DAY})):
                    days = solve_toptw(pool, hub, N_DAYS, BUDGET_HOURS * 60, day_start,
                                       START_DATE, distance_fn, duration_fn,
                                       DROP_PENALTY_M, **kwargs)
                    rows.append({"city": city, "archetype": archetype, "pool": pool_name,
                                 "weights": weights,
                                 **measure(days, day_start, BUDGET_HOURS * 60, prefs)})
        print(f"  {city} done", flush=True)
    return pd.DataFrame(rows)


def main():
    df = run()
    df.to_csv(RESULTS_CSV, index=False)
    g = df.groupby(["pool", "weights"]).agg(
        stops=("stops", "sum"), km=("km", "sum"), days=("n_days", "sum"),
        pref=("mean_pref_match", "mean"), qual=("mean_quality", "mean"),
        cats=("categories_per_day", "mean")).reset_index()
    g["stops_per_day"] = (g.stops / g.days).round(2)
    g["km_per_stop"] = (g.km / g.stops).round(3)
    print(f"\n=== where the score belongs (issue #72) ==="
          f"\n{len(CITIES)} cities x {len(ARCHETYPES)} archetypes, "
          f"{BUDGET_HOURS}h days, P={DROP_PENALTY_M}m\n")
    print(f"{'pool':<11} {'weights':<9} {'stops/day':>10} {'km/stop':>9} "
          f"{'pref':>7} {'qual':>7} {'cat/day':>8}")
    for _, r in g.iterrows():
        print(f"{r['pool']:<11} {r.weights:<9} {r.stops_per_day:>10.2f} "
              f"{r.km_per_stop:>9.3f} {r.pref:>7.3f} {r.qual:>7.3f} {r.cats:>8.2f}")
    print(f"\nwrote {RESULTS_CSV}")


if __name__ == "__main__":
    main()
