"""Does the quota exponent reach the traveler's plan? (issue #143)

`retrieval_coverage.py` counts *reachable* POIs -- the union over archetypes
and cities of everything retrieval could surface. #123 lost 5 of them at the
one-day pool and the acceptance criterion failed on that number alone, which
raises the question #143 asks first: is a POI at the bottom of a 24-place pool
a stop a traveler would have had, or a tail the router never selects?

The two are not the same measure. Reachability is a union over seven
archetypes; a single traveler sees one pool, and the router then drops most of
it -- a one-day plan holds eight to eleven stops out of twenty-four candidates.
A category can lose its last reachable POI without any itinerary changing, and
it can also lose the one place that was holding a day together.

So this measures the plans themselves, at both exponents, over the same grid
`toptw_measurement.py` uses: 2 cities x 7 archetypes x 3 trip lengths, each
archetype starting at its own hour. It reports stops, active minutes, distance
per stop and how many distinct categories a trip contains -- the last because
the quota's whole purpose is spreading the pool across categories, so a change
that moves nothing else should still show there.

Run:  python -m roamwise.evaluation.quota_plan_impact
Writes `quota_plan_impact.csv`. Nothing here mutates the shipped constant; the
exponent is patched per arm and restored.
"""
import datetime
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd

from roamwise.agents.fusion_rag_agent import FusionRAGAgent
from roamwise.agents.router_agent import RouterAgent, start_hour_for
from roamwise.knowledge_graph import build_graph
from roamwise.knowledge_graph.build_graph import CATEGORY_AFFINITY, GraphIndex
from roamwise.evaluation.toptw_measurement import archetype_preferences
from roamwise.retrieval.query import archetype_query

HERE = Path(__file__).parent
IMPACT_CSV = HERE / "quota_plan_impact.csv"

CITIES = ["PAR", "BER"]
TRIP_LENGTHS = [1, 3, 5]
RETRIEVED_POIS_PER_DAY = 24
BUDGET_HOURS = 12
# A Tuesday, so no venue is excluded by a Monday closure -- the plan differences
# this measures should come from the pool, not from the calendar (#70).
START_DATE = datetime.date(2026, 9, 29)

# The shipped value and the candidate the sweep found Pareto-dominant.
EXPONENTS = [0.5, 0.6]


def _plan(router: RouterAgent, graph: GraphIndex, rag: FusionRAGAgent,
          city: str, archetype: str, n_days: int, prefs: dict) -> dict:
    top_k = n_days * RETRIEVED_POIS_PER_DAY
    out = rag.run(archetype_query(archetype), destination_id=city, archetype=archetype,
                  config="fusion", top_k=top_k, narrate=False)
    pool = [graph.g.nodes[r["poi_id"]] | {"poi_id": r["poi_id"]}
            for r in out["results"] if r.get("type") == "poi"]
    itinerary = router.run(city, pool, n_days=n_days, daily_minutes_budget=BUDGET_HOURS * 60,
                           archetype=archetype, narrate=False, start_date=START_DATE,
                           day_start_hour=start_hour_for(archetype),
                           preferences=prefs)["itinerary"]

    stops = [stop for day in itinerary for stop in day["route"]]
    categories = {s.get("category") for s in stops if s.get("category")}
    km = sum(day["distance_km"] for day in itinerary)
    return {
        "pool": len(pool),
        "stops": len(stops),
        "categories": len(categories),
        "km_per_stop": round(km / len(stops), 3) if stops else None,
        "active_minutes": sum(day.get("active_minutes", 0) for day in itinerary),
    }


def main():
    graph = GraphIndex()
    rag = FusionRAGAgent()
    router = RouterAgent(graph)
    prefs_by_archetype = archetype_preferences()
    shipped = build_graph.PREFERENCE_QUOTA_EXPONENT

    rows = []
    try:
        for exponent in EXPONENTS:
            build_graph.PREFERENCE_QUOTA_EXPONENT = exponent
            for n_days in TRIP_LENGTHS:
                for city in CITIES:
                    for archetype in sorted(CATEGORY_AFFINITY):
                        measured = _plan(router, graph, rag, city, archetype, n_days,
                                         prefs_by_archetype[archetype])
                        rows.append({"exponent": exponent, "n_days": n_days,
                                     "city": city, "archetype": archetype, **measured})
                print(f"exponent {exponent} | {n_days}d done", flush=True)
    finally:
        build_graph.PREFERENCE_QUOTA_EXPONENT = shipped

    df = pd.DataFrame(rows)
    df.to_csv(IMPACT_CSV, index=False)
    print(f"\nWrote {IMPACT_CSV.relative_to(_REPO_ROOT)}\n")

    summary = df.groupby(["n_days", "exponent"]).agg(
        stops=("stops", "mean"), categories=("categories", "mean"),
        km_per_stop=("km_per_stop", "mean"), pool=("pool", "mean")).round(3)
    print("Mean over 2 cities x 7 archetypes:")
    print(summary.to_string())

    print("\nPer trip length, 0.6 minus 0.5:")
    for n_days in TRIP_LENGTHS:
        a = df[(df.n_days == n_days) & (df.exponent == 0.5)].set_index(["city", "archetype"])
        b = df[(df.n_days == n_days) & (df.exponent == 0.6)].set_index(["city", "archetype"])
        changed = (b.stops - a.stops)
        print(f"  {n_days}d: stops {changed.sum():+d} across {(changed != 0).sum()} "
              f"of {len(changed)} trips; categories "
              f"{(b.categories - a.categories).sum():+d}")


if __name__ == "__main__":
    main()
