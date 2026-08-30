"""
Do a city's best-known places reach the plan? (issue #122)

A Paris itinerary without the Eiffel Tower is not defensible, and #122 measured
that it never contains one. This script is that measurement, made reproducible
and re-runnable, because the issue's own numbers were taken *before* #126 and
its comment thread says plainly what that costs:

    "Sonra kapanirsa: 0/7, 17/72 gibi butun sayilar yeniden alinmali."

#126 landed first. The chain retriever is anchored at the city centre and fires
on every plan, so the candidate pool the router is handed is ordered differently
now than when the issue's table was written. The diagnosis it argues for --
that the loss is in the router, not in retrieval -- is unaffected: what changed
is the order of the pool, not who drops the Eiffel Tower out of it.

What is reported, per (city, archetype):

  in_pool       the POI is among the `top_k` retrieval hands the router
  in_shortlist  it survives `select_by_score`, which is where the traveler's
                sliders reach the itinerary (issue #80)
  in_plan       it is actually scheduled

Three columns rather than one because they localise the loss. A POI that is not
in the pool is a retrieval finding; one that is in the pool and not the
shortlist is a scoring finding; one that reaches the shortlist and not the plan
is the solver's doing, which is what #122 claims and what a per-POI drop penalty
would address.

Stops per day and km per stop travel alongside, because they are what any fix
here has to not spend: #33 measured the band a crowding-aware change was allowed
to cost (1.3% of stops, no measurable distance), and #122's acceptance criteria
reuse it.

Usage:

    python -m roamwise.evaluation.iconic_coverage            # measure and print
    python -m roamwise.evaluation.iconic_coverage --write    # also write the CSV
    python -m roamwise.evaluation.iconic_coverage --top-n 20
    python -m roamwise.evaluation.iconic_coverage --threshold 0.99 --multiplier 1.5
"""
import argparse
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd

from roamwise.agents.orchestrator import RETRIEVED_POIS_PER_DAY
from roamwise.agents.router_agent import MIN_FOOD_PER_DAY, RouterAgent
from roamwise.evaluation.toptw_measurement import (N_DAYS, START_DATE,
                                                   archetype_preferences, measure)
from roamwise.knowledge_graph.build_graph import GraphIndex
from roamwise.optimization import toptw
from roamwise.optimization.routing import FOOD_CATEGORY
from roamwise.retrieval.fusion import FusionRetriever
from roamwise.retrieval.query import archetype_query

HERE = Path(__file__).parent
DATA_DIR = HERE.parent / "data"
RESULTS_CSV = HERE / "iconic_coverage.csv"

# How many of a city's best-known POIs to follow. 12 is the issue's own figure,
# so its table and this one are about the same set.
DEFAULT_TOP_N = 12
BUDGET_HOURS = 12


def iconic_pois(top_n: int = DEFAULT_TOP_N) -> pd.DataFrame:
    """Each city's `top_n` best-known POIs, by `popularity_score`.

    Read from the catalogue rather than named in a literal, for the reason
    CLAUDE.md gives: the catalogue has changed size twice, and a hardcoded
    "Eiffel Tower" would turn this measurement red for a catalogue edit rather
    than for a routing change. `popularity_score` is a within-city percentile
    of Wikidata sitelinks blended with Wikipedia pageviews (#63), which is the
    same signal the retrieval layer's prominence half reads -- so "best known"
    means here what it means there.
    """
    catalogue = pd.read_csv(DATA_DIR / "poi.csv")
    return (catalogue.sort_values("popularity_score", ascending=False)
            .groupby("destination_id", sort=True)
            .head(top_n)[["destination_id", "poi_id", "name", "category",
                          "popularity_score"]]
            .reset_index(drop=True))


class Harness:
    """The graph, router and retriever, built once.

    The sweep runs this grid a dozen times over; the objects are safe to share
    (the graph and the corpus embeddings are cached per process anyway) and
    rebuilding a `FusionRetriever` per cell would put a model load inside the
    measurement loop.
    """

    def __init__(self):
        self.graph = GraphIndex()
        self.router = RouterAgent(self.graph)
        self.retriever = FusionRetriever()
        self.preferences = archetype_preferences()


def coverage_rows(top_n: int = DEFAULT_TOP_N,
                  harness: "Harness" = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per (city, archetype, iconic POI), plus one row per cell.

    The shortlist is taken from `RouterAgent._select` rather than recomputed
    here. It is a private method and calling it is deliberate: a second copy of
    the food/sight split and the two limits would drift from the router's own,
    and then this script would be measuring something the app does not do.
    """
    iconic = iconic_pois(top_n)
    harness = harness or Harness()
    graph, router = harness.graph, harness.router
    retriever, preferences = harness.retriever, harness.preferences

    poi_rows, cell_rows = [], []
    for city, city_iconic in iconic.groupby("destination_id"):
        for archetype in sorted(preferences):
            prefs = preferences[archetype]
            results = retriever.retrieve(archetype_query(archetype), config="fusion",
                                         destination_id=city, archetype=archetype,
                                         top_k=N_DAYS * RETRIEVED_POIS_PER_DAY,
                                         start_date=START_DATE)
            pool = [graph.g.nodes[r["poi_id"]] | {"poi_id": r["poi_id"]}
                    for r in results if r.get("type") == "poi"]

            sights = [p for p in pool if p.get("category") != FOOD_CATEGORY]
            food = router._food_pois(city, pool)
            shortlist_sights, shortlist_food = router._select(
                sights, food, N_DAYS, prefs, MIN_FOOD_PER_DAY)
            # The working set the solver actually sees, guaranteed landmarks
            # included (#122 item 4) -- otherwise this column would describe a
            # shortlist the router stopped using.
            shortlist_sights = router._with_iconic(city, shortlist_sights)

            routing = router.run(city, pool, n_days=N_DAYS,
                                 daily_minutes_budget=BUDGET_HOURS * 60,
                                 archetype=archetype, narrate=False,
                                 start_date=START_DATE, preferences=prefs)
            metrics = measure(routing["itinerary"], routing["day_start_hour"],
                              BUDGET_HOURS * 60, preferences=prefs)

            # "Reachable" for the solver: what retrieval surfaced, plus the
            # landmarks the router guarantees whatever the archetype asked for.
            in_pool = ({p["poi_id"] for p in pool}
                       | {p["poi_id"] for p in router._iconic_pois(city)})
            in_shortlist = {p["poi_id"] for p in shortlist_sights + shortlist_food}
            in_plan = {p["poi_id"] for day in routing["itinerary"] for p in day["route"]}

            for row in city_iconic.itertuples():
                poi_rows.append({
                    "city": city, "archetype": archetype, "poi_id": row.poi_id,
                    "name": row.name, "category": row.category,
                    "popularity_score": round(row.popularity_score, 4),
                    "in_pool": row.poi_id in in_pool,
                    "in_shortlist": row.poi_id in in_shortlist,
                    "in_plan": row.poi_id in in_plan,
                })
            cell_rows.append({
                "city": city, "archetype": archetype,
                "iconic_in_pool": sum(r.poi_id in in_pool for r in city_iconic.itertuples()),
                "iconic_in_plan": sum(r.poi_id in in_plan for r in city_iconic.itertuples()),
                "iconic_total": len(city_iconic),
                "stops_per_day": metrics["stops_per_day"],
                "km_per_stop": metrics["km_per_stop"],
                "categories_per_day": metrics["categories_per_day"],
                "mean_quality": metrics["mean_quality"],
                "mean_pref_match": metrics["mean_pref_match"],
            })
    return pd.DataFrame(poi_rows), pd.DataFrame(cell_rows)


def report(poi_df: pd.DataFrame, cell_df: pd.DataFrame) -> None:
    n_archetypes = poi_df.archetype.nunique()
    print(f"\n{N_DAYS}-day trips, {BUDGET_HOURS}h days, {n_archetypes} archetypes, "
          f"pool = {N_DAYS * RETRIEVED_POIS_PER_DAY}, start {START_DATE}\n")

    print("Where each best-known POI is lost (x/{} archetypes):".format(n_archetypes))
    per_poi = (poi_df.groupby(["city", "name"], sort=False)
               [["in_pool", "in_shortlist", "in_plan"]].sum())
    for city, block in per_poi.groupby(level=0):
        print(f"\n  {city}")
        print(f"    {'POI':<38} {'pool':>5} {'short':>6} {'plan':>5}")
        for (_, name), row in block.iterrows():
            print(f"    {name[:38]:<38} {row.in_pool:>5} {row.in_shortlist:>6} "
                  f"{row.in_plan:>5}")

    print("\nPer cell:")
    print(f"  {'city':<5} {'archetype':<20} {'iconic in plan':>15} {'stops/day':>10} "
          f"{'km/stop':>8}")
    for row in cell_df.itertuples():
        print(f"  {row.city:<5} {row.archetype:<20} "
              f"{row.iconic_in_plan:>7}/{row.iconic_total:<7} "
              f"{row.stops_per_day:>10.2f} {row.km_per_stop:>8.3f}")

    print(f"\nTotals: {int(cell_df.iconic_in_plan.sum())} of "
          f"{int(cell_df.iconic_total.sum())} (city x archetype x POI) "
          f"iconic slots filled; stops/day {cell_df.stops_per_day.mean():.3f}, "
          f"km/stop {cell_df.km_per_stop.mean():.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N,
                        help="how many best-known POIs per city to follow")
    parser.add_argument("--threshold", type=float,
                        help="override toptw.ICONIC_QUALITY_THRESHOLD for this run")
    parser.add_argument("--multiplier", type=float,
                        help="override toptw.ICONIC_DROP_MULTIPLIER for this run")
    parser.add_argument("--write", action="store_true",
                        help=f"write {RESULTS_CSV.name}")
    args = parser.parse_args()

    # Set as module constants for the same reason the sweep does it: the whole
    # app path -- retrieval, shortlist, solver -- then runs under the setting,
    # so the table describes what a traveler would get.
    if args.threshold is not None:
        toptw.ICONIC_QUALITY_THRESHOLD = args.threshold
    if args.multiplier is not None:
        toptw.ICONIC_DROP_MULTIPLIER = args.multiplier
    print(f"threshold={toptw.ICONIC_QUALITY_THRESHOLD} "
          f"multiplier={toptw.ICONIC_DROP_MULTIPLIER}")

    poi_df, cell_df = coverage_rows(args.top_n)
    report(poi_df, cell_df)
    if args.write:
        poi_df.to_csv(RESULTS_CSV, index=False)
        print(f"\nWrote {RESULTS_CSV}")


if __name__ == "__main__":
    main()
