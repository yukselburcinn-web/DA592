"""Does letting the router choose the *hour* put people in emptier rooms?

Issue #33's remaining half. The crowding factor that #72 shipped is a static
node weight: it knows how busy a POI typically is and picks quieter ones, but
which hour a stop is visited was the solver's decision and nothing priced it.
Measured before this change, over 2 cities x 4 archetypes x 3 day lengths,
stops landed at **56.9% busy against the 47.9% their own typical level would
predict** -- the router was systematically putting people in at the busy hour,
because the hours a day naturally fills are the hours everyone else fills too.

`toptw.solve(hour_aware=True)` offers each measured POI a second node pinned to
its own quietest three hours, priced below the day-wide one. This script asks
whether that offer is taken, and what it costs when it is.

THE METRIC. `exposure` is the busyness at the hour a stop is actually arrived
at; `typical` is the same stops' own average level. Reporting both is what
separates the two halves of the factor, and only the second is on trial here:

  typical falls        -> the itinerary went to quieter *places*
  exposure - typical   -> the itinerary went at quieter *hours*

The gap is the number this change exists to close. A run that lowers exposure
only by lowering `typical` has not done what the issue asked; it has just
leaned harder on the static discount that already shipped.

Both are averaged over the stops that carry an hourly reading -- 41% of the
catalogue (269 of 654 POIs). `known_stops` reports how many that was, because
an arm that visits fewer measured POIs is being scored on a different sample
and its numbers are not comparable to the rest.

Run:  python evaluation/crowding_hour_measurement.py
"""
import datetime
import sys
import time
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd

from roamwise.agents.fusion_rag_agent import FusionRAGAgent
from roamwise.agents.router_agent import RouterAgent, start_hour_for
from roamwise.evaluation.toptw_measurement import (
    ARCHETYPES, BUDGET_HOURS, CITIES, N_DAYS, START_DATE, archetype_preferences,
    build_pools)
from roamwise.knowledge_graph.build_graph import GraphIndex
from roamwise.optimization import toptw
from roamwise.optimization.scoring import busyness

HERE = Path(__file__).parent
RESULTS_CSV = HERE / "crowding_hour_results.csv"
SUMMARY_CSV = HERE / "crowding_hour_summary.csv"

# The pool the app actually asks for: RETRIEVED_POIS_PER_DAY = 8 over three
# days. The larger 72-POI condition `toptw_measurement.py` also runs is left
# out here on purpose -- this change is about the hour a stop is given, and a
# pool with three times as many candidates answers a question about selection.
TOP_K_TOTAL = 24

# What a busy hour costs, in metres of walking. 0 is the arm that isolates the
# extra node from its price: same model, same choice available, nothing
# charged for taking it. Above the drop penalty (8000 m) the sweep is expected
# to stop buying quiet hours and start buying quiet by *not going* -- a stop
# that costs more than it is worth to skip is skipped -- and knowing where that
# turn is is the point of sweeping past it.
CROWD_COSTS_M = [0, 1000, 2000, 4000, 8000, 16000]
_WEEKDAY_CODES = ("Mo", "Tu", "We", "Th", "Fr", "Sa", "Su")


def measure(days: list[dict]) -> dict:
    """Exposure and its two components, plus what the itinerary cost to get."""
    at_hour, typical = [], []
    stops = km = 0
    for day_i, day in enumerate(days):
        weekday = _WEEKDAY_CODES[
            (START_DATE + datetime.timedelta(days=day_i)).weekday()]
        stops += len(day["route"])
        km += day["distance_km"]
        for poi, slot in zip(day["route"], day["schedule"]):
            hour = int(slot["arrival"]) % 24
            reading = busyness(poi, weekday, hour)
            level = busyness(poi)
            if reading is None or level is None:
                continue
            at_hour.append(reading)
            typical.append(level)
    n = len(at_hour)
    return {
        "stops": stops,
        "stops_per_day": stops / len(days) if days else 0.0,
        "km": round(km, 2),
        "km_per_stop": round(km / stops, 3) if stops else None,
        "known_stops": n,
        "exposure": round(sum(at_hour) / n, 1) if n else None,
        "typical": round(sum(typical) / n, 1) if n else None,
        "hour_gap": round((sum(at_hour) - sum(typical)) / n, 1) if n else None,
    }


def run_measurement() -> pd.DataFrame:
    graph = GraphIndex()
    rag = FusionRAGAgent()
    router = RouterAgent(graph)
    prefs_by_archetype = archetype_preferences()
    shipped_cost = toptw.CROWD_COST_M
    rows = []
    for city in CITIES:
        for archetype in ARCHETYPES:
            sightseeing, food, _ = build_pools(graph, rag, city, archetype,
                                               TOP_K_TOTAL)
            prefs = prefs_by_archetype[archetype]
            for hours in BUDGET_HOURS:
                common = dict(city=city, archetype=archetype, budget_hours=hours,
                              day_start_hour=start_hour_for(archetype))
                for arm, hour_aware, cost in (
                        [("baseline", False, None)] +
                        [(f"hour_aware_{c}", True, c) for c in CROWD_COSTS_M]):
                    if cost is not None:
                        toptw.CROWD_COST_M = cost
                    started = time.time()
                    days = router.run(city, sightseeing + food, n_days=N_DAYS,
                                      daily_minutes_budget=hours * 60,
                                      archetype=archetype, narrate=False,
                                      start_date=START_DATE, preferences=prefs,
                                      hour_aware=hour_aware)["itinerary"]
                    rows.append({**common, "arm": arm, "crowd_cost_m": cost,
                                 **measure(days),
                                 "solve_seconds": round(time.time() - started, 2)})
                print(f"  {city} / {archetype} / {hours}h done", flush=True)
    toptw.CROWD_COST_M = shipped_cost
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """km/stop is recomputed from the totals rather than averaged over per-trip
    ratios, and exposure is weighted by `known_stops` for the same reason: a
    trip with two measured stops should not count as much as one with twelve."""
    out = []
    for arm, g in df.groupby("arm", sort=False):
        known = g["known_stops"].sum()
        weighted = lambda col: (
            round((g[col] * g["known_stops"]).sum() / known, 1) if known else None)
        out.append({
            "arm": arm,
            "crowd_cost_m": g["crowd_cost_m"].iloc[0],
            "trips": len(g),
            "stops_per_day": round(g["stops_per_day"].mean(), 2),
            "km_per_stop": round(g["km"].sum() / g["stops"].sum(), 3),
            "known_stops": int(known),
            "exposure": weighted("exposure"),
            "typical": weighted("typical"),
            "hour_gap": weighted("hour_gap"),
            "solve_seconds": round(g["solve_seconds"].mean(), 2),
        })
    return pd.DataFrame(out)


def main():
    df = run_measurement()
    df.to_csv(RESULTS_CSV, index=False)
    summary = summarize(df)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"\nwrote {RESULTS_CSV}\nwrote {SUMMARY_CSV}\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
