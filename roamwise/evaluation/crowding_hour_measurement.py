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

The gap is also reported **split by kind**, and that split is not cosmetic:
issue #33 shipped the quiet window for sightseeing only, on the grounds that a
meal sitting is compulsory and so carries no choice of hour. Split, it showed
sightseeing at -0.4 and meals still at +8.2 -- the whole residual was in the
half that had been left out, because a sitting is pinned to a four-hour band
rather than to a minute. Issue #109 gives the sittings their own quiet window
inside that band, at their own lower price: a meal cannot be skipped, so a
high price is answered by rebuilding the day around a quiet restaurant, and
the sightseeing stops pay for it. Reading the two kinds together hides that
trade entirely.

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
from roamwise.optimization.routing import FOOD_CATEGORY
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
# The same sweep for the meal sittings (#109), with sightseeing held at the
# shipped price. Separate because the two cannot be swept together: a meal
# cannot be declined, so its price is answered by rebuilding the day rather
# than by dropping the stop, and crossing the two sweeps would leave no way to
# tell which of the two prices moved a sightseeing stop.
MEAL_COSTS_M = [0, 1000, 1500, 2000, 2500]
_WEEKDAY_CODES = ("Mo", "Tu", "We", "Th", "Fr", "Sa", "Su")


def measure(days: list[dict]) -> dict:
    """Exposure and its two components, plus what the itinerary cost to get.

    `meals_per_day` and `both_sittings` are here because #109 prices an hour
    the solver cannot decline: a sitting is compulsory, so a change that
    improved the meal gap by quietly dropping a sitting would look like a win
    in every other column. These two are the contract #20 and #29 established
    and the thing to check first when the gap moves."""
    at_hour, typical, gaps_by_kind = [], [], {"sight": [], "meal": []}
    stops = km = meals = 0
    both = 0
    for day_i, day in enumerate(days):
        weekday = _WEEKDAY_CODES[
            (START_DATE + datetime.timedelta(days=day_i)).weekday()]
        stops += len(day["route"])
        km += day["distance_km"]
        on_day = sum(1 for p in day["route"] if p.get("category") == FOOD_CATEGORY)
        meals += on_day
        both += 1 if on_day >= 2 else 0
        for poi, slot in zip(day["route"], day["schedule"]):
            hour = int(slot["arrival"]) % 24
            reading = busyness(poi, weekday, hour)
            level = busyness(poi)
            if reading is None or level is None:
                continue
            at_hour.append(reading)
            typical.append(level)
            kind = "meal" if poi.get("category") == FOOD_CATEGORY else "sight"
            gaps_by_kind[kind].append(reading - level)
    n = len(at_hour)
    mean = lambda v: round(sum(v) / len(v), 1) if v else None
    return {
        "stops": stops,
        "stops_per_day": stops / len(days) if days else 0.0,
        "km": round(km, 2),
        "km_per_stop": round(km / stops, 3) if stops else None,
        "known_stops": n,
        "exposure": round(sum(at_hour) / n, 1) if n else None,
        "typical": round(sum(typical) / n, 1) if n else None,
        "hour_gap": round((sum(at_hour) - sum(typical)) / n, 1) if n else None,
        "sight_gap": mean(gaps_by_kind["sight"]),
        "sight_stops": len(gaps_by_kind["sight"]),
        "meal_gap": mean(gaps_by_kind["meal"]),
        "meal_stops": len(gaps_by_kind["meal"]),
        "meals_per_day": round(meals / len(days), 2) if days else None,
        "both_sittings": round(both / len(days), 3) if days else None,
    }


def run_measurement() -> pd.DataFrame:
    graph = GraphIndex()
    rag = FusionRAGAgent()
    router = RouterAgent(graph)
    prefs_by_archetype = archetype_preferences()
    shipped_cost = toptw.CROWD_COST_M
    shipped_meal_cost = toptw.CROWD_MEAL_COST_M
    rows = []
    for city in CITIES:
        for archetype in ARCHETYPES:
            sightseeing, food, _ = build_pools(graph, rag, city, archetype,
                                               TOP_K_TOTAL)
            prefs = prefs_by_archetype[archetype]
            for hours in BUDGET_HOURS:
                common = dict(city=city, archetype=archetype, budget_hours=hours,
                              day_start_hour=start_hour_for(archetype))
                for arm, hour_aware, cost, meal_cost in (
                        [("baseline", False, None, None)] +
                        [(f"hour_aware_{c}", True, c, shipped_meal_cost)
                         for c in CROWD_COSTS_M] +
                        [(f"meal_{c}", True, shipped_cost, c)
                         for c in MEAL_COSTS_M]):
                    if cost is not None:
                        toptw.CROWD_COST_M = cost
                    if meal_cost is not None:
                        toptw.CROWD_MEAL_COST_M = meal_cost
                    started = time.time()
                    days = router.run(city, sightseeing + food, n_days=N_DAYS,
                                      daily_minutes_budget=hours * 60,
                                      archetype=archetype, narrate=False,
                                      start_date=START_DATE, preferences=prefs,
                                      hour_aware=hour_aware)["itinerary"]
                    rows.append({**common, "arm": arm, "crowd_cost_m": cost,
                                 "meal_cost_m": meal_cost, **measure(days),
                                 "solve_seconds": round(time.time() - started, 2)})
                print(f"  {city} / {archetype} / {hours}h done", flush=True)
    toptw.CROWD_COST_M = shipped_cost
    toptw.CROWD_MEAL_COST_M = shipped_meal_cost
    return pd.DataFrame(rows)


# Solve time is summarised as a median, not a mean, and the committed CSV is
# from a run on an idle machine. An earlier run of the same script on a busy
# one had three solves at 199 s, 603 s and 1035 s against a median of 1.37 s,
# which a mean turned into "7.7 s per solve" -- a serious claim if it were
# true. It was not: re-run idle, the whole file came back with every column
# except `solve_seconds` byte-identical, those three cases at 1.20 s, 0.97 s
# and 1.28 s, and a maximum of 4.60 s over all 288 solves. The model is
# deterministic (a fixed iteration budget, never a wall-clock one), so the same
# input does the same work and a time that large can only be the machine.
# Keeping the median means a busy machine cannot make this file say otherwise.
def _weighted_by(g: pd.DataFrame, col: str, weight: str):
    """A mean over stops rather than over trips: a trip that scheduled two
    measured meals should not count as much as one that scheduled six."""
    rows = g[g[col].notna()]
    total = rows[weight].sum()
    return round((rows[col] * rows[weight]).sum() / total, 1) if total else None


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
            "meal_cost_m": g["meal_cost_m"].iloc[0],
            "trips": len(g),
            "stops_per_day": round(g["stops_per_day"].mean(), 2),
            "km_per_stop": round(g["km"].sum() / g["stops"].sum(), 3),
            "known_stops": int(known),
            "exposure": weighted("exposure"),
            "typical": weighted("typical"),
            "hour_gap": weighted("hour_gap"),
            "sight_gap": _weighted_by(g, "sight_gap", "sight_stops"),
            "meal_gap": _weighted_by(g, "meal_gap", "meal_stops"),
            "meals_per_day": round(g["meals_per_day"].mean(), 2),
            "both_sittings": round(g["both_sittings"].mean(), 3),
            "solve_seconds": round(g["solve_seconds"].median(), 2),
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
