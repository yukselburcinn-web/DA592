"""
What does making a landmark expensive to drop buy, and what does it spend?
(issue #122)

`iconic_coverage.py` establishes the problem: a city's best-known places reach
the router's shortlist and then the solver drops them, because one scalar
`drop_penalty_m` prices every stop the same. This sweeps the fix's two knobs:

  threshold   how prominent a POI has to be, on `scoring.quality`'s within-pool
              scale, to cost more than the flat price
  multiplier  how much more

Both are swept rather than chosen, because #122 asks for that in as many words
("Esik ve carpan supurulup olculsun, tek bir deger secilip iddia edilmesin"),
and because the failure mode of picking one is visible in the grid: a low
threshold with a high multiplier stops being "a handful of landmarks are worth
keeping" and becomes the global score reweighting `toptw_scoring_ablation.py`
already measured and rejected at #72.

The costs are read alongside the gain, on the band #33 set: stops per day and km
per stop. A configuration that adds landmarks by emptying the day is not a fix.

The knobs are set as module constants around each run rather than passed down
through `RouterAgent.run`. That is deliberate: it exercises the same call path
the app uses -- retrieval, shortlist, solver -- so what is measured here is what
a traveler would get, not a reconstruction of it.

Usage:

    python -m roamwise.evaluation.iconic_penalty_sweep           # sweep and print
    python -m roamwise.evaluation.iconic_penalty_sweep --write   # also write the CSV
"""
import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd

from roamwise.evaluation.iconic_coverage import Harness, coverage_rows
from roamwise.optimization import toptw

HERE = Path(__file__).parent
RESULTS_CSV = HERE / "iconic_penalty_sweep.csv"

# `quality` is a min-max of `popularity_score` within the working set, and it
# compresses hard at the top: a 78-POI Paris working set holds 17 POIs above
# 0.95 and 4 above 0.99. So the interesting thresholds are the high ones -- 0.95
# is already a fifth of the pool, which is closer to reweighting everything than
# to keeping a landmark.
THRESHOLDS = [0.99, 0.97, 0.95]
# Below 1.5 the penalty barely moves against an 8000 m flat price; above 5 a
# landmark is worth 40 km of walking, which buys it at any cost in stops.
MULTIPLIERS = [1.5, 2.0, 3.0, 5.0]


def sweep(thresholds=None, multipliers=None) -> pd.DataFrame:
    thresholds = thresholds or THRESHOLDS
    multipliers = multipliers or MULTIPLIERS
    harness = Harness()
    original = (toptw.ICONIC_QUALITY_THRESHOLD, toptw.ICONIC_DROP_MULTIPLIER)

    # The baseline is `multiplier = 1.0`, which is what ships today: the
    # threshold is irrelevant there and is recorded as such.
    cells = [(None, 1.0)] + [(t, m) for m in multipliers for t in thresholds]
    rows = []
    try:
        for threshold, multiplier in cells:
            toptw.ICONIC_QUALITY_THRESHOLD = 1.0 if threshold is None else threshold
            toptw.ICONIC_DROP_MULTIPLIER = multiplier
            started = time.time()
            poi_df, cell_df = coverage_rows(harness=harness)
            rows.append({
                "threshold": "-" if threshold is None else threshold,
                "multiplier": multiplier,
                "iconic_in_plan": int(cell_df.iconic_in_plan.sum()),
                "iconic_slots": int(cell_df.iconic_total.sum()),
                "iconic_in_pool": int(cell_df.iconic_in_pool.sum()),
                # Of the iconic POIs the router could have taken, how many it
                # took. The denominator is what reached the pool, because a POI
                # retrieval never surfaced is not the solver's to keep.
                "kept_of_reachable": round(
                    cell_df.iconic_in_plan.sum() / max(cell_df.iconic_in_pool.sum(), 1), 4),
                "stops_per_day": round(cell_df.stops_per_day.mean(), 3),
                "km_per_stop": round(cell_df.km_per_stop.mean(), 3),
                "categories_per_day": round(cell_df.categories_per_day.mean(), 3),
                "mean_quality": round(cell_df.mean_quality.mean(), 4),
                "mean_pref_match": round(cell_df.mean_pref_match.mean(), 4),
                "solve_seconds": round(time.time() - started, 1),
            })
            print(f"  done: threshold={rows[-1]['threshold']} "
                  f"multiplier={multiplier} -> {rows[-1]['iconic_in_plan']} iconic, "
                  f"{rows[-1]['stops_per_day']} stops/day "
                  f"({rows[-1]['solve_seconds']}s)")
    finally:
        toptw.ICONIC_QUALITY_THRESHOLD, toptw.ICONIC_DROP_MULTIPLIER = original
    return pd.DataFrame(rows)


def report(df: pd.DataFrame) -> None:
    base = df.iloc[0]
    print("\nBaseline is the first row (multiplier 1.0 = what ships today).")
    print(f"{'thresh':>7} {'mult':>5} {'iconic':>8} {'kept':>7} {'stops/day':>10} "
          f"{'km/stop':>8} {'cat/day':>8} {'d stops':>8} {'d km':>7}")
    for row in df.itertuples():
        d_stops = (row.stops_per_day - base.stops_per_day) / base.stops_per_day * 100
        d_km = (row.km_per_stop - base.km_per_stop) / base.km_per_stop * 100
        print(f"{str(row.threshold):>7} {row.multiplier:>5} "
              f"{row.iconic_in_plan:>4}/{row.iconic_slots:<3} {row.kept_of_reachable:>7.3f} "
              f"{row.stops_per_day:>10.3f} {row.km_per_stop:>8.3f} "
              f"{row.categories_per_day:>8.3f} {d_stops:>7.1f}% {d_km:>6.1f}%")
    print("\n`d stops` and `d km` are against the baseline row. #33's band for a "
          "change of this kind was 1.3% of stops and no measurable distance;\n"
          "#122's acceptance criteria reuse it, so anything wider than that has "
          "to be argued rather than shipped quietly.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help=f"write {RESULTS_CSV.name}")
    args = parser.parse_args()
    df = sweep()
    report(df)
    if args.write:
        df.to_csv(RESULTS_CSV, index=False)
        print(f"\nWrote {RESULTS_CSV}")


if __name__ == "__main__":
    main()
