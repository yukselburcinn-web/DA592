"""Per-POI, per-hour busyness for the score's crowding factor (issue #71/#72).

`optimization/scoring.crowding_discount` returned a constant 1.0 because the
catalogue held no per-POI and no hourly demand: `ForecasterAgent` answers at
the granularity of a city and a month, and one scalar multiplied onto every
candidate cancels out of a normalised score. This writes the series that
factor needs.

Output `data/crowding.csv` is long-format -- one row per (poi, weekday, hour)
with the busyness percentage that hour carries -- because the two consumers
want different shapes: the score wants each POI's typical level, and the
measurement in REPORT §5 wants the hour-by-hour curve. Deriving both from one
long table beats committing two summaries that can drift apart.

Same cache contract as `enrich_gmaps.py`: the scraped records are not in this
repository, `crowding.csv` is, and this script exists so the provenance of its
rows is readable rather than asserted.

Every row carries a `source`, following `poi.csv`'s `hours_source` /
`price_source` (issue #33). It reads `gmaps` on all 35,141 of today's rows,
and a constant column is still worth its bytes for two reasons: the file
states where it came from instead of a README asserting it, and the catalogue
is expected to gain a second, thinner source -- the issue's fallback is a
monthly Wikipedia pageviews series for the 59% this scrape never reached.
When that lands it has to sit *beside* these rows rather than be told apart
from them by which POI it is about, and a consumer weighting an hourly
reading differently from a monthly one needs the column to do it.

    python pipeline/build_crowding.py --write
"""
import argparse
import csv
import json
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE.parent / "data"
DEFAULT_CACHE = HERE.parent.parent / "local" / "gmaps" / "cache"
# What produced a row, in `poi.csv`'s `*_source` vocabulary. Every row this
# script writes is a Google Maps popular-times reading; the column exists so a
# second source can land beside them without a schema change.
SOURCE = "gmaps"
DAY_CODE = {"Monday": "Mo", "Tuesday": "Tu", "Wednesday": "We", "Thursday": "Th",
            "Friday": "Fr", "Saturday": "Sa", "Sunday": "Su"}


def rows_from_cache(cache_dir: Path):
    if not cache_dir.exists():
        raise SystemExit(f"crowding cache not found: {cache_dir}\n"
                         "data/crowding.csv already ships; this script only rebuilds it.")
    out = []
    for f in sorted(cache_dir.glob("*.json")):
        rec = json.loads(f.read_text())
        if rec.get("status") != "ok" or not rec.get("popular_times"):
            continue
        for day, hours in rec["popular_times"].items():
            code = DAY_CODE.get(day)
            if not code:
                continue
            for hour, busy in hours.items():
                # "_current_usual" is the live reading the widget shows for the
                # hour you happen to look at it. It is a measurement of one
                # moment, not part of the typical week, and keeping it would put
                # the time of the scrape into the catalogue.
                if not str(hour).isdigit():
                    continue
                out.append((rec["poi_id"], code, int(hour), int(busy), SOURCE))
    out.sort(key=lambda r: (r[0], list(DAY_CODE.values()).index(r[1]), r[2]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    rows = rows_from_cache(args.cache)
    pois = {r[0] for r in rows}
    print(f"{len(rows)} rows for {len(pois)} POIs")
    if not args.write:
        print("dry run; pass --write to update data/crowding.csv")
        return
    path = DATA / "crowding.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["poi_id", "day", "hour", "busy", "source"])
        w.writerows(rows)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
