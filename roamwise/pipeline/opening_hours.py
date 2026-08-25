"""Carry OSM's `opening_hours` tag into the catalogue verbatim.

`open_hour`/`close_hour` are a single pair of integers, and the tag they come
from is a grammar. Measured over the 4,404 distinct tags in this project's own
Overpass cache: 94% name a day of the week, 53% carry more than one rule, 17%
close for lunch and 11% have an explicit `off`. The regex that produced the
pair keeps the first `HH:MM-HH:MM` it finds and drops the rest, so `Tu-Su
10:00-18:00; Mo off` became `(10, 18)` -- a museum shut on Mondays, schedulable
on a Monday (issue #70).

The fix is to keep the tag itself. `opening_hours_raw` is added alongside the
existing columns rather than replacing them: the pair stays as the coarse
fallback for rows OSM never described, and every consumer that cannot read the
grammar keeps working untouched.

This is a pass over the *committed* catalogue, not a rebuild -- the same shape
as `sight_filter.py` and for the same reason: `build_catalogue.py` ranks
candidates on a 60-day rolling pageview window, so re-running it returns a
different POI set and cannot reproduce the shipped catalogue. The tag is joined
onto the rows already on disk, by Wikidata QID, out of the shared disk cache;
`build_catalogue.py` writes the same column so a future rebuild is correct at
source.

Offline: every Overpass response this needs is already cached. It reads the
cache and writes a column -- no network, and the same input always gives the
same output.

    python opening_hours.py               # report only
    python opening_hours.py --apply       # add the column to poi.csv
"""
import argparse

import pandas as pd

from build_catalogue import fetch_longtail, fetch_osm_wikidata_tags
from common import CITIES, DATA

COLUMN = "opening_hours_raw"
# poi.csv stores coordinates at six decimal places, so the long-tail join keys
# on the same rounding the catalogue was written with.
PRECISION = 6


def raw_hours():
    """(by QID, by rounded coordinate) across every city in the registry.

    Two joins because the catalogue is built from two OSM sweeps. The spine
    carries a Wikidata QID and joins on it. The long tail -- restaurants, cafés,
    bars, nightclubs, markets -- has no QID at all, and those are precisely the
    rows whose hours matter most, since they are what the meal and evening
    passes schedule. Joining on QID alone silently missed every one of them.

    Both reuse build_catalogue's own fetchers, so the join sees exactly the
    objects the catalogue was built from -- including its rule for picking,
    where two OSM objects share a QID, whichever carries the most detail.
    """
    by_qid, by_point = {}, {}
    for code, city in CITIES.items():
        tags = fetch_osm_wikidata_tags(city)
        spine = {qid: t["opening_hours"] for qid, t in tags.items() if t.get("opening_hours")}
        tail = {(round(r["lat"], PRECISION), round(r["lon"], PRECISION)): r["tags"]["opening_hours"]
                for r in fetch_longtail(city) if r["tags"].get("opening_hours")}
        print(f"  {code}: omurgada {len(spine)}, uzun kuyrukta {len(tail)} opening_hours")
        by_qid.update(spine)
        by_point.update(tail)
    return by_qid, by_point


def backfill(catalogue: pd.DataFrame) -> pd.DataFrame:
    """Return the catalogue with `opening_hours_raw` filled in where OSM has it."""
    by_qid, by_point = raw_hours()

    def lookup(row):
        qid = str(row.wikidata_qid)
        if qid.startswith("Q") and qid in by_qid:
            return by_qid[qid]
        return by_point.get((round(row.lat, PRECISION), round(row.lon, PRECISION)), "")

    out = catalogue.copy()
    out[COLUMN] = out.apply(lookup, axis=1)
    return out


def report(before: pd.DataFrame, after: pd.DataFrame) -> None:
    filled = (after[COLUMN].astype(str).str.len() > 0).sum()
    print(f"\n{len(after)} POI, {filled} tanesinde ham opening_hours ({filled / len(after):.1%})")
    print(f"  karsilastirma: hours_source == 'osm' olan satir "
          f"{(before.hours_source == 'osm').sum()}")

    # A tag the pair could not represent is the whole point of the column, so
    # say how many there are rather than just how many rows were filled.
    tagged = after[after[COLUMN].astype(str).str.len() > 0]
    day_aware = tagged[COLUMN].str.contains(r"\b(?:Mo|Tu|We|Th|Fr|Sa|Su|PH)\b", regex=True)
    multi_rule = tagged[COLUMN].str.contains(";", regex=False)
    print(f"  bunlarin {day_aware.sum()}'inde gun bilgisi, "
          f"{multi_rule.sum()}'inde birden fazla kural var")

    for column in before.columns:
        if not before[column].equals(after[column]):
            print(f"  UYARI: mevcut kolon degisti -> {column}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="poi.csv'yi yeniden yaz")
    ap.add_argument("--catalogue", default=str(DATA / "poi.csv"))
    args = ap.parse_args()

    catalogue = pd.read_csv(args.catalogue)
    filled = backfill(catalogue)
    report(catalogue, filled)

    if args.apply:
        filled.to_csv(args.catalogue, index=False)
        print(f"\nyazildi: {args.catalogue}")
    else:
        print("\n(rapor modu -- yazmak icin --apply)")


if __name__ == "__main__":
    main()
