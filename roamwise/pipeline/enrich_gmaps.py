"""Apply the Google Maps enrichment to the catalogue (issue #71).

What this does: rewrites `opening_hours_raw`, `hours_source`, `price_level`,
`price_source` and the coarse `open_hour`/`close_hour` fallback pair for every
POI the enrichment covers. It adds no columns -- the catalogue schema is
depended on by retrieval and the graph, so this is deliberately a value-only
change.

What this needs, and does not ship: a cache directory of scraped records, one
JSON per `poi_id`, produced outside this repository (default
`local/gmaps/cache`, which `.gitignore` excludes). Running it is therefore not
part of the normal pipeline -- `poi.csv` is committed with the enrichment
already applied, and this script is here so the provenance of those rows is
readable in code rather than asserted in prose.

Hours are converted to the OSM `opening_hours` grammar rather than to a pair of
integers, because that is the grammar `optimization/routing._tag_intervals`
already parses (issue #70): a weekday-aware rule set survives the trip to the
router intact, while a pair collapses "shut on Mondays" into "open every day".

    python pipeline/enrich_gmaps.py --write
"""
import argparse
import csv
import json
import re
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE.parent / "data"
DEFAULT_CACHE = HERE.parent.parent / "local" / "gmaps" / "cache"

OSM_DAY = {"Monday": "Mo", "Tuesday": "Tu", "Wednesday": "We", "Thursday": "Th",
           "Friday": "Fr", "Saturday": "Sa", "Sunday": "Su"}


def _hhmm(tok: str, is_end: bool):
    """'9 AM' / '7:30 PM' / '12 AM' -> 'HH:MM'.

    A closing '12 AM' is midnight at the *end* of the day, so it has to be
    24:00. Written as 00:00 the interval runs backwards and the venue reads as
    shut all day."""
    m = re.match(r"(\d{1,2})(?::(\d{2}))?\s*(AM|PM)", tok.strip(), re.I)
    if not m:
        return None
    h, mi, ap = int(m.group(1)), int(m.group(2) or 0), m.group(3).upper()
    h = h % 12 + (12 if ap == "PM" else 0)
    if is_end and h == 0 and mi == 0:
        return "24:00"
    return f"{h:02d}:{mi:02d}"


def _spans(text: str):
    """'7:30 AM to 1 AM' or '9 AM to 12 PM, 2 to 6 PM' -> ['07:30-01:00'], ..."""
    out = []
    for part in re.split(r",\s*", text.strip()):
        m = re.match(r"(.+?)\s+to\s+(.+)", part.strip(), re.I)
        if not m:
            continue
        a_raw, b_raw = m.group(1).strip(), m.group(2).strip()
        if not re.search(r"AM|PM", a_raw, re.I):
            # "2 to 6 PM": the meridiem is only written once, on the end.
            ap = re.search(r"(AM|PM)", b_raw, re.I)
            if ap:
                a_raw = f"{a_raw} {ap.group(1)}"
        a, b = _hhmm(a_raw, False), _hhmm(b_raw, True)
        if a and b:
            out.append(f"{a}-{b}")
    return out


def to_osm(weekly: dict) -> str | None:
    """{'Monday': '9 AM to 6 PM', 'Tuesday': 'Closed'} -> 'Mo 09:00-18:00; Tu off'"""
    if not weekly:
        return None
    rules = []
    for day, text in weekly.items():
        code = OSM_DAY.get(day)
        if not code:
            continue
        low = (text or "").strip().lower()
        if low.startswith("closed"):
            rules.append(f"{code} off")
        elif "24 hours" in low:
            rules.append(f"{code} 00:00-24:00")
        else:
            spans = _spans(text)
            rules.append(f"{code} {','.join(spans)}" if spans else f"{code} off")
    return "; ".join(rules) or None


def coarse_pair(weekly: dict):
    """The `open_hour`/`close_hour` fallback, from the most common open day.

    Kept in step with the tag because it is what the router falls back to when
    a tag cannot be parsed; leaving the old category default there would make
    the fallback disagree with the row it belongs to."""
    starts, ends = [], []
    for text in (weekly or {}).values():
        low = (text or "").strip().lower()
        if low.startswith("closed"):
            continue
        if "24 hours" in low:
            starts.append(0); ends.append(24); continue
        spans = _spans(text)
        if not spans:
            continue
        first, last = spans[0], spans[-1]
        sh, sm = (int(x) for x in first.split("-")[0].split(":"))
        eh, em = (int(x) for x in last.split("-")[1].split(":"))
        starts.append(sh + sm / 60)
        ends.append(eh + em / 60)
    if not starts:
        return None
    starts.sort(); ends.sort()
    return int(starts[len(starts) // 2]), int(round(ends[len(ends) // 2]))


def load_cache(cache_dir: Path) -> dict:
    if not cache_dir.exists():
        raise SystemExit(f"enrichment cache not found: {cache_dir}\n"
                         "poi.csv already ships with the enrichment applied; this "
                         "script only re-applies it from a local cache.")
    out = {}
    for f in sorted(cache_dir.glob("*.json")):
        rec = json.loads(f.read_text())
        if rec.get("status") == "ok":
            out[rec["poi_id"]] = rec
    return out


def apply(rows: list[dict], scraped: dict) -> dict:
    """Fill only where the catalogue is silent. An existing observed value wins.

    This is a gap-filler, not an overwrite, and the reason is measured. Of the
    326 rows the enrichment covers, 194 already carried an OSM `opening_hours`
    tag. Overwriting those buys **no coverage at all** -- the enrichment's whole
    contribution is the 132 rows where OSM said nothing -- and it costs three
    things:

      * 58 of those 194 tags carry rules the scraped weekly view cannot
        express: seasons and public holidays. The Eiffel Tower's tag is
        `09:30-23:45; Jun 21-Sep 02: 09:00-00:45; Jul 14,Jul 15 off` -- summer
        hours and a Bastille Day closure. A seven-day table flattens both away.
      * On 19 of the 194 the two sources disagree about which *days* the place
        is open at all, and only 36% agree exactly. Overwriting picks a winner
        silently on every one of them.
      * Every overwritten row enlarges the amount of third-party content in a
        public repository for no gain.

    So OSM wins wherever it spoke. Conflicts are counted and reported rather
    than resolved by rule, because a disagreement about opening days is a data
    question for a person, not something a preference order should bury."""
    stats = {"hours": 0, "price": 0, "coarse": 0, "untouched": 0,
             "hours_kept_osm": 0, "price_kept": 0}
    for row in rows:
        rec = scraped.get(row["poi_id"])
        if not rec:
            stats["untouched"] += 1
            continue
        tag = to_osm(rec.get("hours_weekly"))
        if tag:
            if (row.get("opening_hours_raw") or "").strip():
                stats["hours_kept_osm"] += 1          # katalog konuşmuş, dokunma
            else:
                row["opening_hours_raw"] = tag
                row["hours_source"] = "gmaps"
                stats["hours"] += 1
                pair = coarse_pair(rec["hours_weekly"])
                if pair:
                    row["open_hour"], row["close_hour"] = str(pair[0]), str(pair[1])
                    stats["coarse"] += 1
        if rec.get("price_tier"):
            if row.get("price_source") in ("osm",):
                stats["price_kept"] += 1
            else:
                row["price_level"] = str(rec["price_tier"])
                row["price_source"] = "gmaps"
                stats["price"] += 1
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--write", action="store_true", help="write data/poi.csv in place")
    args = ap.parse_args()

    path = DATA / "poi.csv"
    raw = path.read_text().splitlines(keepends=True)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    before = [dict(r) for r in rows]

    stats = apply(rows, load_cache(args.cache))
    touched = [i for i, (a, b) in enumerate(zip(before, rows)) if a != b]
    print(f"{len(rows)} rows | hours filled {stats['hours']} | price filled {stats['price']} "
          f"| coarse pair {stats['coarse']}")
    print(f"kept existing: {stats['hours_kept_osm']} hours, {stats['price_kept']} price "
          f"| no enrichment: {stats['untouched']}")
    print(f"rows whose values actually change: {len(touched)}")
    if not args.write:
        print("dry run; pass --write to update data/poi.csv")
        return

    # Only the changed rows are re-serialised. Rewriting the whole file would
    # re-quote every description that happens to contain a comma or a quote and
    # bury 331 real edits in a 654-line diff nobody can review.
    import io
    for i in touched:
        buf = io.StringIO()
        csv.DictWriter(buf, fieldnames=fields, lineterminator="\n").writerow(rows[i])
        raw[i + 1] = buf.getvalue()
    path.write_text("".join(raw))
    print(f"wrote {path} ({len(touched)} lines rewritten)")


if __name__ == "__main__":
    main()
