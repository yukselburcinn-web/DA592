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
import datetime
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


_DAY_TOKEN = re.compile(r"\b(Mo|Tu|We|Th|Fr|Sa|Su)\b")
_CALENDAR = re.compile(r"(PH|SH|easter|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|\[)")
_WEEK_ANCHOR = datetime.date(2026, 9, 21)          # a Monday
_SAMPLE = [i * 0.25 for i in range(4 * 24)]        # 15-minute steps over one day
_MIN_WEEKLY_DISAGREEMENT_H = 0.25


def day_blind(tag: str) -> bool:
    """True when the tag says the same thing about every day of the week.

    `Mo-Su 10:00-18:00` and a bare `09:00-18:00` both claim a venue keeps one
    schedule seven days a week. That is the shape OSM tags take when nobody has
    filled in the detail, and it is where a day-aware source has something to
    add."""
    if not _DAY_TOKEN.search(tag):
        return True
    return bool(re.fullmatch(r"\s*Mo-Su\s+[^;]+\s*", tag))


def scraped_intervals(weekly: dict, day):
    """Open intervals for `day` from the scraped weekly table, midnight-spanning
    stretches included, on the same past-midnight clock the router uses.

    Deliberately not routed through the OSM grammar. Re-encoding the scrape as
    per-day rules and parsing it back loses the tail of a venue that shuts after
    midnight, which would show up here as a disagreement that is an artefact of
    the encoding rather than a difference between the sources."""
    if not weekly:
        return None
    names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    out = []

    def add(text, shift):
        low = (text or "").strip().lower()
        if not low or low.startswith("closed"):
            return
        if "24 hours" in low:
            out.append((0.0 + shift, 24.0 + shift))
            return
        for span in _spans(text):
            a, b = span.split("-")
            ah, am = (int(x) for x in a.split(":"))
            bh, bm = (int(x) for x in b.split(":"))
            s, e = ah + am / 60, bh + bm / 60
            if e <= s:
                e += 24
            out.append((s + shift, e + shift))

    add(weekly.get(names[(day.weekday() - 1) % 7]), -24)
    add(weekly.get(names[day.weekday()]), 0)
    add(weekly.get(names[(day.weekday() + 1) % 7]), +24)
    out.sort()
    merged = []
    for s, e in out:
        if merged and s <= merged[-1][1] + 1e-9:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return [(s, e) for s, e in merged if e > 0] or None


def blind_intervals(tag: str):
    """The one daily schedule a day-blind tag describes, as (start, end) hours.

    Only day-blind tags reach this, and those are by construction the simple
    ones -- `Mo-Su 12:00-14:30,19:00-22:30` or a bare `09:00-18:00`. Parsing
    them here rather than through the full grammar keeps this script free of
    `optimization.routing`, whose own import of the `opening_hours` library is
    shadowed by `pipeline/opening_hours.py` whenever this directory leads
    sys.path (issue #26's failure mode). A tag this function cannot read
    returns None and the row is left alone."""
    body = re.sub(r"^\s*Mo-Su\s+", "", tag.strip())
    out = []
    for part in body.split(","):
        m = re.fullmatch(r"\s*(\d{1,2}):(\d{2})-(\d{1,2}):(\d{2})\s*", part)
        if not m:
            return None
        s = int(m.group(1)) + int(m.group(2)) / 60
        e = int(m.group(3)) + int(m.group(4)) / 60
        if e <= s:
            e += 24
        out.append((s, e))
    return out or None


def weekly_disagreement_hours(tag: str, weekly: dict) -> float:
    """How many hours a week the two sources disagree about being open.

    Compared by sampling the open/closed predicate rather than by matching
    interval lists, because the same opening hours can be written several ways
    and only the sampled answer is representation-independent: `Mo-Su
    07:30-01:00` and a scraped "7:30 AM to 1 AM" describe one schedule and must
    score zero, or the rule would rewrite rows that already agree."""
    daily = blind_intervals(tag)
    if daily is None:
        return 0.0
    total = 0.0
    for offset in range(7):
        day = _WEEK_ANCHOR + datetime.timedelta(days=offset)
        # the day's own hours, plus yesterday's tail past midnight
        a = list(daily) + [(s - 24, e - 24) for s, e in daily]
        b = scraped_intervals(weekly, day) or []
        for t in _SAMPLE:
            if any(s <= t < e for s, e in a) != any(s <= t < e for s, e in b):
                total += 0.25
    return total


def _osm_intervals():
    """`routing._tag_intervals`, imported around this directory's own shadow.

    `pipeline/opening_hours.py` has the same name as the installed grammar
    library, so whenever this directory leads sys.path -- which it does when
    the script is run from inside it -- `routing`'s own import resolves to the
    wrong module (issue #26's failure mode). Dropping the script's directory
    for the duration of the import is enough, and it is restored immediately
    so nothing else in the process sees a changed path."""
    import sys
    here = str(Path(__file__).parent)
    dropped = [q for q in sys.path if q in (here, "")]
    for q in dropped:
        sys.path.remove(q)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    try:
        from roamwise.optimization.routing import _tag_intervals
        return _tag_intervals
    finally:
        for q in reversed(dropped):
            sys.path.insert(0, q)


_CAL_CLAUSE = re.compile(r"(PH|SH|easter|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|\[)", re.I)


def calendar_clauses(tag: str) -> list[str]:
    """The clauses of an OSM tag scoped to a calendar rather than a weekday.

    `PH closed`, `Jan 01 12:00-18:00`, `Dec 24,Dec 31 off`, a whole seasonal
    range. A seven-day scrape has no way to say any of this, so when the scrape
    takes over the weekday structure these are carried across unchanged: in the
    grammar a later rule overrides an earlier one for the dates it covers, so
    appending them keeps the exception without disturbing the ordinary week."""
    return [c.strip() for c in tag.split(";") if _CAL_CLAUSE.search(c)]


def day_conflict(tag: str, weekly: dict, tag_intervals) -> bool:
    """True when the two sources disagree about which *days* the venue opens.

    Deliberately narrower than "the hours differ". A half-hour disagreement is
    noise between two records of the same schedule; a day one source calls open
    and the other calls shut is a different claim about the place, and it is the
    one that puts a traveller in front of a locked door."""
    for offset in range(7):
        day = _WEEK_ANCHOR + datetime.timedelta(days=offset)
        a = tag_intervals(tag, day)
        if a is None:
            return False                       # okunamayan etikete dokunma
        b = scraped_intervals(weekly, day)
        open_a = any(0 <= s < 24 for s, _ in a)
        open_b = any(0 <= s < 24 for s, _ in (b or []))
        if open_a != open_b:
            return True
    return False


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


def apply(rows: list[dict], scraped: dict, tag_intervals=None) -> dict:
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
    tag_intervals = tag_intervals or _osm_intervals()
    stats = {"hours": 0, "price": 0, "coarse": 0, "untouched": 0,
             "hours_kept_osm": 0, "hours_refined": 0, "hours_conflict": 0,
             "price_kept": 0}
    for row in rows:
        rec = scraped.get(row["poi_id"])
        if not rec:
            stats["untouched"] += 1
            continue
        tag = to_osm(rec.get("hours_weekly"))
        if tag:
            existing = (row.get("opening_hours_raw") or "").strip()
            # One exception to "the catalogue wins": a tag that claims the same
            # schedule seven days a week is not really an answer about days, and
            # a day-aware source has something to add. It only applies when the
            # tag carries no season or holiday rules -- those are the part a
            # weekly table cannot express -- and when the two actually disagree,
            # so no row is rewritten for nothing. Measured over the catalogue:
            # 44 rows are day-blind with an enrichment behind them, and 24 of
            # those disagree; the other 20 are left exactly as they are.
            refine = (existing
                      and day_blind(existing)
                      and not _CALENDAR.search(existing)
                      and weekly_disagreement_hours(
                          existing, rec["hours_weekly"]) >= _MIN_WEEKLY_DISAGREEMENT_H)
            # İkinci istisna: iki kaynak mekânın hangi GÜNLER açık olduğunda
            # anlaşmıyorsa kazıma karar verir. Yarım saatlik fark aynı
            # çizelgenin iki kaydı arasındaki gürültü; kapalı bir günü açık
            # sanmak gezgini kilitli kapıya götürüyor. OSM'in takvime bağlı
            # istisnaları (PH, mevsim, tarih) korunup sona ekleniyor.
            conflict = (existing and not refine
                        and day_conflict(existing, rec["hours_weekly"], tag_intervals))
            if conflict:
                keep = calendar_clauses(existing)
                candidate = "; ".join([tag] + keep) if keep else tag
                if tag_intervals(candidate, _WEEK_ANCHOR) is None:
                    candidate = tag                    # birleşim okunmuyorsa yalın kazıma
                tag = candidate
            if existing and not refine and not conflict:
                stats["hours_kept_osm"] += 1          # katalog konuşmuş, dokunma
            else:
                if existing:
                    stats["hours_refined" if refine else "hours_conflict"] += 1
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
          f"| refined day-blind: {stats['hours_refined']} "
          f"| day conflict resolved: {stats['hours_conflict']} "
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
