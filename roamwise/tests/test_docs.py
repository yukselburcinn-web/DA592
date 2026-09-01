"""The numbers README.md and REPORT.md quote, checked against the data.

Why this file exists. A cleanup pass over the repository found 25 figures in
the prose that no longer described the repository: the catalogue was quoted at
700 POIs after #65 removed 46, the comparative table still held the
pre-#175/#176 query set, and the test count appeared in four places as three
different numbers -- REPORT §1 said 235 while REPORT §4, two pages later, said
241. Every one of them was correct when written. Prose has no equivalent of a
failing build, so a document drifts silently the way `data/knowledge_graph.gml`
drifted for nineteen commits before #145 guarded it (#128, #145). This is the
same guard applied to the numbers instead of to the export.

What it does *not* do is check the prose. It checks the figures that are
derivable from a committed file -- the catalogue, the evaluation CSVs, the
retrieval gold set -- plus one consistency rule for the test count, which no
committed file holds. A sentence can still be out of date here; a number
cannot.

When one of these fails after a deliberate change, the fix is to update the
document, not the expectation: the CSV is the source and the sentence is the
copy. `_figure()` prints the phrase it was looking for, so the failure names
the sentence to edit.
"""
import csv
import re
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
README = (REPO / "README.md").read_text(encoding="utf-8")
REPORT = (REPO / "REPORT.md").read_text(encoding="utf-8")
CLAUDE_MD = (REPO / "CLAUDE.md").read_text(encoding="utf-8")
PYTEST_INI = (REPO / "roamwise" / "pytest.ini").read_text(encoding="utf-8")
WORKFLOW = (REPO / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")

DATA = REPO / "roamwise" / "data"
EVAL = REPO / "roamwise" / "evaluation"


def _rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


POIS = _rows(DATA / "poi.csv")


def _figure(text, pattern, label):
    """The one number `pattern` captures, or a failure naming the sentence.

    Every pattern here is anchored on words either side of the figure rather
    than on a line number, so re-flowing a paragraph does not break the test
    and moving the sentence to another section does not either. A pattern that
    stops matching is itself the finding: the sentence it describes was
    rewritten, and whoever rewrote it should say what the new number is.
    """
    found = re.findall(pattern, text)
    assert len(found) == 1, (
        f"{label}: expected exactly one match for {pattern!r}, found {len(found)}. "
        "The sentence was rewritten or removed -- update this pattern with it.")
    return found[0]


def _int(text, pattern, label):
    return int(str(_figure(text, pattern, label)).replace(",", ""))


# ---------------------------------------------------------------- catalogue

def test_the_catalogue_size_the_docs_quote_is_the_catalogue_that_ships():
    """654, not the 700 the pipeline aimed at before #65 dropped 46 rows.

    `common.CITIES` carries a `target` of 400 and 300, and both documents used
    to quote those targets as if they were counts. They are what the build
    aims at; what ships is what survived the filters.
    """
    actual = len(POIS)
    assert _int(README, r"Paris \((\d+) POIs\)", "README catalogue size") == \
        Counter(p["destination_id"] for p in POIS)["PAR"]
    assert _int(README, r"Berlin \((\d+)\), (?:\d+) in all", "README Berlin size") == \
        Counter(p["destination_id"] for p in POIS)["BER"]
    assert _int(README, r"Berlin \(\d+\), (\d+) in all", "README total") == actual
    assert _int(REPORT, r"\*\*(\d+) POIs\*\* — \d+ Paris, \d+ Berlin", "REPORT total") == actual
    assert _int(REPORT, r"\*\*\d+ POIs\*\* — (\d+) Paris", "REPORT Paris") == \
        Counter(p["destination_id"] for p in POIS)["PAR"]
    assert _int(REPORT, r"\*\*\d+ POIs\*\* — \d+ Paris, (\d+) Berlin", "REPORT Berlin") == \
        Counter(p["destination_id"] for p in POIS)["BER"]


def test_the_description_provenance_split_matches_poi_csv():
    """The three `description_source` values, as both documents count them."""
    src = Counter(p["description_source"] for p in POIS)
    assert _int(README, r"where one exists \((\d+) of \d+\)", "README wikipedia") == src["wikipedia"]
    assert _int(README, r"the Wikidata one-liner otherwise \((\d+)\)", "README wikidata") == src["wikidata"]
    assert _int(README, r"a category template only as a last resort \((\d+)\)", "README template") == src["template"]
    assert _int(REPORT, r"article where one exists \(\*\*(\d+)\*\*", "REPORT wikipedia") == src["wikipedia"]
    assert _int(REPORT, r"where it does not \(\*\*(\d+)\*\*\)", "REPORT wikidata") == src["wikidata"]
    assert _int(REPORT, r"only as a last resort \(\*\*(\d+)\*\*\)", "REPORT template") == src["template"]


def test_the_hours_and_price_provenance_counts_match_poi_csv():
    """#71 added `gmaps` beside `osm`; REPORT §3.1 states all six counts."""
    hours = Counter(p["hours_source"] for p in POIS)
    price = Counter(p["price_source"] for p in POIS)
    assert _int(REPORT, r"`hours_source` reads `osm` for \*\*(\d+)\*\*", "REPORT hours osm") == hours["osm"]
    assert _int(REPORT, r"`gmaps` for \*\*(\d+)\*\* and `category_default` for \*\*\d+\*\*;", "REPORT hours gmaps") == hours["gmaps"]
    assert _int(REPORT, r"and `category_default` for \*\*(\d+)\*\*;", "REPORT hours default") == hours["category_default"]
    assert _int(REPORT, r"`price_source` reads `osm` for \*\*(\d+)\*\*", "REPORT price osm") == price["osm"]
    assert _int(REPORT, r"`gmaps` for \*\*(\d+)\*\* and `category_default` for \*\*\d+\*\*\.", "REPORT price gmaps") == price["gmaps"]
    assert _int(REPORT, r"`gmaps` for \*\*\d+\*\* and `category_default` for \*\*(\d+)\*\*\.", "REPORT price default") == price["category_default"]
    assert _int(README, r"when present \((\d+) rows\)", "README fee rows") == price["osm"]


def test_the_verbatim_opening_hours_coverage_matches_poi_csv():
    """`opening_hours_raw` is what #70's weekday-aware resolver reads.

    The count is not `hours_source == "osm"`: 15 rows carry a tag the source
    column does not vouch for, and the enrichment (#71) wrote 180 more, so the
    three numbers have to be stated separately or the sentence is wrong three
    ways at once.
    """
    carried = [p for p in POIS if p["opening_hours_raw"].strip()]
    weekday = re.compile(r"\b(Mo|Tu|We|Th|Fr|Sa|Su)\b")
    named = [p for p in carried if weekday.search(p["opening_hours_raw"])]
    assert _int(README, r"(\d+) of \d+ rows carry an `opening_hours` tag verbatim", "README raw rows") == len(carried)
    assert _int(README, r"and (\d+) of those name a day of the week", "README weekday rows") == len(named)


def test_the_wikivoyage_gold_set_size_the_report_quotes_is_the_committed_one():
    gold = _rows(EVAL / "retrieval_gold.csv")
    assert _int(REPORT, r"(\d+) of the \d+ catalogue POIs carry a Wikivoyage listing",
                "REPORT gold size") == len({g["poi_id"] for g in gold})
    assert _int(REPORT, r"\d+ of the (\d+) catalogue POIs carry a Wikivoyage listing",
                "REPORT gold denominator") == len(POIS)


# ------------------------------------------------------- comparative analysis

def _summary():
    return {r["config"]: r for r in _rows(EVAL / "comparative_analysis_summary.csv")}


def test_the_query_count_both_docs_quote_is_the_query_set_that_was_run():
    """68 since #176 grew the chain tier from six to ten."""
    results = _rows(EVAL / "comparative_analysis_results.csv")
    n = len({r["query_id"] for r in results})
    assert _int(README, r"It scores (\d+) queries across 2 cities", "README query count") == n
    assert _int(REPORT, r"runs (\d+) queries in three tiers", "REPORT query count") == n


def test_the_tier_sizes_readme_quotes_are_the_tiers_that_were_run():
    results = [r for r in _rows(EVAL / "comparative_analysis_results.csv") if r["config"] == "fusion"]
    tiers = Counter(r["tier"] for r in results)
    assert _int(README, r"\*\*(\d+)\*\* are hand-written the way", "README handwritten") == tiers["handwritten"]
    assert _int(README, r"\*\*(\d+)\*\* are generated to sweep", "README grid") == tiers["grid"]
    assert _int(README, r"and \*\*(\d+)\*\* ask what can be \*sequenced\*", "README chain") == tiers["chain"]


@pytest.mark.parametrize("config,column,pattern", [
    ("fusion", "mean_recall_at_k", r"\| \*\*Fusion RAG\*\* \| \*\*([\d.]+)\*\*"),
    ("fusion", "mean_normalized_recall", r"\| \*\*Fusion RAG\*\* \| \*\*[\d.]+\*\* \| \*\*([\d.]+)\*\*"),
    ("fusion", "mean_archetype_precision", r"\| \*\*Fusion RAG\*\* \| \*\*[\d.]+\*\* \| \*\*[\d.]+\*\* \| \*\*([\d.]+)\*\*"),
    ("hybrid", "mean_recall_at_k", r"\| Hybrid RAG \| ([\d.]+)"),
    ("hybrid", "mean_normalized_recall", r"\| Hybrid RAG \| [\d.]+ \| ([\d.]+)"),
    ("hybrid", "mean_archetype_precision", r"\| Hybrid RAG \| [\d.]+ \| [\d.]+ \| ([\d.]+)"),
    ("standard", "mean_archetype_precision", r"\| Standard prompting \| [\d.]+ \| [\d.]+ \| ([\d.]+)"),
])
def test_readmes_headline_table_matches_the_committed_summary(config, column, pattern):
    """README's table is the first thing a reader sees, and it is a copy.

    Rounded to three decimals, which is what the table prints -- the CSV
    carries more and the point is that the copy tracks the source, not that
    the reader is shown every digit.
    """
    quoted = re.findall(pattern, README)
    assert len(quoted) == 1, f"README table row for {config} did not match {pattern!r}"
    assert float(quoted[0]) == round(float(_summary()[config][column]), 3)


def test_the_report_and_the_readme_do_not_disagree_about_hybrids_precision():
    """§3.5's closing bullet quoted a precision from two revisions earlier.

    It read `0.967 vs 0.693` beside a table on the same page reading 0.659.
    Two numbers for one measurement in one document is the failure this whole
    file exists to catch, so it gets its own test.
    """
    hybrid = round(float(_summary()["hybrid"]["mean_archetype_precision"]), 3)
    assert _figure(REPORT, r"archetype precision \(0\.967 vs ([\d.]+)\)",
                   "REPORT closing precision") == f"{hybrid:.3f}"


def test_the_hallucination_figures_the_readme_quotes_match_the_committed_run():
    """Place mentions and the ungrounded rate, per config, from #132's CSV."""
    rows = {r["config"]: r for r in _rows(EVAL / "hallucination_summary.csv")}
    total_names = sum(int(r["places_named"]) for r in rows.values())
    assert _int(README, r"across ([\d,]+) place mentions", "README place mentions") == total_names
    quoted = re.findall(r"while ([\d.]+)% / ([\d.]+)% / ([\d.]+)% of named places", README)
    assert len(quoted) == 1, "README's ungrounded-rate sentence was rewritten"
    for got, config in zip(quoted[0], ("fusion", "hybrid", "standard")):
        assert float(got) == round(float(rows[config]["ungrounded_mention_rate"]) * 100, 1)


# ------------------------------------------------------------- the test count

def test_every_place_that_states_the_suite_size_states_the_same_number():
    """Four files quote it and no committed file holds it.

    So this cannot be checked against a source the way the figures above can;
    what it can check is that the four agree. They did not: REPORT §1 said 235,
    REPORT §4 and CLAUDE.md said 241, `pytest.ini` said 161, and the CI comment
    said 235. Any change to the suite size now has to touch all four together
    or this fails, which is the property that was missing.

    Re-derive the real numbers with `pytest tests/ --collect-only` and
    `pytest tests/ -m slow --collect-only`.
    """
    total = {
        "REPORT §1": _int(REPORT, r"`roamwise/tests/`, (\d+) collected", "REPORT §1 total"),
        "REPORT §4": _int(REPORT, r"(\d+) passing \(1 skipped\) out of \d+ collected", "REPORT §4 passing") + 1,
        "REPORT §4 collected": _int(REPORT, r"\d+ passing \(1 skipped\) out of (\d+) collected", "REPORT §4 total"),
        "CLAUDE.md": _int(CLAUDE_MD, r"(\d+) tests carry", "CLAUDE.md total"),
        "pytest.ini": _int(PYTEST_INI, r"of the (\d+)\n        tests carry it", "pytest.ini total"),
        "workflow": _int(WORKFLOW, r"# all (\d+)\. Nothing reaches", "workflow total"),
    }
    assert len(set(total.values())) == 1, f"the suite size is quoted inconsistently: {total}"


def test_every_place_that_states_the_slow_count_states_the_same_number():
    slow = {
        "REPORT §4": _int(REPORT, r"of which (\d+) carry `@pytest.mark.slow`", "REPORT slow"),
        "CLAUDE.md": _int(CLAUDE_MD, r"\*\* (\d+) of the \d+ tests carry", "CLAUDE.md slow"),
        "pytest.ini": _int(PYTEST_INI, r"analysis\. (\d+) of the", "pytest.ini slow"),
    }
    assert len(set(slow.values())) == 1, f"the slow-test count is quoted inconsistently: {slow}"


def test_claude_md_still_describes_backlogs_size_and_scope():
    """CLAUDE.md tells a reader not to open BACKLOG.md, and says how big it is.

    That size was 81 KB against a 93 KB file when the cleanup pass measured it,
    and the file has since been synced to 112 KB. The tolerance is 5 KB: an
    ordinary edit should not turn this red, a migration should. The issue count
    is exact -- it comes from `gh issue list --state all`, and if it moves the
    sentence promising a full sync is no longer true.
    """
    backlog = REPO / "BACKLOG.md"
    stated_kb = _int(CLAUDE_MD, r"it is (\d+) KB", "CLAUDE.md BACKLOG size")
    actual_kb = backlog.stat().st_size / 1024
    assert abs(stated_kb - actual_kb) <= 5, (
        f"CLAUDE.md says BACKLOG.md is {stated_kb} KB; it is {actual_kb:.0f} KB")

    stated_issues = _int(CLAUDE_MD, r"all\n(\d+) issues, none open", "CLAUDE.md issue count")
    headings = re.findall(r"^### .*$", backlog.read_text(encoding="utf-8"), re.M)
    covered = {n for h in headings for n in re.findall(r"#(\d+)", h)}
    assert len(covered) == stated_issues, (
        f"CLAUDE.md claims {stated_issues} issues are written up; BACKLOG.md has sections for "
        f"{len(covered)}. Re-sync it, or say it lags again.")
