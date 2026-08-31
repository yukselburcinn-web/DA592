"""The comparative analysis behind the Results tab: the query set it runs, the
answer key it is graded against, and the paired significance test that decides
whether a lead is real.

Split out of `test_orchestration.py` in #164. These test
`evaluation/comparative_analysis.py`, not the orchestrators -- and leaving them
there put the measurement work and the orchestrator work in one file.
"""

import collections
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

from roamwise.evaluation import comparative_analysis
from roamwise.evaluation.comparative_analysis import (
    TEST_QUERIES,
    dependence_level,
    gold_for,
    paired_significance,
    run_comparative_analysis,
    summarize,
)
from roamwise.knowledge_graph.build_graph import CATEGORY_AFFINITY, GraphIndex
from roamwise.retrieval.graph_search import GraphSearchIndex
from roamwise.tests.helpers import DATA_DIR, MAIN_CITY, city_with_category


@pytest.mark.slow
def test_comparative_analysis_shows_fusion_advantage():
    df = run_comparative_analysis(top_k=8)
    summary = summarize(df)
    assert summary.loc["fusion", "mean_archetype_precision"] >= summary.loc["standard", "mean_archetype_precision"]
    assert summary.loc["fusion", "mean_structural_grounding_rate"] == 1.0
    assert summary.loc["standard", "mean_structural_grounding_rate"] == 0.0

    # The headline on the Results tab is generated from these verdicts, so a
    # metric silently changing category would rewrite a user-facing claim.
    verdicts = paired_significance(df).set_index(["metric", "opponent"]).verdict
    assert verdicts[("archetype_precision", "hybrid")] == "better"
    # Grounded-entity rate is 1.0 by construction for anything that retrieves,
    # so it can separate Fusion from standard prompting but never from Hybrid.
    assert verdicts[("structural_grounding_rate", "hybrid")] == "identical"
    assert verdicts[("structural_grounding_rate", "standard")] == "better"


def test_every_test_query_has_an_answer():
    """A query whose gold set is empty scores nothing and silently shrinks the
    sample. One shipped that way: Lisbon has four beach POIs and none within
    reach of a hub, so 'beach spots reachable from the airport' was graded
    against an empty key for every configuration (issue #50)."""
    idx = GraphIndex()
    empty = [q.text for q in TEST_QUERIES if not gold_for(idx, q)]
    assert not empty, f"queries with no possible correct answer: {empty}"


def test_query_set_is_powered_and_balanced_by_difficulty():
    """The comparison was run on 18 queries and at that size a real effect is
    detected about a third of the time, so the set has a floor.

    The second half used to be about circularity -- the answer key was the
    retriever's own traversal, so a query naming its category graded the
    retriever against itself. That is fixed at the source (see
    `test_the_answer_key_is_not_one_the_retriever_can_produce`), and what the
    split now guards is difficulty: a query that names its category tells the
    router where to look, and a headline number must not be carried entirely
    by that easy half (#48).
    """
    assert len(TEST_QUERIES) >= 45
    assert {q.tier for q in TEST_QUERIES} == {"handwritten", "grid", "chain"}

    # The chain tier is graded against a key the traversal itself defines,
    # gated on Wikivoyage (#126, K1). That gate is what stops recall measuring
    # self-agreement, but the level is still the weakest of the four, so the
    # tier has to stay a minority of the set -- a headline carried by
    # traversal-defined keys would be the circularity #48 removed, returning
    # under a new name.
    chain = [q for q in TEST_QUERIES if q.tier == "chain"]
    assert chain, "the relation the graph exists for should be measured"
    assert len(chain) / len(TEST_QUERIES) < 0.15

    name_no_category = [q for q in TEST_QUERIES if dependence_level(q) != "subset"]
    assert len(name_no_category) >= 30

    # No archetype may own the set the way Culture Enthusiast owned 44% of it.
    counts = collections.Counter(q.archetype for q in TEST_QUERIES)
    assert max(counts.values()) / len(TEST_QUERIES) < 0.40
    assert set(counts) == set(CATEGORY_AFFINITY), "every archetype should be exercised"


def test_dependence_level_spots_the_queries_that_name_their_own_category():
    query = comparative_analysis.TestQuery
    # Names the key's category next to a transport word, so the graph router
    # dispatches straight at it -- the easiest shape of query there is.
    assert dependence_level(query(MAIN_CITY, "Culture Enthusiast", ("landmark",), True,
                                  "landmarks near a transport hub")) == "subset"
    # Same category, no transport constraint -- the router's filter is wider
    # than the key rather than nested inside it.
    assert dependence_level(query(MAIN_CITY, "Culture Enthusiast", ("landmark",), False,
                                  "the best landmarks to visit")) == "superset"
    # The router never reaches for the key's category at all.
    assert dependence_level(query(MAIN_CITY, "Culture Enthusiast", ("history",), False,
                                  "somewhere calm to spend a slow afternoon")) == "independent"


def test_the_answer_key_is_not_one_the_retriever_can_produce():
    """`recall_at_k`'s answer key used to be every catalogue POI of the queried
    category -- `GraphIndex.city_pois`, which is the traversal the graph
    retriever dispatches to. On a query naming its category, retrieval was a
    subset of the key by construction, so its recall measured that it agrees
    with itself rather than that it finds good answers (#48).

    The key is now gated on Wikivoyage, which is written by travellers and owes
    nothing to Wikidata sitelinks, OSM tagging or this project's graph. The
    property that matters is that neither set contains the other: the retriever
    can return plenty of category matches and still score zero.
    """
    idx = GraphIndex()
    graph = GraphSearchIndex(idx)
    city = city_with_category("museum")
    if city is None:
        pytest.skip("no city holds museums")

    query = comparative_analysis.TestQuery(
        city, "Culture Enthusiast", ("museum",), True,
        "museums within easy reach of a transport hub")
    assert dependence_level(query) == "subset", "this test needs the easiest query shape"

    gold = gold_for(idx, query)
    assert gold, "the key must not be empty"

    # The key is a strict subset of the category, so something outside this
    # project decided which members are in it.
    every_museum = {p["poi_id"] for p in idx.city_pois(city, category="museum")}
    assert gold < every_museum, \
        "the key is every POI of the category again -- the circularity is back"

    # And the retriever's own traversal is not contained in it.
    retrieved = {r["poi_id"] for r in graph.search(query.text, top_k=40,
                                                   destination_id=city,
                                                   archetype=query.archetype)}
    assert retrieved - gold, \
        "graph retrieval falls entirely inside the key -- it is grading itself"


def test_every_recommended_poi_is_a_poi_the_catalogue_still_holds():
    """The key is committed rather than recomputed, so it can go stale against
    a catalogue that dropped rows -- which #65 did, by 46."""
    catalogue = set(pd.read_csv(DATA_DIR / "poi.csv").poi_id)
    orphans = comparative_analysis.RECOMMENDED_POIS - catalogue
    assert not orphans, (
        f"{len(orphans)} POIs in retrieval_gold.csv are no longer in the catalogue; "
        "rebuild with `cd roamwise/pipeline && python retrieval_gold.py --write`")


def _synthetic_pairs(**per_config_values) -> pd.DataFrame:
    """Long-form results frame with one row per (query, config)."""
    rows = []
    for config, columns in per_config_values.items():
        for query_id in range(10):
            rows.append({"query_id": query_id, "config": config,
                         **{column: values[query_id] for column, values in columns.items()}})
    return pd.DataFrame(rows)


def test_paired_significance_reads_direction_and_ties():
    """Wilcoxon cannot rank an all-zero difference vector, and half the metrics
    are better when *lower*. Both used to be latent ways to either crash the
    Results tab or print a backwards verdict (issue #46)."""
    df = _synthetic_pairs(
        fusion={"archetype_precision": [0.9] * 10, "retrieval_ms": [20.0] * 10,
                "structural_grounding_rate": [1.0] * 10, "km_per_stop_day1": [1.0] * 10,
                "recall_at_k": [0.5] * 10},
        hybrid={"archetype_precision": [0.5] * 10, "retrieval_ms": [5.0] * 10,
                "structural_grounding_rate": [1.0] * 10, "km_per_stop_day1": [1.0] * 10,
                "recall_at_k": [0.5] * 10},
    )
    verdicts = paired_significance(df, champion="fusion").set_index("metric")

    assert verdicts.loc["archetype_precision", "verdict"] == "better"
    # Higher latency is worse even though the number went up -- the sign has to
    # be flipped for lower-is-better metrics or this reads as a win.
    assert verdicts.loc["retrieval_ms", "verdict"] == "worse"
    assert verdicts.loc["retrieval_ms", "mean_advantage"] < 0
    # Identical columns: no test is possible, and it must not raise.
    assert verdicts.loc["structural_grounding_rate", "verdict"] == "identical"
    assert pd.isna(verdicts.loc["structural_grounding_rate", "p_value"])


def test_paired_significance_calls_a_coin_flip_no_difference():
    """A metric that trades wins back and forth must not be reported as a lead
    just because its mean happens to land higher."""
    df = _synthetic_pairs(
        fusion={"km_per_stop_day1": [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]},
        hybrid={"km_per_stop_day1": [2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.1]},
    )
    verdict = paired_significance(df, champion="fusion").set_index("metric")
    assert verdict.loc["km_per_stop_day1", "verdict"] == "no difference"


# --- issue #143: the quota exponent is a pool knob, not a plan knob ---

def test_the_quota_sweeps_cover_the_exponent_the_code_ships():
    """#143's argument only holds while the measurement describes this code.

    Two committed sweeps decide `PREFERENCE_QUOTA_EXPONENT`: one over the
    reachable pool (`quota_topk_sweep.csv`) and one over the plans built from
    it (`quota_plan_impact.csv`). The conclusion -- that the knob moves the
    pool and not the itinerary -- is only readable if the shipped value is
    actually one of the rows. Change the constant without re-running them and
    the constant's own comment starts citing numbers measured for a different
    setting, which is how #123's shipped choice quietly stopped being optimal
    once the catalogue moved underneath it.
    """
    from roamwise.knowledge_graph.build_graph import PREFERENCE_QUOTA_EXPONENT

    here = Path(comparative_analysis.__file__).parent
    for name in ("quota_topk_sweep.csv", "quota_plan_impact.csv"):
        path = here / name
        assert path.exists(), f"{name} is the measurement behind the constant; regenerate it"
        swept = set(pd.read_csv(path).exponent)
        assert PREFERENCE_QUOTA_EXPONENT in swept, (
            f"{name} does not cover the shipped exponent "
            f"{PREFERENCE_QUOTA_EXPONENT} (has {sorted(swept)}) -- re-run "
            f"`python -m roamwise.evaluation.{name[:-4]}`")


@pytest.mark.slow
def test_the_quota_exponent_does_not_change_what_a_traveller_is_shown():
    """The finding #143 shipped, guarded (`quota_plan_impact.csv`).

    Over 84 plans -- 2 cities x 7 archetypes x 3 trip lengths -- moving the
    exponent from 0.5 to 0.6 changes the aggregate stop count by zero, because
    the POIs it moves sit in the tail the router never selects: a one-day plan
    takes eight stops from twenty-four candidates. That is the whole reason
    #123's unmet acceptance criterion was not chased with a code change.

    Asserted with room, not exactly: the two arms differ on a handful of
    individual trips in both directions, and pinning the sum to 0 would make
    this test fail on a catalogue change that means nothing. What must not
    happen is the pool knob quietly becoming a plan knob.
    """
    impact = pd.read_csv(Path(comparative_analysis.__file__).parent / "quota_plan_impact.csv")
    totals = impact.groupby("exponent").stops.sum()
    assert len(totals) == 2, "the measurement compares exactly two exponents"

    spread = totals.max() - totals.min()
    assert spread <= 2, (
        f"the quota exponent moved {spread} stops across {len(impact) // 2} plans; it is "
        f"documented as reaching none, so either the finding or the constant needs revisiting")


# --- issue #144: the chain threshold was decided, not just measured ---

def test_the_chain_sweep_covers_the_threshold_the_code_ships():
    """#144 rejected 15 minutes on three measurements at once; the file that
    holds them has to describe this code.

    `REACHABLE_MAX_MIN` is argued for in its own comment with numbers from
    `chain_threshold_weight_sweep.csv` -- the KN-2 band, the catalogue share
    and the chain tier's discrimination ratio. Move the constant without
    re-running the sweep and that comment starts citing a configuration nobody
    measured, which is exactly the state #126 left the threshold in.
    """
    from roamwise.retrieval.graph_search import REACHABLE_MAX_MIN

    path = Path(comparative_analysis.__file__).parent / "chain_threshold_weight_sweep.csv"
    assert path.exists(), "the sweep is the argument for the threshold; regenerate it"
    swept = pd.read_csv(path)
    assert REACHABLE_MAX_MIN in set(swept.reachable_max_min), (
        f"the sweep does not cover the shipped threshold {REACHABLE_MAX_MIN} "
        f"(has {sorted(set(swept.reachable_max_min))}) -- re-run "
        f"`python -m roamwise.evaluation.chain_threshold_weight_sweep`")

    # The finding itself: at the shipped threshold some weight holds KN-2's
    # 2-4 band, and at 15 minutes none does. That asymmetry is the decision.
    shipped = swept[swept.reachable_max_min == REACHABLE_MAX_MIN]
    assert (shipped.kn2_band == "in band").any(), (
        "no weight keeps the chain inside KN-2's band at the shipped threshold; "
        "the threshold or RETRIEVER_WEIGHTS['chain'] needs revisiting")


# --- issue #132: the hallucination rate, measured rather than proxied ---
#
# `structural_grounding_rate` (this file's `mean_structural_grounding_rate`
# assertions above) is 1.0 by construction for anything that retrieves, so it
# cannot be the hallucination number the proposal asks for. These cover the
# measurement that replaces it. None of them call a model: the grading is pure
# text work, and the one thing that must happen when no model is configured is
# a refusal.

from roamwise.agents.llm_client import TemplateLLMClient  # noqa: E402
from roamwise.evaluation.hallucination import (  # noqa: E402
    TemplateClientRefused,
    build_prompts,
    classify,
    gazetteer,
    places_named,
    places_pattern,
    run_hallucination_measurement,
    run_zero_context_probe,
)

SAMPLE_RUN = Path(__file__).resolve().parents[1] / "evaluation" / "local_llm_sample_run.md"


@pytest.fixture(scope="module")
def gaz():
    places = gazetteer()
    return places, places_pattern(places)


def test_place_names_match_on_word_boundaries_not_substrings(gaz):
    """CLAUDE.md gotcha 8, and the exact flaw in the probe this replaces.

    The old matcher was `any(name in line or line in name for name in
    known_names)`. The second direction is the worse one: a short line matches
    many long names at once, so the match rate inflates and the hallucination
    number comes out *lower* than the truth -- a measurement that fails safe in
    the direction that flatters the system.
    """
    places = {"bar": {"name": "Bar", "cities": {"BER"}},
              "station": {"name": "Station", "cities": {"BER"}}}
    pattern = places_pattern(places)

    assert places_named("a border barrier near a television station-house", places, pattern) == []
    assert places_named("we ate at Bar, by the Station.", places, pattern) == ["bar", "station"]


def test_the_longest_matching_name_wins(gaz):
    """"Museum of Asian Art" is one place, not "Asian Art" inside something."""
    places, pattern = gaz
    named = places_named("Then the Museum of Asian Art, part of the Humboldt Forum.",
                         places, pattern)
    assert [places[k]["name"] for k in named] == ["Museum of Asian Art", "Humboldt Forum"]


def test_every_routed_stop_is_recovered_from_a_real_generated_narrative(gaz):
    """The extractor is checked against prose a model actually wrote.

    `evaluation/local_llm_sample_run.md` is a committed, hand-checked run: 11
    routed stops, all 11 named in the narrative. Recovering fewer than 11 would
    mean the extractor misses real names, which inflates the rate's denominator
    error in both directions. The hard cases are all in there -- a name ending
    in a period ("Kila."), one with initials ("F. W. Borchardt"), one with a
    comma inside it ("Academy of Arts, Berlin"), one in caps ("BLOCK HOUSE"),
    and one the model expanded mid-sentence ("Wall Museum - Museum Haus am
    Checkpoint Charlie").
    """
    import re
    places, pattern = gaz
    text = SAMPLE_RUN.read_text()
    narrative = text.split("## Final plan (raw LLM output)")[1]
    routed = re.findall(r"^- Day \d: (.+?) \[", text, re.M)
    assert len(routed) == 11, "fixture changed; re-check the counts in the file's header"

    found = {places[k]["name"] for k in places_named(narrative, places, pattern)}
    assert set(routed) <= found


def test_a_place_the_prompt_introduced_is_not_counted_as_invention(gaz):
    """The Humboldt Forum's own description says it stands on Museum Island.

    A narrative that repeats that is describing its stop correctly. Counting it
    would make the metric fire hardest on good output -- the same distinction
    `test_orchestration.ungrounded_places` draws.
    """
    places, pattern = gaz
    prompt = "Day 1: Humboldt Forum -- a cultural centre on Museum Island."
    graded = classify("Visit the Humboldt Forum, on Museum Island.", prompt, "BER", places, pattern)

    assert graded["unshown_same_city"] == 0
    assert graded["geographical_hallucination_rate"] == 0.0
    assert graded["ungrounded_mention_rate"] == 0.0


def test_a_place_from_the_other_city_is_geographical_hallucination(gaz):
    """The proposal's own term: a real place, named for the wrong city."""
    places, pattern = gaz
    prompt = "Day 1: Humboldt Forum."
    graded = classify("Visit the Humboldt Forum, then the Eiffel Tower.",
                      prompt, "BER", places, pattern)

    assert graded["wrong_city"] == 1
    assert graded["wrong_city_names"] == "Eiffel Tower"
    assert graded["geographical_hallucination_rate"] == 0.5


def test_a_narrative_that_names_no_place_is_not_a_zero_percent_score(gaz):
    """An empty denominator is no measurement, not a perfect one."""
    places, pattern = gaz
    graded = classify("A pleasant day out.", "Day 1: Humboldt Forum.", "BER", places, pattern)

    assert graded["places_named"] == 0
    assert graded["geographical_hallucination_rate"] is None
    assert graded["ungrounded_mention_rate"] is None


@pytest.mark.parametrize("run", [run_hallucination_measurement, run_zero_context_probe])
def test_the_measurement_refuses_to_run_without_a_model(run):
    """A template run would report 0.0 hallucination -- a perfect score earned
    by having no model. #133 made the fallback audible; for this one number it
    has to be fatal, because the CSV it would write is indistinguishable from a
    genuine result."""
    with pytest.raises(TemplateClientRefused):
        run(llm=TemplateLLMClient())


@pytest.mark.slow
def test_generating_once_per_distinct_prompt_covers_every_row():
    """The dedup the quota budget rests on, asserted rather than assumed.

    Identical prompt means identical generation, so one call per distinct
    prompt reconstructs what a call per row would report -- provided every row
    maps to a prompt that was actually generated. Standard prompting is the
    clearest case: it retrieves nothing, so its candidate set does not depend
    on the question and 67 queries collapse onto city x archetype.
    """
    prompts = build_prompts()

    assert len(prompts) == len(TEST_QUERIES) * 3
    assert prompts.prompt_key.nunique() < len(prompts), "no dedup means no saving to claim"
    assert set(prompts.prompt_key) == set(prompts.groupby("prompt_key").prompt_key.first())

    standard = prompts[prompts.config == "standard"]
    cities = standard.destination_id.nunique()
    archetypes = standard.archetype.nunique()
    assert standard.prompt_key.nunique() == cities * archetypes


def test_the_committed_hallucination_numbers_describe_today_s_measurement():
    """The guard on a measurement CI cannot re-derive.

    `hallucination_summary.csv` is produced by a run against a live endpoint,
    so no test can regenerate it and nothing else would notice the day
    TEST_QUERIES -- or the prompt those queries become -- changes and leaves
    the committed numbers describing something that no longer exists.
    `data/knowledge_graph.gml` went stale for nineteen commits for exactly
    this reason (#145, CLAUDE.md gotcha 16).

    When this goes red the fix is to re-run the measurement:
    `set -a; source .env; set +a; python -m roamwise.evaluation.hallucination`.
    It is cheaper than it looks -- generations are cached per prompt, so only
    the queries whose prompt actually changed cost anything.
    """
    from roamwise.evaluation.hallucination import SUMMARY_CSV, measurement_fingerprint

    summary = pd.read_csv(SUMMARY_CSV)
    # `.get`, not `summary.fingerprint`: the column was `query_set` until #173
    # widened the hash to cover the prompt as well. A summary written before
    # that rename has to fail this assertion with its message, not crash with
    # a KeyError two lines above it.
    committed = set(summary.get("fingerprint", pd.Series(dtype=str)).astype(str))
    assert committed == {measurement_fingerprint()}, (
        f"hallucination_summary.csv'nin parmak izi {committed}, bugunku parmak izi "
        f"{measurement_fingerprint()} -- olcumu yeniden kostur")


def test_the_fingerprint_moves_when_the_prompt_the_narrator_sees_moves():
    """The half of the guard #173 had to add, and why it is not enough on its
    own to say "the fingerprint covers the prompt".

    #173 changed `RouterAgent._facts()` and touched no query. The old
    fingerprint hashed only TEST_QUERIES, so it would not have moved, the
    guard above would have stayed green, and the committed CSVs would have
    gone on reporting numbers measured against a prompt the code no longer
    builds -- the `knowledge_graph.gml` story again, one file over.

    Two things are asserted, because either alone is satisfiable by an empty
    list: that the real prompt-building code is what gets hashed, and that
    changing any of it changes the hash.
    """
    from roamwise.agents.orchestrator import SYNTHESIS_SYSTEM
    from roamwise.evaluation import hallucination

    sources = hallucination.prompt_sources()
    assert SYNTHESIS_SYSTEM in sources
    for signature in ("def synthesis_prompt(", "def _facts(", "def _leg_clause("):
        assert any(signature in s for s in sources), f"{signature} is not fingerprinted"

    before = hallucination.measurement_fingerprint()
    for i in range(len(sources)):
        edited = list(sources)
        edited[i] += "  # a comment is enough to invalidate a cached run"
        with mock.patch.object(hallucination, "prompt_sources", lambda e=edited: e):
            assert hallucination.measurement_fingerprint() != before, (
                f"prompt_sources()[{i}] is hashed but does not move the fingerprint")


def test_a_catalogue_name_that_is_also_a_common_word_needs_its_capital(gaz):
    """Word boundaries do not help when the whole word is the name.

    Paris holds a POI called "Nation"; Berlin holds a "Bunker", an "Eden", a
    "Matrix" and a "Tresor". Matched case-insensitively, ordinary prose scored
    a hallucination: "the heart of the nation's capital" in a Berlin narrative
    came back `wrong_city=1` naming a Paris place -- a false positive on the
    headline metric, produced by a sentence that names nothing at all. Found in
    review of #132; the committed run was unaffected, which is exactly why it
    needed a test rather than a re-measurement.
    """
    places, pattern = gaz
    prompt = "Day 1: Brandenburg Gate -- a landmark."

    prose = classify("The Brandenburg Gate stands at the heart of the nation's capital.",
                     prompt, "BER", places, pattern)
    assert prose["wrong_city"] == 0
    assert prose["ungrounded_mention_rate"] == 0.0

    # The proper noun still has to be caught, or the fix would be a mute button.
    named = classify("Then on to Nation, in Paris.", prompt, "BER", places, pattern)
    assert "Nation" in named["wrong_city_names"]


def test_two_places_on_one_line_do_not_hide_an_unmatched_one(gaz):
    """The probe counts lines, so its denominator has to be lines.

    `len(lines) - len(distinct places)` mixed two denominators: a line naming
    two known places cancelled out a line naming an invented one, and three
    would have made the column negative.
    """
    places, pattern = gaz
    text = "Brandenburg Gate and Berlin Cathedral\nFernsehturm Berlin\nSome Invented Place"
    lines = [l for l in text.splitlines() if l.strip()]

    matched_lines = sum(1 for line in lines if places_named(line, places, pattern))

    assert len(lines) - matched_lines == 1, "uydurma satir gorunmez kaldi"


# --- issue #124: how much of the personalization is the synthetic survey? ---

def test_the_resampler_reproduces_the_shipped_survey_s_distribution():
    """`survey_sensitivity.draw_survey` is a copy of `generate_data.py:262-281`,
    and a copy is a thing that drifts.

    It cannot import the original: that module writes five shipped files at
    import time and leaves the app unable to start (CLAUDE.md). So the copy is
    checked against the artefact the original produced instead -- same shape,
    same per-archetype count, and per-archetype means within sampling error of
    the committed ones. If someone retunes `ARCHETYPES` in `generate_data.py`
    and regenerates `user_survey.csv`, this goes red and the measurement's
    baseline stops being fictional.

    The tolerance is five standard errors of the *difference* between two
    independent 60-row means -- sigma * sqrt(2/60) = 0.0146, so 0.073. The
    sqrt(2) matters: both sides of this comparison are sampled, and using the
    standard error of one mean (0.0103) makes the bound too tight for the
    largest of 42 cells to stay under, which is how this test failed when it
    was first written. Loose enough that a fresh draw never trips it, tight
    enough that a moved centre always does.
    """
    import numpy as np

    from roamwise.evaluation.survey_sensitivity import (
        ARCHETYPE_CENTRES, PREFERENCE_DIMS, ROWS_PER_ARCHETYPE, SHIPPED_SIGMA, draw_survey)
    from roamwise.optimization.scoring import DATA_DIR

    shipped = pd.read_csv(DATA_DIR / "user_survey.csv")
    drawn = draw_survey(np.random.default_rng(0))

    assert drawn.shape == shipped.shape
    assert set(drawn.archetype) == set(shipped.archetype)
    assert set(drawn.archetype.value_counts()) == {ROWS_PER_ARCHETYPE}
    assert set(shipped.archetype.value_counts()) == {ROWS_PER_ARCHETYPE}

    tolerance = 5 * SHIPPED_SIGMA * np.sqrt(2 / ROWS_PER_ARCHETYPE)
    for archetype in ARCHETYPE_CENTRES:
        theirs = shipped[shipped.archetype == archetype][PREFERENCE_DIMS].mean()
        ours = drawn[drawn.archetype == archetype][PREFERENCE_DIMS].mean()
        assert (theirs - ours).abs().max() < tolerance, (
            f"{archetype}: the resampler and the committed survey disagree by more "
            f"than sampling error -- generate_data.ARCHETYPES has moved")


def test_the_sensitivity_control_moves_what_resampling_does_not():
    """The guard on the measurement's own ability to see anything.

    #124 can conclude "the survey does not matter" only if the same machinery
    says "the survey matters" when it demonstrably does. `shuffle` is that
    case: the feature rows are untouched and only the label association is
    destroyed, which is the one thing both consumers read. If shuffling leaves
    the matrix's rankings intact, every insensitive row above it in the table
    is unreadable and the conclusion has to be withdrawn.

    Asserted together with the resample, because the pair is the claim: one
    moves, the other does not, and the same code produced both.
    """
    import numpy as np

    from roamwise.evaluation.survey_sensitivity import (
        draw_survey, matrix_deltas, shuffle_labels)
    from roamwise.optimization.scoring import PREFERENCE_CATEGORY_WEIGHTS, _fit_preference_matrix

    rng = np.random.default_rng(0)
    survey = draw_survey(rng)
    resampled = matrix_deltas(PREFERENCE_CATEGORY_WEIGHTS, _fit_preference_matrix(survey))
    shuffled = matrix_deltas(PREFERENCE_CATEGORY_WEIGHTS,
                             _fit_preference_matrix(shuffle_labels(rng, survey)))

    assert resampled["matrix_top1_agreement"] == 1.0, \
        "a fresh draw from the same centres reranked a category -- the baseline is not stable"
    assert shuffled["matrix_top1_agreement"] < 0.5, \
        "shuffling the labels barely moved the matrix -- this measurement cannot see dependence"
    assert shuffled["matrix_mean_abs_delta"] > 10 * resampled["matrix_mean_abs_delta"]


def test_swapping_the_preference_matrix_restores_both_globals():
    """`preference_match` reads the matrix and the normaliser derived from it
    as two separate module globals, so a swap that moves one and not the other
    scores the new weights against the old maximum. The context manager has to
    set both and put both back -- including when the block raises, since a
    measurement that dies mid-variant would otherwise leave the process
    scoring every later plan with a survey nobody asked for."""
    from roamwise.evaluation.survey_sensitivity import preference_matrix
    from roamwise.optimization import scoring

    shipped, shipped_max = scoring.PREFERENCE_CATEGORY_WEIGHTS, scoring._MAX_MATCH
    doubled = {d: {c: 2 * w for c, w in row.items()} for d, row in shipped.items()}

    with preference_matrix(doubled):
        assert scoring.PREFERENCE_CATEGORY_WEIGHTS == doubled
        assert scoring._MAX_MATCH == pytest.approx(2 * shipped_max)

    assert scoring.PREFERENCE_CATEGORY_WEIGHTS is shipped
    assert scoring._MAX_MATCH == shipped_max

    with pytest.raises(RuntimeError):
        with preference_matrix(doubled):
            raise RuntimeError("the variant blew up")
    assert scoring.PREFERENCE_CATEGORY_WEIGHTS is shipped
    assert scoring._MAX_MATCH == shipped_max


def test_the_committed_sensitivity_table_covers_the_perturbations_the_module_defines():
    """#124's conclusion is only readable while the committed CSV describes the
    perturbations the code runs today. Add a sigma level or drop the control
    and forget to re-run, and REPORT quotes a table with a missing row --
    the `quota_topk_sweep.csv` failure mode one measurement over."""
    from roamwise.evaluation.survey_sensitivity import SUMMARY_CSV, variants

    assert SUMMARY_CSV.exists(), (
        "survey_sensitivity.csv is the measurement behind REPORT 5's survey item; "
        "run `python -m roamwise.evaluation.survey_sensitivity`")
    committed = set(pd.read_csv(SUMMARY_CSV).variant)
    defined = {label for label, _, _ in variants()}
    assert committed == defined, (
        f"survey_sensitivity.csv olculdugu varyant seti {sorted(committed - defined)} "
        f"fazla / {sorted(defined - committed)} eksik -- olcumu yeniden kostur")


# --- issue #175: a cell whose answer key holds one POI is not a measurement ---

def test_no_query_is_graded_against_a_key_too_small_to_grade_it():
    """The guard `MIN_GOLD` exists to be. Goes red when the catalogue or the
    Wikivoyage gate moves a cell under the threshold, which is the event that
    put three BER/food queries into the table scoring 0.00 with nothing
    anywhere saying why -- the same class of silent staleness as #145's graph
    export and #132's query-set fingerprint.

    Asserted on the assembled set rather than on the builders: the filter used
    to sit inside `build_grid_queries` and covered one tier of three, so a
    hand-written query with a one-POI key walked past it.
    """
    from roamwise.evaluation.comparative_analysis import MIN_GOLD, TEST_QUERIES, gold_for

    idx = GraphIndex()
    thin = [(q.destination_id, q.categories, q.tier, len(gold_for(idx, q)))
            for q in TEST_QUERIES if len(gold_for(idx, q)) < MIN_GOLD]
    assert not thin, (
        f"queries kept with a key smaller than MIN_GOLD={MIN_GOLD}: {thin} -- "
        f"recall@k cannot grade them")


def test_the_queries_the_threshold_drops_are_published_rather_than_vanishing():
    """A query that disappears silently hides a hole in the answer key; a query
    that silently scores 0.00 reports that hole as a retrieval failure. #175 is
    about the second, and this is the assertion that fixing it did not create
    the first.

    Each reject carries the size of the key it was rejected for, so the reader
    can tell a cell that holds three from a cell that holds none.
    """
    from roamwise.evaluation.comparative_analysis import (
        MIN_GOLD, UNDERCOVERED_QUERIES, gold_coverage)

    for query, size in UNDERCOVERED_QUERIES:
        assert size < MIN_GOLD
        assert query.destination_id and query.tier

    # Whatever was dropped has to be explainable from the coverage table -- the
    # cell it came from must be one the key covers thinly. Without this the two
    # halves of #175 could drift into disagreeing about which cells are thin.
    if UNDERCOVERED_QUERIES:
        coverage = gold_coverage().set_index(["destination_id", "category"])
        for query, _ in UNDERCOVERED_QUERIES:
            for category in query.categories or ():
                cell = coverage.loc[(query.destination_id, category)]
                assert cell.coverage_pct < coverage.coverage_pct.median(), (
                    f"{query.destination_id}/{category} produced an ungradeable query "
                    f"but reads as well covered ({cell.coverage_pct}%)")


def test_the_committed_comparison_was_measured_on_the_queries_the_code_keeps():
    """`comparative_analysis_results.csv` is the table REPORT 3.4 quotes, and
    #175 changed which queries reach it. A committed CSV holding a query the
    threshold now rejects is reporting a number the code no longer produces --
    and unlike the hallucination measurement this one CI *could* re-derive, so
    the guard is cheap to satisfy: re-run the module."""
    from roamwise.evaluation.comparative_analysis import RESULTS_CSV, TEST_QUERIES

    committed = pd.read_csv(RESULTS_CSV)
    assert committed.query_id.nunique() == len(TEST_QUERIES), (
        f"comparative_analysis_results.csv {committed.query_id.nunique()} sorgu tutuyor, "
        f"kod bugun {len(TEST_QUERIES)} sorgu uretiyor -- "
        f"`python -m roamwise.evaluation.comparative_analysis` ile yeniden kostur")
    assert committed.gold_size.min() >= 5


# --- issue #177: are the relations between the named places true? ---

def test_a_travel_duration_is_told_apart_from_a_visit_duration():
    """#173 put three kinds of minutes in the prompt -- the leg, the wait and
    the visit -- and the model repeats all three. Only the first is a claim
    about geography, and an extractor that cannot tell them apart reports the
    day's total walking time as an invented leg.

    The last case is a regression: the disqualifying words used to be searched
    in a fixed 60-character window, which reaches past the end of the sentence
    into the next paragraph's "This route totals ...", and a correct
    four-minute leg claim was dropped because of a "total" belonging to
    another sentence.
    """
    from roamwise.evaluation.geographic_validation import extract_claims

    kinds = lambda t: [(c["kind"], c["value"]) for c in extract_claims(t)]

    assert ("duration", 4.0) in kinds("A 4-minute walk brings you to the gate.")
    assert ("duration", 7.0) in kinds("The bunker is reached in seven minutes.")
    assert ("distance", 1.7) in kinds("A 1.7 km walk brings you there.")
    assert kinds("With a visit duration of 60 minutes, you will see it all.") == []
    assert kinds("The itinerary includes a scheduled 85-minute wait.") == []
    assert kinds("This route totals 7 hours and 55 minutes of walking.") == []

    both = kinds("Only a four-minute walk away, this monument is next.\n\n"
                 "This route totals 7 hours and 55 minutes of walking.")
    assert both == [("duration", 4.0)], both


def test_a_claim_that_cannot_be_placed_on_a_leg_is_not_scored():
    """#175's lesson, one measurement over: a thing that cannot be graded must
    not be given a score. A narrative can assert a hop without saying between
    which two stops, and attaching it to the nearest guess would turn this
    file's uncertainty into the model's error."""
    from roamwise.evaluation.geographic_validation import extract_claims, resolve_leg

    stops = ["Reichstag", "Brandenburg Gate"]
    floating = extract_claims("Everything here is a 10-minute walk from everything else.")[0]
    assert resolve_leg(floating, "Everything here is a 10-minute walk.", stops) == (None, "unresolved")

    text = "**Reichstag** opens the day.\n\n**Brandenburg Gate** is a 4-minute walk away."
    placed = extract_claims(text)[0]
    assert resolve_leg(placed, text, stops) == (1, "heading")

    # A claim inside the first stop's paragraph has no leg before it.
    first = extract_claims("**Reichstag** is a 6-minute walk from the station.")[0]
    assert resolve_leg(first, "**Reichstag** is a 6-minute walk from the station.",
                       stops)[0] is None


def test_the_vague_threshold_is_the_walkable_distance_the_repo_already_committed():
    """"A short stroll" needs a number before it can be graded, and inventing
    one would make the metric arguable. It is taken from
    `travel_modes.HYBRID_WALK_THRESHOLD_KM`, which the repo already documents
    as the distance past which a traveler stops walking by choice -- so if that
    judgement is ever retuned, this metric follows it instead of disagreeing
    with the router about what "walkable" means."""
    from roamwise.evaluation.geographic_validation import grade
    from roamwise.optimization.travel_modes import HYBRID_WALK_THRESHOLD_KM

    vague = {"kind": "vague", "value": None, "at": 0, "phrase": "a short stroll"}
    inside = grade(vague, None, (HYBRID_WALK_THRESHOLD_KM - 0.1, 12.0))
    outside = grade(vague, None, (HYBRID_WALK_THRESHOLD_KM + 0.1, 20.0))

    assert inside["vs_street"] is True and outside["vs_street"] is False
    # Graded on the ground truth only: the prompt's own leg is the estimate
    # under suspicion, so it cannot also be the yardstick.
    assert inside["vs_prompt"] is None


def test_the_geographic_validation_spends_no_quota():
    """#177's own acceptance criterion. The module scores generations that were
    already paid for, so it must reach the cache and nothing else -- a call
    slipped in here would spend a full set of generations every time someone
    re-derived the table, which is exactly the cost #132 built the cache to
    avoid.

    Asserted on the source rather than by running it: the run needs a TOPTW
    solve per query and does not belong in the suite, and the property is a
    static one -- this module names no way of generating."""
    import inspect

    from roamwise.evaluation import geographic_validation

    source = inspect.getsource(geographic_validation)
    for forbidden in ("generate(", "complete(", "complete_verbose(", "require_real_model"):
        assert forbidden not in source, (
            f"{forbidden} in geographic_validation.py -- this measurement must read the "
            f"committed cache, not produce new generations")
    assert "load_cache" in source
