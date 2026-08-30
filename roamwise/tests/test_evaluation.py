"""The comparative analysis behind the Results tab: the query set it runs, the
answer key it is graded against, and the paired significance test that decides
whether a lead is real.

Split out of `test_orchestration.py` in #164. These test
`evaluation/comparative_analysis.py`, not the orchestrators -- and leaving them
there put the measurement work and the orchestrator work in one file.
"""

import collections
from pathlib import Path

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

    assert graded["hallucinated"] == 0
    assert graded["hallucination_rate"] == 0.0


def test_a_place_from_the_other_city_is_geographical_hallucination(gaz):
    """The proposal's own term: a real place, named for the wrong city."""
    places, pattern = gaz
    prompt = "Day 1: Humboldt Forum."
    graded = classify("Visit the Humboldt Forum, then the Eiffel Tower.",
                      prompt, "BER", places, pattern)

    assert graded["wrong_city"] == 1
    assert graded["wrong_city_names"] == "Eiffel Tower"
    assert graded["hallucination_rate"] == 0.5


def test_a_narrative_that_names_no_place_is_not_a_zero_percent_score(gaz):
    """An empty denominator is no measurement, not a perfect one."""
    places, pattern = gaz
    graded = classify("A pleasant day out.", "Day 1: Humboldt Forum.", "BER", places, pattern)

    assert graded["places_named"] == 0
    assert graded["hallucination_rate"] is None


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
