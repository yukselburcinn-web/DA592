"""The comparative analysis behind the Results tab: the query set it runs, the
answer key it is graded against, and the paired significance test that decides
whether a lead is real.

Split out of `test_orchestration.py` in #164. These test
`evaluation/comparative_analysis.py`, not the orchestrators -- and leaving them
there put the measurement work and the orchestrator work in one file.
"""

import collections

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
    assert summary.loc["fusion", "mean_grounded_entity_rate"] == 1.0
    assert summary.loc["standard", "mean_grounded_entity_rate"] == 0.0

    # The headline on the Results tab is generated from these verdicts, so a
    # metric silently changing category would rewrite a user-facing claim.
    verdicts = paired_significance(df).set_index(["metric", "opponent"]).verdict
    assert verdicts[("archetype_precision", "hybrid")] == "better"
    # Grounded-entity rate is 1.0 by construction for anything that retrieves,
    # so it can separate Fusion from standard prompting but never from Hybrid.
    assert verdicts[("grounded_entity_rate", "hybrid")] == "identical"
    assert verdicts[("grounded_entity_rate", "standard")] == "better"


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
                "grounded_entity_rate": [1.0] * 10, "km_per_stop_day1": [1.0] * 10,
                "recall_at_k": [0.5] * 10},
        hybrid={"archetype_precision": [0.5] * 10, "retrieval_ms": [5.0] * 10,
                "grounded_entity_rate": [1.0] * 10, "km_per_stop_day1": [1.0] * 10,
                "recall_at_k": [0.5] * 10},
    )
    verdicts = paired_significance(df, champion="fusion").set_index("metric")

    assert verdicts.loc["archetype_precision", "verdict"] == "better"
    # Higher latency is worse even though the number went up -- the sign has to
    # be flipped for lower-is-better metrics or this reads as a win.
    assert verdicts.loc["retrieval_ms", "verdict"] == "worse"
    assert verdicts.loc["retrieval_ms", "mean_advantage"] < 0
    # Identical columns: no test is possible, and it must not raise.
    assert verdicts.loc["grounded_entity_rate", "verdict"] == "identical"
    assert pd.isna(verdicts.loc["grounded_entity_rate", "p_value"])


def test_paired_significance_calls_a_coin_flip_no_difference():
    """A metric that trades wins back and forth must not be reported as a lead
    just because its mean happens to land higher."""
    df = _synthetic_pairs(
        fusion={"km_per_stop_day1": [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]},
        hybrid={"km_per_stop_day1": [2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.1]},
    )
    verdict = paired_significance(df, champion="fusion").set_index("metric")
    assert verdict.loc["km_per_stop_day1", "verdict"] == "no difference"
