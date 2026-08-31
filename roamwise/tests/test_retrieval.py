"""Retrieval: the Fusion RAG layer, the queries it issues, the category quotas
that keep every wanted category represented, and the catalogue and city guides
underneath it.
"""

import collections

import pandas as pd
import pytest

from roamwise.agents.orchestrator import RETRIEVED_POIS_PER_DAY
from roamwise.agents.router_agent import RouterAgent
from roamwise.knowledge_graph.build_graph import CATEGORY_AFFINITY, GraphIndex
from roamwise.pipeline.common import NOT_A_SIGHT_TYPES
from roamwise.retrieval.fusion import FusionRetriever
from roamwise.retrieval.query import archetype_query
from roamwise.tests.helpers import CITY_CODES, DATA_DIR, MAIN_CITY, city_with_category


@pytest.mark.slow
def test_fusion_retrieval_all_configs_run():
    fr = FusionRetriever()
    for config in ["fusion", "hybrid", "standard"]:
        results = fr.retrieve("museums near a transport hub", config=config,
                              destination_id=MAIN_CITY, top_k=5)
        if config == "standard":
            assert results == []
        else:
            assert len(results) > 0
            assert all(r["destination_id"] == MAIN_CITY for r in results)


def test_a_crowded_category_does_not_starve_a_more_famous_one():
    """Ordering archetype preferences lexicographically on (weight, popularity)
    let one category bury every lower-weighted one outright. Culture Enthusiast
    weights `museum` 1.0 and `landmark` 0.9, and the deepest city holds far
    more museums than retrieval asks for, so every rank up to the museum count
    was a museum and the catalogue's single most popular POI -- a landmark --
    sat behind the city's *last* museum, past the depth anything reads (#63).

    Asserted as a property rather than against a POI name: what must hold is
    that the top-weighted category cannot monopolise the ranking while a
    better-known place in a nearly-as-preferred category waits behind it.
    """
    idx = GraphIndex()
    city = city_with_category("landmark")
    if city is None or not city_with_category("museum", [city]):
        pytest.skip("needs a city holding both museums and landmarks")

    ranked = idx.archetype_preferred_pois("Culture Enthusiast", city, top_k=200)
    assert ranked, "the archetype should prefer something in this city"

    # The most popular POI in any category this archetype prefers at all.
    most_popular = max(ranked, key=lambda p: p["popularity_score"])
    rank = next(i for i, p in enumerate(ranked, 1) if p["poi_id"] == most_popular["poi_id"])

    top_category = max(CATEGORY_AFFINITY["Culture Enthusiast"].items(), key=lambda kv: kv[1])[0]
    n_top_category = sum(1 for p in ranked if p.get("category") == top_category)
    if most_popular.get("category") == top_category:
        pytest.skip("the most popular POI is already in the top-weighted category")

    assert rank < n_top_category, (
        f"{most_popular['name']} ({most_popular['category']}, "
        f"popularity {most_popular['popularity_score']}) ranks {rank}, behind all "
        f"{n_top_category} {top_category} POIs -- the starvation of #63")
    # And it has to survive the depth retrieval actually reads.
    assert rank <= 48


def test_retrieval_query_describes_what_the_traveler_wants_not_the_label():
    """The query was built by interpolating the archetype's name, which made
    BM25 match the literal label word: "culture" surfaced a television channel
    whose description happens to use it (#63). The query has to name
    categories, not the archetype.

    "Strongest category" now means the strongest one the *catalogue holds*
    (#79). Two archetypes lead with `beach`, and these two inland cities hold
    zero beach POIs -- a query term for an empty category can only match the
    wrong thing, so it is dropped and the next real category leads instead."""
    from roamwise.optimization.scoring import DATA_DIR
    from roamwise.retrieval.query import CATEGORY_PHRASE, archetype_query

    held = set(pd.read_csv(DATA_DIR / "poi.csv", usecols=["category"])["category"])
    for archetype, affinities in CATEGORY_AFFINITY.items():
        query = archetype_query(archetype).lower()
        assert archetype.lower() not in query, \
            f"{archetype!r} query still contains the archetype label: {query!r}"
        real = {c: w for c, w in affinities.items() if c in held}
        strongest = max(real.items(), key=lambda kv: kv[1])[0]
        assert CATEGORY_PHRASE[strongest] in query, \
            f"{archetype!r} query omits its strongest category {strongest!r}: {query!r}"
        for empty in set(affinities) - held:
            assert CATEGORY_PHRASE[empty] not in query, \
                f"{archetype!r} query asks for {empty!r}, which the catalogue has none of"


def test_the_preference_matrix_only_spans_categories_the_catalogue_holds():
    """A weight on an empty category is a preference the product cannot act on
    (#79). `beach` was 8.2% of the fitted matrix -- 30% of the `relax` row and
    25% of `nature`'s -- in two cities holding zero beach POIs. Nothing about
    any real POI changed when it went; what changed is that the table stopped
    describing a catalogue we do not have."""
    from roamwise.optimization.scoring import (DATA_DIR, PREFERENCE_CATEGORY_WEIGHTS,
                                               PREFERENCE_DIMS)

    held = set(pd.read_csv(DATA_DIR / "poi.csv", usecols=["category"])["category"])
    for dim in PREFERENCE_DIMS:
        fitted = set(PREFERENCE_CATEGORY_WEIGHTS[dim])
        assert fitted <= held, f"{dim} carries weight on empty categories: {fitted - held}"
    assert held & set(PREFERENCE_CATEGORY_WEIGHTS["culture"]), "the fit lost every category"


def test_the_traveler_is_not_asked_for_a_preference_the_score_cannot_read():
    """Why `views/itinerary.py` has five sliders and not six (#79).

    `adventure` fits to a row of zeros because the catalogue has no
    adventure-ish category, so the score cannot act on it -- and the slider was
    not merely inert: it moved the KMeans archetype, so a traveler asking for
    maximum adventure came back a Budget Backpacker with a different plan for
    an unrelated reason.

    **If this test fails because the row is no longer zero, that is good news**
    -- the catalogue grew a category the slider can act on, and the slider
    should go back into the sidebar."""
    from roamwise.optimization.scoring import PREFERENCE_CATEGORY_WEIGHTS, select_by_score

    row = PREFERENCE_CATEGORY_WEIGHTS["adventure"]
    assert not any(row.values()), \
        f"adventure now reaches the score ({row}) -- restore its slider in the sidebar"

    pois = GraphIndex().city_pois(MAIN_CITY)
    base = {d: 0.5 for d in PREFERENCE_CATEGORY_WEIGHTS}
    picked = [[p["poi_id"] for p in select_by_score(pois, {**base, "adventure": v}, 24)]
              for v in (0.1, 0.9)]
    assert picked[0] == picked[1], "adventure changes the selection after all"


def test_every_category_an_archetype_wants_gets_a_share_of_the_ranking():
    """Issue #113. One ranking cut at `top_k` starves the weakest category
    whenever the stronger ones hold enough POIs to fill the cut: a Culture
    Enthusiast in Paris ranked 241 preferred POIs and the first `religion`
    (weight 0.6) sat at rank 132, against the 72 a three-day trip retrieves.
    The categories are merged proportionally now, so the share each gets
    tracks what the archetype asked for."""
    from roamwise.knowledge_graph.build_graph import CATEGORY_AFFINITY

    idx = GraphIndex()
    wanted = CATEGORY_AFFINITY["Culture Enthusiast"]
    ranked = idx.archetype_preferred_pois("Culture Enthusiast", MAIN_CITY, top_k=72)
    seen = collections.Counter(p.get("category") for p in ranked)

    for category, weight in wanted.items():
        available = [p for p in idx.city_pois(MAIN_CITY) if p.get("category") == category]
        if not available:
            continue     # `beach` holds nothing in these two cities
        assert seen[category], f"{category!r} (weight {weight}) got no slot at all"
    # Proportional, not equal: the strongest category still leads and still
    # takes the most. Both halves matter -- equal shares would be a different
    # bug, and would throw away what the traveler said.
    assert seen["museum"] > seen["religion"], "the strongest category lost its lead"
    assert ranked[0].get("category") == "museum", "the head is no longer what is most wanted"


def test_the_best_known_church_in_the_city_can_actually_be_retrieved():
    """The regression this is really about. Notre-Dame de Paris is the second
    most popular POI in the catalogue and no traveler could be shown it: it
    lost the graph ranking to 131 other POIs and the keyword ranking to a query
    that asked for words the corpus does not use. 83 of 84 `religion` POIs were
    unreachable at a three-day trip length."""
    from roamwise.agents.fusion_rag_agent import FusionRAGAgent
    from roamwise.agents.orchestrator import RETRIEVED_POIS_PER_DAY
    from roamwise.retrieval.query import archetype_query

    idx = GraphIndex()
    churches = [p for p in idx.city_pois(MAIN_CITY) if p.get("category") == "religion"]
    assert churches, "no religion POI in the catalogue to test with"
    best = max(churches, key=lambda p: p["popularity_score"])

    out = FusionRAGAgent().run(archetype_query("Culture Enthusiast"),
                               destination_id=MAIN_CITY, archetype="Culture Enthusiast",
                               config="fusion", top_k=3 * RETRIEVED_POIS_PER_DAY,
                               narrate=False)
    retrieved = {r["poi_id"] for r in out["results"] if r.get("type") == "poi"}
    assert best["poi_id"] in retrieved, (
        f"{best['name']} is the best-known place of worship in {MAIN_CITY} and a "
        "Culture Enthusiast still cannot be shown it")


def test_every_category_query_asks_for_words_the_corpus_actually_uses():
    """`tokenize` does no stemming, so a query matches only the words it
    literally contains. `religion` asked for `worship` (4 of 654 documents) and
    `places` (5) while the documents said `church` (69), and the category was
    invisible to BM25 for a reason that reads as a wording detail (#113).

    Asserted for every category the catalogue holds rather than for `religion`
    alone, because two more were in the same state and nobody had looked
    (#123): `landmarks` matched 2 documents against `landmark`'s 117, and
    `museums` 17 against `museum`'s 127. They stayed hidden because the graph
    carries both at 0.9 and 1.0 affinity, so something always came back --
    `religion` surfaced only because 0.6 could not carry it.

    Tokens the whole corpus contains are excluded before the check. `and`
    appears in 462 of 654 documents and `of` in 546, so counting them would let
    any phrase pass by containing a conjunction -- which is exactly how the
    earlier version of this test passed while `religion`'s phrase was the
    thing under repair. It is also what BM25 itself does: a term in most
    documents has almost no inverse document frequency.
    """
    from roamwise.retrieval.corpus import load_documents, tokenize
    from roamwise.retrieval.query import CATEGORY_PHRASE, _catalogue_categories

    docs = [d for d in load_documents() if d.get("type") == "poi"]
    frequency = collections.Counter()
    for doc in docs:
        frequency.update(set(tokenize(doc["text"])))
    too_common = len(docs) / 2

    for category in sorted(_catalogue_categories()):
        phrase = CATEGORY_PHRASE.get(category)
        if phrase is None:
            continue
        carrying = {t: frequency[t] for t in tokenize(phrase)
                    if frequency[t] < too_common}
        assert carrying and max(carrying.values()) >= 30, (
            f"{category!r} asks for {phrase!r}, which carries no word the corpus "
            f"uses: {sorted(carrying.items(), key=lambda kv: -kv[1])}")


def test_the_category_quota_was_swept_with_the_phrasings_not_after_them():
    """#123's finding, locked in: the retrieval pool is zero-sum at a fixed
    `top_k`, so giving `museum` and `landmark` phrasings that actually match
    pushes `religion` out of the fused top-72 -- from the 30 POIs #113 won back
    down to 23.

    `PREFERENCE_QUOTA_EXPONENT` is the counterweight, and it is below 1.0 for
    that reason and no other. The sweep over both knobs together is
    `evaluation/category_phrase_sweep.py`; of its 100 configurations exactly
    one costs no category anything, and that one is the baseline.
    """
    from roamwise.knowledge_graph.build_graph import PREFERENCE_QUOTA_EXPONENT

    assert 0 < PREFERENCE_QUOTA_EXPONENT < 1.0


def test_the_graph_router_understands_words_travelers_actually_use():
    """It matched only the catalogue's taxonomy words, so "places of worship" --
    the phrasing both the evaluation grid and the orchestrator emit -- routed as
    naming no category at all (#63). Matching is on whole words: "pub" inside
    "public transit" must not make a nightlife query."""
    from roamwise.retrieval.graph_search import categories_in

    assert categories_in("places of worship") == ["religion"]
    assert categories_in("quiet gardens away from the crowds") == ["nature"]
    assert "nightlife" not in categories_in("accessible via late-night public transit")
    assert "nature" not in categories_in("is there parking nearby")
    assert "food" not in categories_in("a great theatre")
    # A profile query names several; a constrained one names exactly one. The
    # retriever routes on that difference.
    assert len(categories_in("museums close to a train station")) == 1
    assert len(categories_in(
        "the best museums, landmarks, history sites, culture venues "
        "and places of worship to visit in this city")) > 1


@pytest.mark.slow
def test_a_trip_gets_every_day_it_asked_for_even_from_a_food_only_pool():
    """A trip used to come back short of the days it was asked for whenever
    candidates ran thin -- day assignment produced one day per POI, so a 5-day
    trip with 3 sightseeing POIs quietly became a 3-day itinerary, and an empty
    sightseeing pool produced no days at all, which `_rebalance_days` crashed
    on with "min() iterable argument is empty" once a query surfaced nothing
    but food (#63).

    The zoner that first carried this guarantee is gone (#72 moved day
    assignment into the TOPTW model, #81 removed the module), so the claim is
    now made where it actually has to hold: end to end, on the pool that
    exposed the crash -- every candidate is food, leaving nothing to sightsee.
    """
    idx = GraphIndex()
    city = city_with_category("food")
    if city is None:
        pytest.skip("no city holds food POIs")
    food = idx.city_pois(city, category="food")[:8]
    itinerary = RouterAgent(idx).run(city, food, n_days=3, narrate=False)["itinerary"]
    assert len(itinerary) == 3


def test_the_catalogue_holds_only_places_a_traveller_can_visit():
    """46 of 700 catalogue rows were entities documented like a landmark that
    nobody can visit -- 25 universities and a hospital filed as `culture`, 15
    metro and mainline stations filed as `landmark`, a television channel, a
    radio station and the 2019 fire at Notre-Dame. They reached itineraries:
    5 of 14 city x archetype plans contained at least one, and a Budget
    Backpacker's Paris culture day opened Sorbonne University -> Sciences Po
    (#65).

    `data/dropped_pois.csv` records each removal with its Wikidata types, so
    the decision stays auditable without a network call.
    """
    catalogue = pd.read_csv(DATA_DIR / "poi.csv")
    dropped = pd.read_csv(DATA_DIR / "dropped_pois.csv")

    assert not set(catalogue.poi_id) & set(dropped.poi_id), \
        "a row recorded as dropped is still in the catalogue"
    assert dropped.drop_reason.notna().all(), "every drop must record its reason"
    assert set(dropped.drop_reason) <= set(NOT_A_SIGHT_TYPES), \
        f"unknown drop reason: {set(dropped.drop_reason) - set(NOT_A_SIGHT_TYPES)}"

    # poi_id is the join key the graph and every retriever use; a rebuild that
    # renumbered the survivors would silently invalidate dropped_pois.csv.
    assert catalogue.poi_id.is_unique


def test_a_disqualifying_type_only_counts_when_it_is_the_whole_story():
    """The rule has to remove an institution without taking a real place that
    merely carries an administrative second type, and REPORT.md section 5's
    argument applies: an entity ending is not its site being gone (#65)."""
    from roamwise.pipeline.common import not_a_sight

    # Removed: the disqualifying type is all there is.
    assert not_a_sight(["comprehensive university", "public research university"])
    assert not_a_sight(["university hospital", "medical school"])
    assert not_a_sight(["multi-level interchange railway station", "central station"])
    assert not_a_sight(["structure fire"])
    # "television station" is not a railway; it must not rescue itself either.
    assert not_a_sight(["television channel", "television station"])
    assert not_a_sight(["broadcaster", "radio station"])

    # Kept: a real place type survives the disqualifying one.
    assert not_a_sight(["art museum", "nonprofit organization"]) is None
    assert not_a_sight(["museum", "event venue"]) is None
    assert not_a_sight(["urban park", "event venue", "public garden"]) is None
    assert not_a_sight(["flea market", "business", "shopping center"]) is None
    assert not_a_sight(["railway station", "palace"]) is None

    # Kept: a heritage listing overrules any disqualifying type at all.
    assert not_a_sight(["cordon", "destroyed building or structure"],
                       has_heritage_listing=True) is None
    assert not_a_sight(["gallows", "destroyed building or structure"]) == "demolished"


def test_the_city_guide_counts_match_the_catalogue_it_describes():
    """The guides are generated from poi.csv and state exact counts, and they
    enter the retrieval corpus as the `guide::<CITY>` document -- so a stale
    guide is a retrievable false statement about the catalogue. Berlin's said
    "300 stops" and anchored the city's central cluster on a university (#65).
    """
    catalogue = pd.read_csv(DATA_DIR / "poi.csv")
    for code in CITY_CODES:
        guide = DATA_DIR / "city_guides" / f"{code}.txt"
        if not guide.exists():
            continue
        text = guide.read_text()
        n = len(catalogue[catalogue.destination_id == code])
        assert f"{n} stops" in text, \
            f"{code} guide does not state the catalogue's real size ({n}); regenerate " \
            f"it with `cd roamwise/pipeline && python city_guide.py --write`"


def test_fusion_beats_hybrid_on_archetype_grounding():
    fr = FusionRetriever()
    city = city_with_category("nightlife") or MAIN_CITY
    fusion_results = fr.retrieve("things to do", config="fusion", destination_id=city,
                                 archetype="Nightlife Seeker", top_k=8)
    fusion_categories = {r.get("text", "") for r in fusion_results}
    # The old assertion also accepted "trastevere", a Rome neighbourhood -- a
    # content literal that cannot survive a change of destination. What this
    # test measures is archetype grounding, and that lives in the first half.
    assert any("nightlife" in t.lower() for t in fusion_categories), \
        f"Nightlife Seeker retrieval for {city} surfaced no nightlife text"


def test_the_preference_matrix_can_be_fitted_on_a_survey_that_is_not_the_shipped_one():
    """The other half of #124's injection point, and its round trip.

    Two things have to hold for `survey_sensitivity.py` to mean anything.
    Passing the shipped survey back in has to reproduce the shipped matrix
    exactly -- otherwise the baseline the whole measurement is differenced
    against is not the matrix the app uses. And passing a survey whose labels
    are shuffled has to move it, because a fit that ignored its argument would
    satisfy the first assertion and report a sensitivity of zero for every
    perturbation.
    """
    import numpy as np
    import pandas as pd

    from roamwise.optimization.scoring import (DATA_DIR, PREFERENCE_CATEGORY_WEIGHTS,
                                               _fit_preference_matrix)

    survey = pd.read_csv(DATA_DIR / "user_survey.csv")
    assert _fit_preference_matrix(survey) == PREFERENCE_CATEGORY_WEIGHTS

    shuffled = survey.copy()
    shuffled["archetype"] = np.random.default_rng(0).permutation(shuffled["archetype"].values)
    refitted = _fit_preference_matrix(shuffled)

    top = lambda m, dim: max(m[dim], key=m[dim].get)
    moved = [d for d in PREFERENCE_CATEGORY_WEIGHTS if top(PREFERENCE_CATEGORY_WEIGHTS, d) != top(refitted, d)]
    assert moved, "shuffling the survey's labels left every category ranking intact"
