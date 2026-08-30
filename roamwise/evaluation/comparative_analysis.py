"""
Comparative analysis across the three retrieval configurations the proposal
asks for: Fusion RAG (Semantic+Graph+Keyword) vs Hybrid RAG (Semantic+Keyword)
vs standard prompting (no retrieval).

Design note on metrics: this project's default LLM layer (agents/llm_client.py)
is a deterministic template, not a live generative model, so it cannot
actually "hallucinate" free text -- there is nothing for it to invent. Rather
than fabricate a fake hallucination number, we measure the thing that
determines hallucination risk at the architecture level: whether the content
handed to the generation step is *grounded* (verifiably present in the
knowledge base) or not, plus two task-accuracy metrics that make the value of
each retrieval component concrete:

  1. Multi-hop query resolution accuracy (Recall@k against a graph-computed
     gold set) -- this is the metric that should most clearly separate Fusion
     RAG from Hybrid RAG, since only Fusion RAG has graph traversal.
  2. Archetype/category relevance precision -- fraction of surfaced POIs that
     actually match the requesting traveler archetype's preferred categories.
  3. Grounded-entity rate -- fraction of surfaced content traceable to a real
     KB node (structural hallucination proxy; 1.0 by construction for any
     retrieval-based config, 0.0 for standard prompting with no fallback).
  4. Itinerary coherence -- average per-day walking distance once the
     RouterAgent builds a route from each config's candidate POIs (lower is
     more geographically coherent).
  5. Retrieval latency -- wall-clock time of the retrieve() call alone. This
     is what the extra accuracy costs: fusion runs three retrievers and an RRF
     merge where standard prompting runs nothing, so the comparison is only
     honest if the price is on the table next to the benefit.

The fifth metric this file used to claim, `grounded_entity_rate`, is renamed
`structural_grounding_rate` here (#132). It asks whether retrieval returned a
row, which is 1.0 by construction for anything that retrieves at all -- a
hallucination *risk* proxy and never a hallucination measurement, however the
old name read in a table. The measurement itself lives in
`evaluation/hallucination.py`: it needs a generative model, and unlike this
file it produces text for the model to be graded on.

`run_llm_hallucination_probe()` here is the zero-context baseline from that
module, kept under its old name because REPORT.md cites it. It is optional and
skipped by default: as a script it needs `--probe`, because rebuilding the
committed CSVs -- the everyday reason to run this file -- costs no generations
at all, and should not start costing them just because a key is exported
(issue #7).
"""
import json
import os
import sys
import datetime
import functools
import time
from pathlib import Path
import warnings
from typing import NamedTuple

# Run as a script (`python evaluation/comparative_analysis.py`), sys.path[0] is
# this file's directory, so the repo root -- where the `roamwise` package lives
# -- never enters the path. Put it there before the roamwise.* imports below.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd
from scipy import stats

from roamwise.agents.llm_client import get_default_llm_client
from roamwise.agents.router_agent import RouterAgent
from roamwise.knowledge_graph.build_graph import CATEGORY_AFFINITY, GraphIndex, haversine_km
from roamwise.retrieval.fusion import FusionRetriever
# The evaluation reaches into the graph retriever's routing keywords on
# purpose: dependence_level() below exists to measure how often the answer key
# and the retriever end up walking the same path.
from roamwise.retrieval import graph_search
from roamwise.retrieval.graph_search import (TRANSPORT_KEYWORDS, GraphSearchIndex,
                                             categories_in)

HERE = Path(__file__).parent
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CONFIGS = ["fusion", "hybrid", "standard"]

# Committed alongside the code so the System logs screen can show the
# comparison without every viewer paying to recompute it.
RESULTS_CSV = HERE / "comparative_analysis_results.csv"
SUMMARY_CSV = HERE / "comparative_analysis_summary.csv"
SIGNIFICANCE_CSV = HERE / "comparative_analysis_significance.csv"

class TestQuery(NamedTuple):
    """One evaluation query and the specification of what counts as a correct answer.

    `categories` is a set, not a single label, because a naturally-phrased
    question rarely maps onto exactly one taxonomy bucket: "historical
    monuments" is answered by a landmark just as well as by a history site.
    Grading those against `category == "history"` alone scored real answers as
    misses and dragged mean recall on naturally-phrased queries down to 0.10
    against 0.45 on keyword-shaped ones -- a measurement artefact that looked
    like a retrieval failure (issue #50).

    `near_transport` is likewise per-query: the gold set only carries the
    "within reach of a hub" constraint when the question actually asks for it.

    `chain_anchor` marks the third tier (#126). A chain query is not answered
    by a category at all: it asks which places can be *sequenced* from where
    the traveler starts the day, and its answer key comes from the graph
    traversal rather than from a category filter. The value is the anchor --
    a city code for the centre, a `transport.csv` id for a gateway.
    """
    destination_id: str
    archetype: str
    categories: tuple[str, ...]
    near_transport: bool
    text: str
    tier: str = "handwritten"
    chain_anchor: str = None


# How close a POI must be to a transport hub to count, when a query asks for it.
# Deliberately looser than the radius the graph retriever traverses with, so a
# correct answer is not defined as "whatever that one traversal returns".
#
# 2.0 km, down from 6.0. That 6.0 was twice the retriever's old 3.0 and
# discriminated on the previous catalogue, where a city carried two hand-picked
# transport rows and one of them (CDG) sat 23 km outside. With real interchanges
# and a denser catalogue it stopped selecting anything at all: every POI in both
# cities lies within 6 km of a hub, so `near_transport` silently became a no-op
# and those queries were graded against the whole category. At 2.0 km the gold
# set is 75% of the unconstrained one -- a real constraint again -- and no query
# is left with an empty answer key. The 2x ratio to the traversal radius is
# preserved.
GOLD_MAX_KM = 2.0

# Written by hand to read like something a traveler would actually type. These
# are the realism tier: they keep the evaluation honest about natural phrasing,
# which the generated grid below cannot test.
HANDWRITTEN_QUERIES = [
    TestQuery("BER", "Culture Enthusiast", ("landmark",), True,
              "landmarks within walking distance of a transport hub"),
    TestQuery("PAR", "Culture Enthusiast", ("museum",), True,
              "museums close to a train station or airport"),
    TestQuery("BER", "Culture Enthusiast", ("museum",), True,
              "museums near a transport hub for history lovers"),
    TestQuery("PAR", "Nightlife Seeker", ("nightlife",), False,
              "nightlife spots this traveler would enjoy"),
    TestQuery("BER", "Nature & Adventure", ("nature",), True,
              "parks and nature spots near transport"),
    TestQuery("PAR", "Budget Backpacker", ("nightlife",), True,
              "cheap nightlife near the train station"),
    TestQuery("BER", "Luxury Traveler", ("shopping",), True,
              "upscale shopping close to transport hubs"),
    # Was a `beach` query. Neither city in the current catalogue has a single
    # beach POI, so the gold set was empty and it scored nothing at all -- the
    # same failure this query was already moved off Lisbon to escape. Kept on
    # Beach & Relax, against the categories the catalogue can actually answer,
    # so the archetype stays represented in the evaluation.
    TestQuery("PAR", "Beach & Relax", ("nature",), True,
              "relaxing green spaces and waterside walks reachable from the airport"),
    TestQuery("BER", "Family Traveler", ("food",), True,
              "we just arrived at the central station with heavy luggage and are starving, "
              "where can we get a quick meal without walking much?"),
    # "suitable places" names no category at all; grading it against museums
    # alone was arbitrary. The archetype is what narrows it.
    TestQuery("PAR", "Culture Enthusiast", ("museum", "landmark", "history", "culture"), True,
              "suitable places for the first day after a morning arrival"),
    TestQuery("BER", "Culture Enthusiast", ("history", "landmark"), True,
              "my elderly parents want to see historical monuments but cannot handle steep "
              "hills or long walks from the subway."),
    TestQuery("PAR", "Nightlife Seeker", ("nightlife",), True,
              "traveling on a tight budget and want to experience local nightlife that is "
              "safely accessible via late-night public transit."),
    TestQuery("BER", "Luxury Traveler", ("shopping", "food"), True,
              "we want to do a full day of luxury shopping and fine dining without needing "
              "to hail a taxi between locations."),
    TestQuery("PAR", "Nature & Adventure", ("nature",), True,
              "are there any quiet natural escapes in the city that are directly connected "
              "to the main train lines?"),
    # Asks where to spend an evening, not where to find culture -- the old
    # `culture` gold graded a nightlife question against museums.
    TestQuery("BER", "Culture Enthusiast", ("nightlife", "food"), False,
              "where should we head for a relaxed evening out right after spending the "
              "entire afternoon at the main art gallery?"),
    TestQuery("PAR", "Budget Backpacker", ("landmark",), True,
              "we have a short layover in the city, what is the most iconic landmark we can "
              "realistically visit and still catch our flight?"),
    TestQuery("BER", "Culture Enthusiast", ("history", "landmark", "food"), False,
              "which neighborhoods offer the best mix of ancient architecture and modern "
              "cafes within a compact walking area?"),
    TestQuery("PAR", "Culture Enthusiast", ("museum", "shopping"), False,
              "if we start our morning at the central square, what is a logical path to hit "
              "a museum and a local market before lunch?"),
    # Also formerly a `beach` query. The text needs no rewriting -- both cities
    # answer it from the river and lake shores already in `nature`.
    TestQuery("BER", "Beach & Relax", ("nature",), False,
              "somewhere calm by the water to spend a slow afternoon"),

    # --- Independently-graded realism queries -------------------------------
    # Every one of these describes an intent without naming its answer key's
    # category or any synonym of it, so `dependence_level` scores them
    # `independent` and the graph retriever cannot reach the key by routing on
    # the query's own words. They were added because teaching that router the
    # synonyms travelers actually use (`places of worship` -> religion, #63)
    # correctly moved four queries the other way, into `subset`, and dropped
    # the independently-graded count to 29 -- below the floor this file's
    # `_GRID_TARGET` comment and `test_query_set_is_powered_and_not_mostly_
    # self_graded` both defend. The grid cannot make up the difference: nine
    # categories x two phrasings x two cities is 36 cells and it already emits
    # all of them, and every grid phrasing names its category by construction.
    # So the shortfall has to be answered where the independent queries live,
    # which is the hand-written tier.
    TestQuery("PAR", "Culture Enthusiast", ("museum",), False,
              "we have one rainy afternoon and want to see the city's most famous "
              "art collection"),
    TestQuery("PAR", "Culture Enthusiast", ("religion",), False,
              "we would like to hear an organ recital beneath a vaulted ceiling"),
    TestQuery("BER", "Nightlife Seeker", ("nightlife",), False,
              "where does this city stay awake until sunrise?"),
    TestQuery("PAR", "Luxury Traveler", ("shopping",), False,
              "my partner would like to bring home something from a famous "
              "Parisian department store"),
    TestQuery("BER", "Family Traveler", ("nature",), False,
              "somewhere the children can run around outdoors for an afternoon"),
    TestQuery("BER", "Nature & Adventure", ("history",), False,
              "we want to understand what daily life was like here before the wall "
              "came down"),
]

# Phrasings for the generated tier. Two families on purpose: one names the
# category next to a transport word, which is the shape the graph retriever's
# router recognises, and one does not. Keeping both means the dependence split
# below stays populated instead of every generated query landing in one bucket.
_GRID_PHRASE = {
    "museum": "museums", "landmark": "landmarks", "history": "history sites",
    "nature": "nature spots", "nightlife": "nightlife venues", "shopping": "shopping",
    "food": "food markets and restaurants", "beach": "beaches",
    "culture": "culture venues", "religion": "places of worship",
}
# `religion` was missing here while _GRID_PHRASE already carried its phrasing.
# Both cities hold plenty of it (50 and 31 POIs), so leaving it out dropped a
# populated category out of the sweep for no stated reason. `beach` stays out:
# it has a phrasing too, but no POI in either city, so every cell would be
# discarded for an empty answer key.
_GRID_CATEGORIES = ["museum", "landmark", "history", "nature", "nightlife",
                    "shopping", "food", "culture", "religion"]


# The day the chain queries are graded on. Fixed, because a chain is a claim
# about opening hours and opening hours are a rule over weekdays (#70) -- an
# answer key that moved with the calendar could not be compared against itself
# a week later. 2026-09-24 is a Thursday, the ordinary case: no Monday
# closures, no weekend hours.
CHAIN_DATE = datetime.date(2026, 9, 24)

# How many gateway-anchored chain queries to add per city, on top of the city
# centre. The centre is the default anchor every plan uses, so it is the one
# that has to be measured; the gateways are there because #126's K3 decision
# rests on the two behaving differently, and a claim like that should be in the
# evaluation rather than only in the issue.
_CHAIN_HUBS_PER_CITY = 2


def build_chain_queries(idx: GraphIndex = None) -> list[TestQuery]:
    """One query per anchor: the city centre, plus its busiest gateways.

    These ask the question the other two tiers cannot. A category query is
    answered by a filter and a naturally-phrased one by a description, but
    neither can express "and then" -- which is the relation #126 built the
    `SERVES`/`REACHABLE` edges for, and the one the proposal argues flat
    document retrieval cannot answer.

    The text still reads like something a traveler would type, because the
    other two configurations have to be able to compete: the chain is not
    keyword-triggered (#126, K3), so `hybrid` and `standard` see the same words
    and answer them however they can. A query whose text was a marker only the
    chain recognised would be a rigged comparison.
    """
    idx = idx or GraphIndex()
    # The traversal reads no archetype -- a chain is the same chain whoever is
    # walking it -- so the archetype here only decides what the *other three*
    # retrievers contribute to the same query. Rotated rather than fixed for
    # exactly that reason: pinning the whole tier to Culture Enthusiast would
    # make the comparison a property of one traveler, and would have pushed
    # that archetype to 37% of the whole query set, near the 40% concentration
    # the set's own balance test forbids.
    archetypes = sorted(CATEGORY_AFFINITY)
    queries = []
    for city in _grid_cities():
        name = idx.g.nodes[city].get("name", city)
        queries.append(TestQuery(
            city, archetypes[len(queries) % len(archetypes)], (), False,
            f"we are staying in the middle of {name} -- what can we see in an afternoon "
            "and still get somewhere else that is open afterwards?",
            "chain", city))
        # Mainline stations, not airports. An airport anchor correctly returns
        # no chain at all -- Charles de Gaulle is 45 minutes out, and #126
        # calls that the right answer -- but a query with an empty answer key
        # is discarded rather than graded, so four of six chain queries would
        # have measured nothing. The airport case is held by a test instead
        # (`test_the_chain_anchor_follows_the_arrival_hub_when_one_is_named`),
        # which is where a "returns nothing, on purpose" claim belongs. The
        # rule is the gateway's own type rather than how many chains it
        # produces: picking hubs by their score would be choosing the
        # evaluation to suit the result.
        stations = [h for h in idx.city_transport(city) if h.get("ttype") == "train_station"]
        for hub in sorted(stations, key=lambda h: h["transport_id"])[:_CHAIN_HUBS_PER_CITY]:
            queries.append(TestQuery(
                city, archetypes[len(queries) % len(archetypes)], (), False,
                f"we arrive at {hub['name']} in the morning -- what can we do first and "
                "what is still open by the time we finish?",
                "chain", hub["transport_id"]))
    return queries


def chain_gold(query: TestQuery) -> tuple[set, float]:
    """Decision K1: the answer key for a chain query, and the key-free metric
    that sits beside it.

    The key is *the POIs in a valid chain from this anchor*, kept to those
    Wikivoyage recommends. Both halves matter. Without the traversal there is
    no way to say what "sequenceable from here" means; without the Wikivoyage
    gate the retriever would be graded against its own output and recall would
    measure that it agrees with itself -- the trap #48 spent an issue climbing
    out of. Gated, a retriever can walk every edge in the graph and still score
    zero for surfacing places nobody recommends.

    The residual circularity is real and is reported rather than argued away:
    these queries are labelled `traversal` in the dependence split, which is
    the coarsest of the four levels on purpose.

    The second value is the **invalid-chain rate**, and it needs no key at all:
    of the pairs the two edges admit from this anchor, what share are not
    actually doable in sequence -- the second place already shut by the time
    you could reach it. It is what the hour constraint is worth, stated in a
    number that no answer key can inflate.
    """
    graph = _chain_index()
    valid = graph.chains(query.destination_id, arrival_hub_id=_hub_or_none(query),
                         start_date=CHAIN_DATE)
    admitted = graph.chains(query.destination_id, arrival_hub_id=_hub_or_none(query),
                            start_date=CHAIN_DATE, enforce_hours=False)
    gold = {poi["poi_id"] for chain in valid for poi in (chain[0], chain[2])}
    invalid_rate = (1 - len(valid) / len(admitted)) if admitted else None
    return gold & RECOMMENDED_POIS, invalid_rate


def _hub_or_none(query: TestQuery) -> str:
    """A chain anchor is either the city itself -- meaning the centre -- or a
    gateway. Retrieval takes the gateway as `arrival_hub_id` and reads the
    centre as the absence of one, so the city code becomes None here."""
    return None if query.chain_anchor == query.destination_id else query.chain_anchor


@functools.lru_cache(maxsize=1)
def _chain_index() -> GraphSearchIndex:
    """One graph-search index for the whole run. Chain gold is computed per
    query and each computation walks thousands of edges; rebuilding the index
    around that would dominate the evaluation."""
    return GraphSearchIndex()


def _grid_cities() -> list[str]:
    """The destinations actually in the catalogue, deepest first.

    This was a literal list of eight city codes, which meant the evaluation
    silently emptied out when the catalogue changed underneath it: on the
    two-city set, seven of nine hand-written cities and twenty-eight of
    thirty-two generated cells graded against an empty answer key.
    """
    pois = pd.read_csv(DATA_DIR / "poi.csv")
    dests = pd.read_csv(DATA_DIR / "destinations.csv")
    counts = pois.destination_id.value_counts()
    return sorted((c for c in dests.destination_id if counts.get(c, 0)),
                  key=lambda c: -counts[c])


# How many generated queries to aim for. The grid exists to give the pairwise
# tests enough paired observations: at 11 gradeable queries a real effect is
# detected about a third of the time, which is what issue #50 set out to fix.
# Holding the *target* fixed rather than the categories-per-city keeps that
# power when the destination list shrinks -- two cities simply sweep more
# categories each instead of contributing four cells apiece.
#
# 36 rather than 32: sweeping each (city, category) cell in both phrasings puts
# half the generated tier in the "subset" bucket, where grading leans on the
# retriever's own traversal (issue #48). The independently-graded count is what
# carries the pairwise tests, so the target is set by that floor, not by the
# total.
_GRID_TARGET = 36


def _eligible_archetypes(category: str) -> list[str]:
    """Every archetype that has any appetite for this category.

    Assigning each category to its single strongest archetype looks tidy and
    ruins the balance: Culture Enthusiast tops museum, landmark, history,
    culture and religion, so it would claim five of the eight categories in
    every city and end up owning half the set -- the exact skew this grid
    exists to correct.
    """
    return sorted(a for a, affinity in CATEGORY_AFFINITY.items() if affinity.get(category, 0.0) > 0)


def build_grid_queries(idx: GraphIndex = None) -> list[TestQuery]:
    """A balanced sweep of (city, category) cells with non-empty gold sets.

    The hand-written set covers a minority of the city x category cells and leans
    hard on one archetype, which left the comparison underpowered: the queries
    whose grading does not lean on the retriever's own traversal numbered 11,
    and at that size a real effect is only detected about a third of the time.
    """
    idx = idx or GraphIndex()
    cities = _grid_cities()
    if not cities:
        return []

    # Spread the target over the cities, then over categories within each. With
    # eight cities this is four categories each, exactly as before; with two it
    # is all eight categories in both phrasings.
    per_city = max(1, round(_GRID_TARGET / len(cities)))
    queries, used = [], set()
    for position, city in enumerate(cities):
        for slot in range(per_city):
            # Walking the category list with a per-city offset gives every
            # category the same number of cells instead of front-loading the
            # first four.
            category = _GRID_CATEGORIES[(position * per_city + slot) % len(_GRID_CATEGORIES)]
            eligible = _eligible_archetypes(category)
            archetype = eligible[(position + slot) % len(eligible)]

            # Alternate the two phrasings across the grid rather than splitting
            # by city or category, so neither family clusters in one city.
            near_transport = (position + slot) % 5 < 2
            # Fewer cities than categories means a city sweeps the category
            # list more than once, and the pattern above would then re-emit a
            # cell it had already produced -- six queries came back byte-identical
            # on the two-city catalogue. Asking the repeat the other way keeps
            # the observation distinct instead of double-counting one.
            if (city, category, near_transport) in used:
                near_transport = not near_transport
            if (city, category, near_transport) in used:
                continue          # both phrasings already asked for this cell
            used.add((city, category, near_transport))

            phrase = _GRID_PHRASE[category]
            text = (f"{phrase} within easy reach of a transport hub" if near_transport
                    else f"the best {phrase} to visit in this city")
            query = TestQuery(city, archetype, (category,), near_transport, text, tier="grid")
            if gold_for(idx, query):  # a cell with no answer cannot be scored
                queries.append(query)
    return queries


# The answer key's membership test, built by `pipeline/retrieval_gold.py` from
# Wikivoyage listings and committed so the evaluation runs offline.
#
# This is what stops the key being circular. It used to be "every catalogue POI
# of the queried category", which is `GraphIndex.city_pois` -- the same
# traversal the graph retriever dispatches to -- so on a query naming its
# category the retriever's results were a subset of the key by construction,
# and its recall measured agreement with itself (issue #48). Wikivoyage is
# written by travellers listing what is worth going to and owes nothing to
# Wikidata sitelinks, OSM tagging or this project's graph, so being on it is an
# independent statement that a place is worth recommending. Retrieval can now
# return plenty of category matches and still score zero, which is the point.
_GOLD_CSV = HERE / "retrieval_gold.csv"


def _recommended_pois() -> set:
    if not _GOLD_CSV.exists():
        raise FileNotFoundError(
            f"{_GOLD_CSV.name} is missing -- rebuild it with "
            "`cd roamwise/pipeline && python retrieval_gold.py --write`")
    return set(pd.read_csv(_GOLD_CSV).poi_id)


RECOMMENDED_POIS = _recommended_pois()


def gold_for(idx: GraphIndex, query: TestQuery) -> set:
    """POIs that would answer `query`.

    The union of its accepted categories, kept to those Wikivoyage recommends,
    and narrowed to those near a hub only when the query asked for that. A
    chain query is answered by a traversal instead -- see `chain_gold`.
    """
    if query.chain_anchor:
        return chain_gold(query)[0]
    pois = [poi for category in query.categories
            for poi in idx.city_pois(query.destination_id, category=category)
            if poi["poi_id"] in RECOMMENDED_POIS]
    if not query.near_transport:
        return {poi["poi_id"] for poi in pois}

    hubs = idx.city_transport(query.destination_id)
    return {
        poi["poi_id"] for poi in pois
        if min((haversine_km(poi["lat"], poi["lon"], hub["lat"], hub["lon"])
                for hub in hubs), default=999) <= GOLD_MAX_KM
    }


def dependence_level(query: TestQuery) -> str:
    """Whether this query hands the graph retriever its answer key's category.

    This used to measure circularity: the key was every catalogue POI of the
    queried category, which is the traversal the retriever dispatches to, so a
    query naming its category made retrieval a subset of the key by
    construction. That is fixed at the source -- the key is now gated on
    Wikivoyage (see RECOMMENDED_POIS), which no traversal here can reach, so a
    retriever cannot satisfy it by agreeing with itself.

    What the split still measures is *difficulty*. A query that names its
    category tells the router where to look; one that describes an intent
    ("somewhere calm by the water") does not, and answering it needs the
    semantic layer to carry its weight. Reported per query so a headline number
    cannot be carried entirely by the easy half (issue #48).
    """
    if query.chain_anchor:
        # The key is the traversal's own output, gated on Wikivoyage (K1). The
        # gate is what stops recall measuring self-agreement, but it does not
        # make the key independent of the retriever the way the other three
        # levels are, and calling it `superset` would have understated that.
        return "traversal"
    text = query.text.lower()
    # The same router the retriever runs, synonyms included -- classifying with
    # a stricter rule than the code under test reported queries as
    # independently graded when the retriever did reach for the key's category
    # (#63).
    matched = next(iter(categories_in(text)), None)
    names_transport = any(keyword in text for keyword in TRANSPORT_KEYWORDS)

    if matched not in query.categories:
        return "independent"          # the router never reaches for the gold's category
    if names_transport and query.near_transport and len(query.categories) == 1:
        return "subset"               # retrieval is a strict subset of the gold set
    return "superset"                 # same category filter, wider than the gold


def _live_handwritten() -> list[TestQuery]:
    """Hand-written queries whose destination is still in the catalogue.

    The hand-written tier is maintained by hand against a particular set of
    cities -- that is what makes it the realism tier. When the catalogue drops
    a city, its queries would otherwise stay in the set and grade against an
    empty answer key, quietly shrinking the sample instead of failing.
    """
    live = set(_grid_cities())
    kept = [q for q in HANDWRITTEN_QUERIES if q.destination_id in live]
    dropped = {q.destination_id for q in HANDWRITTEN_QUERIES} - live
    if dropped:
        warnings.warn(
            f"hand-written queries dropped for cities not in the catalogue: "
            f"{sorted(dropped)}. Re-target them in HANDWRITTEN_QUERIES.",
            stacklevel=2)
    return kept


TEST_QUERIES = _live_handwritten() + build_grid_queries() + build_chain_queries()


def recall_at_k(retrieved_ids: list, gold: set) -> float:
    if not gold:
        return None
    hit = len(set(retrieved_ids) & gold)
    return hit / len(gold)


def run_comparative_analysis(top_k: int = 8, use_real_routing: bool = False) -> pd.DataFrame:
    """`use_real_routing` prices day 1 on the committed street network instead
    of the straight line, which moves `n_stops_day1` and `km_per_stop_day1`
    and nothing else -- retrieval does not know the flag exists. It is a
    parameter rather than a constant because #93/#94 asked what the table
    looks like with it on, and the answer has to be reproducible."""
    idx = GraphIndex()
    retriever = FusionRetriever()
    router = RouterAgent(idx)
    rows = []

    for query_id, test_query in enumerate(TEST_QUERIES):
        dest_id, archetype, query = test_query.destination_id, test_query.archetype, test_query.text
        if test_query.chain_anchor:
            gold, invalid_chain_rate = chain_gold(test_query)
        else:
            gold, invalid_chain_rate = gold_for(idx, test_query), None
        level = dependence_level(test_query)
        preferred_categories = set(CATEGORY_AFFINITY[archetype])
        # A chain query names where the day starts and which day it is; the
        # other two tiers name neither, and passing them anyway would let the
        # chain fire on queries that never asked to be sequenced.
        anchor_kwargs = ({"arrival_hub_id": _hub_or_none(test_query),
                          "start_date": CHAIN_DATE}
                         if test_query.chain_anchor else {})

        for config in CONFIGS:
            # Timed around retrieve() only: routing runs on every config alike
            # and would bury the difference this table exists to show.
            started = time.perf_counter()
            results = retriever.retrieve(query, config=config, destination_id=dest_id,
                                         archetype=archetype, top_k=top_k, **anchor_kwargs)
            retrieval_ms = (time.perf_counter() - started) * 1000
            poi_results = [r for r in results if r.get("type") == "poi"]
            retrieved_ids = [r["poi_id"] for r in poi_results]

            recall = recall_at_k(retrieved_ids, gold)

            candidate_pois = [idx.g.nodes[pid] | {"poi_id": pid} for pid in retrieved_ids]
            if not candidate_pois:  # standard prompting has no retrieval -> unfiltered fallback
                candidate_pois = idx.city_pois(dest_id)[:top_k]

            relevant = sum(1 for p in candidate_pois if p.get("category") in preferred_categories)
            precision = relevant / len(candidate_pois) if candidate_pois else 0.0

            # Renamed from `grounded_entity_rate` in #132. The old name read
            # as a hallucination measurement in every table it appeared in,
            # and it is not one: it asks whether retrieval returned a row, so
            # it is 1.0 for any retrieval-based config by construction and the
            # Wilcoxon test on it can only ever answer "identical". The real
            # measurement is evaluation/hallucination.py, which needs a
            # generative model and generates text for it.
            structural_grounding = 1.0 if poi_results else 0.0

            # narrate=False: only the day's geometry is scored below, never
            # the prose. Left on, this fires one generation per query per
            # config -- free under TemplateLLMClient, but hours of model time
            # once a real LLM is configured (same waste as issue #57).
            routing = router.run(dest_id, candidate_pois, n_days=1, narrate=False,
                                use_real_routing=use_real_routing)
            day = routing["itinerary"][0]
            n_stops = len(day["route"])
            km_per_stop = day["distance_km"] / n_stops if n_stops else None

            rows.append({
                # Explicit key rather than row position: the significance test
                # pairs configs per query, and (destination, archetype,
                # category) is not unique -- ROM/Culture Enthusiast/museum
                # appears three times in TEST_QUERIES.
                "query_id": query_id,
                "tier": test_query.tier, "dependence": level,
                "destination_id": dest_id, "archetype": archetype,
                # The question itself, not just its metadata. Without it the
                # results identify a query only by id, and a reader cannot
                # judge whether it was fair or whether its answer key matches
                # what it asked -- which is how the mislabelled keys in #50
                # went unnoticed (issue #85).
                "query": test_query.text,
                "category": "+".join(test_query.categories),
                "near_transport": test_query.near_transport, "gold_size": len(gold),
                # `top_k` is written down because recall's own ceiling is
                # `min(top_k, gold_size) / gold_size` and the System logs
                # screen reports recall against it. Reading that ceiling off a
                # constant somewhere else is how it went stale and got
                # hardcoded at 49% in the first place (#126).
                "top_k": top_k,
                # Which anchor, and whether the traversal that answers it was
                # switched on when the row was measured. A chain row taken with
                # the flag off is a real measurement -- it is the "before" the
                # flag decision is read against -- but it is not the same
                # measurement as one taken with it on, and a table that did not
                # say which would be unreadable.
                "chain_anchor": test_query.chain_anchor,
                "chain_enabled": graph_search.CHAIN_ENABLED if test_query.chain_anchor else None,
                # Key-free, so no answer key can inflate it: of the pairs the
                # two edges admit from this anchor, the share that are not
                # actually doable in sequence (K1).
                "invalid_chain_rate": (round(invalid_chain_rate, 3)
                                       if invalid_chain_rate is not None else None),
                "config": config, "recall_at_k": recall,
                # Recall against what this query could actually have scored,
                # not against 1.0. Only `top_k` places come back and most keys
                # are larger, so raw recall reads as a failure that is mostly
                # the key's size (#49). Stored per row rather than derived in
                # the UI so the significance test can pair on it like any other
                # metric.
                "normalized_recall": (round(recall / (min(top_k, len(gold)) / len(gold)), 4)
                                      if gold and recall is not None else None),
                "archetype_precision": round(precision, 3),
                "structural_grounding_rate": structural_grounding, "n_candidate_pois": len(candidate_pois),
                "n_stops_day1": n_stops, "km_per_stop_day1": round(km_per_stop, 2) if km_per_stop else None,
                "retrieval_ms": round(retrieval_ms, 2),
            })

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("config")
        .agg(
            mean_recall_at_k=("recall_at_k", "mean"),
            mean_normalized_recall=("normalized_recall", "mean"),
            mean_archetype_precision=("archetype_precision", "mean"),
            mean_structural_grounding_rate=("structural_grounding_rate", "mean"),
            mean_km_per_stop=("km_per_stop_day1", "mean"),
            mean_retrieval_ms=("retrieval_ms", "mean"),
        )
        .reindex(CONFIGS)
        .round(3)
    )


ALPHA = 0.05

# (results column, higher_is_better) -- the direction decides which sign of the
# per-query difference counts as a win.
METRICS_UNDER_TEST = [
    ("recall_at_k", True),
    # Tested alongside raw recall rather than instead of it. The two can
    # disagree: a config that wins on queries with small answer keys and loses
    # on large ones moves them in opposite directions, and which of those is
    # the better retriever is exactly the question the split is here to expose
    # (#49, #126).
    ("normalized_recall", True),
    ("archetype_precision", True),
    ("structural_grounding_rate", True),
    ("km_per_stop_day1", False),
    ("retrieval_ms", False),
]


def paired_significance(df: pd.DataFrame, champion: str = "fusion", alpha: float = ALPHA) -> pd.DataFrame:
    """Is each config's lead real, or is it noise?

    Every config answers the same TEST_QUERIES, so the observations are paired
    and a Wilcoxon signed-rank test applies -- signed-rank rather than a t-test
    because these are bounded rates and per-day distances over 18 queries, with
    no reason to assume normal differences.

    Reporting means alone was actively misleading here: Fusion's 1.14 km/stop
    against Hybrid's 1.23 reads as a win, but Fusion is actually *behind* on 9
    of 18 queries (p=0.46). Without this table the summary invites a claim the
    data does not support (issue #46).
    """
    if "query_id" not in df.columns:
        # Results written before query_id existed. Rows are emitted in
        # TEST_QUERIES order within each config and a CSV round-trip preserves
        # that order, so position within a config recovers the same pairing --
        # which is why the committed results do not need regenerating for this.
        df = df.assign(query_id=df.groupby("config").cumcount())

    rows = []
    present = set(df["config"].unique())
    for opponent in [c for c in CONFIGS if c != champion and c in present]:
        for column, higher_is_better in METRICS_UNDER_TEST:
            if column not in df.columns:
                continue  # a results set predating this metric
            paired = (df.pivot(index="query_id", columns="config", values=column)
                        [[champion, opponent]].dropna())
            # A query whose gold set is empty scores None for recall and drops
            # out of that metric's pairing, not out of the whole comparison.
            advantage = paired[champion] - paired[opponent]
            if not higher_is_better:
                advantage = -advantage

            wins = int((advantage > 0).sum())
            losses = int((advantage < 0).sum())
            ties = int((advantage == 0).sum())

            if ties == len(advantage):
                # Wilcoxon cannot rank an all-zero difference vector, and there
                # is nothing to test: structural_grounding_rate is 1.0 by
                # construction for every retrieval-based config, so fusion and
                # hybrid are identical on it and it cannot separate them.
                p_value, verdict = None, "identical"
            else:
                p_value = float(stats.wilcoxon(paired[champion], paired[opponent]).pvalue)
                if p_value >= alpha:
                    verdict = "no difference"
                else:
                    verdict = "better" if wins > losses else "worse"

            rows.append({
                "metric": column, "champion": champion, "opponent": opponent,
                "n": len(advantage), "mean_advantage": round(float(advantage.mean()), 4),
                "wins": wins, "losses": losses, "ties": ties,
                "p_value": None if p_value is None else round(p_value, 5),
                "verdict": verdict,
            })
    return pd.DataFrame(rows)


def run_llm_hallucination_probe():
    """The zero-context generative baseline, delegated to evaluation/hallucination.py.

    Kept under this name because REPORT.md §3.5 and §7 cite it. Everything
    that made it unrunnable is gone: it was gated on `ANTHROPIC_API_KEY` and
    constructed `AnthropicLLMClient` directly, so choosing NVIDIA (#7) left it
    permanently skipped and `llm_hallucination_probe.csv` was never written
    once. It goes through `get_default_llm_client()` now, and it refuses under
    the offline template rather than reporting a score no model produced.
    """
    from roamwise.evaluation.hallucination import run_zero_context_probe
    return run_zero_context_probe()


if __name__ == "__main__":
    df = run_comparative_analysis()
    df.to_csv(RESULTS_CSV, index=False)
    summary = summarize(df)
    summary.to_csv(SUMMARY_CSV)
    print(summary.to_string())

    significance = paired_significance(df)
    significance.to_csv(SIGNIFICANCE_CSV, index=False)
    print("\nIs the lead real? (Wilcoxon signed-rank, paired by query)\n",
          significance.to_string(index=False))

    # Behind a flag as well as behind the key (issue #7). Rebuilding the
    # committed CSVs is the everyday reason to run this file, and it needs no
    # model at all -- run_comparative_analysis passes narrate=False throughout.
    # Spending live API calls on it because a key happens to be exported is a
    # cost nobody asked for, and the key alone cannot express the difference.
    if "--probe" in sys.argv:
        from roamwise.evaluation.hallucination import PROBE_CSV, TemplateClientRefused
        try:
            probe = run_llm_hallucination_probe()
        except TemplateClientRefused as exc:
            print(f"\n(--probe given but {exc})")
        else:
            probe.to_csv(PROBE_CSV, index=False)
            print("\nLive LLM hallucination probe:\n", probe.to_string())
    else:
        print("\n(Skipped the live-LLM hallucination probe; pass --probe to run it. The full "
              "per-config measurement is `python -m roamwise.evaluation.hallucination`.)")
