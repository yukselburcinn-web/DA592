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

If ANTHROPIC_API_KEY is set, `run_llm_hallucination_probe()` additionally
runs a real generative hallucination check (see its docstring) -- this part
is optional and skipped by default.
"""
import json
import os
import sys
import time
from pathlib import Path
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
from roamwise.retrieval.graph_search import CATEGORY_KEYWORDS, TRANSPORT_KEYWORDS

HERE = Path(__file__).parent
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
    """
    destination_id: str
    archetype: str
    categories: tuple[str, ...]
    near_transport: bool
    text: str
    tier: str = "handwritten"


# How close a POI must be to a transport hub to count, when a query asks for it.
# Deliberately looser than the 3.0 km the graph retriever traverses with, so a
# correct answer is not defined as "whatever that one traversal returns".
GOLD_MAX_KM = 6.0

# Written by hand to read like something a traveler would actually type. These
# are the realism tier: they keep the evaluation honest about natural phrasing,
# which the generated grid below cannot test.
HANDWRITTEN_QUERIES = [
    TestQuery("IST", "Culture Enthusiast", ("landmark",), True,
              "landmarks within walking distance of a transport hub"),
    TestQuery("PAR", "Culture Enthusiast", ("museum",), True,
              "museums close to a train station or airport"),
    TestQuery("ROM", "Culture Enthusiast", ("museum",), True,
              "museums near a transport hub for history lovers"),
    TestQuery("BCN", "Nightlife Seeker", ("nightlife",), False,
              "nightlife spots this traveler would enjoy"),
    TestQuery("AMS", "Nature & Adventure", ("nature",), True,
              "parks and nature spots near transport"),
    TestQuery("PRG", "Budget Backpacker", ("nightlife",), True,
              "cheap nightlife near the train station"),
    TestQuery("VIE", "Luxury Traveler", ("shopping",), True,
              "upscale shopping close to transport hubs"),
    # Moved off Lisbon: it has four beach POIs and none within GOLD_MAX_KM of a
    # hub, so this query's gold set was empty and it scored nothing at all.
    TestQuery("BCN", "Beach & Relax", ("beach",), True,
              "relaxing beach spots reachable from the airport"),
    TestQuery("IST", "Family Traveler", ("food",), True,
              "we just arrived at the central station with heavy luggage and are starving, "
              "where can we get a quick meal without walking much?"),
    # "suitable places" names no category at all; grading it against museums
    # alone was arbitrary. The archetype is what narrows it.
    TestQuery("PAR", "Culture Enthusiast", ("museum", "landmark", "history", "culture"), True,
              "suitable places for the first day after a morning arrival"),
    TestQuery("ROM", "Culture Enthusiast", ("history", "landmark"), True,
              "my elderly parents want to see historical monuments but cannot handle steep "
              "hills or long walks from the subway."),
    TestQuery("AMS", "Nightlife Seeker", ("nightlife",), True,
              "traveling on a tight budget and want to experience local nightlife that is "
              "safely accessible via late-night public transit."),
    TestQuery("BCN", "Luxury Traveler", ("shopping", "food"), True,
              "we want to do a full day of luxury shopping and fine dining without needing "
              "to hail a taxi between locations."),
    TestQuery("LIS", "Nature & Adventure", ("nature",), True,
              "are there any quiet natural escapes in the city that are directly connected "
              "to the main train lines?"),
    # Asks where to spend an evening, not where to find culture -- the old
    # `culture` gold graded a nightlife question against museums.
    TestQuery("VIE", "Culture Enthusiast", ("nightlife", "food"), False,
              "where should we head for a relaxed evening out right after spending the "
              "entire afternoon at the main art gallery?"),
    TestQuery("PRG", "Budget Backpacker", ("landmark",), True,
              "we have a short layover in the city, what is the most iconic landmark we can "
              "realistically visit and still catch our flight?"),
    TestQuery("IST", "Culture Enthusiast", ("history", "landmark", "food"), False,
              "which neighborhoods offer the best mix of ancient architecture and modern "
              "cafes within a compact walking area?"),
    TestQuery("ROM", "Culture Enthusiast", ("museum", "shopping"), False,
              "if we start our morning at the central square, what is a logical path to hit "
              "a museum and a local market before lunch?"),
    # Lisbon's catalogue is small enough that these gold sets fit inside top_k,
    # so at least a few queries can actually reach recall 1.0 (issue #49).
    TestQuery("LIS", "Beach & Relax", ("beach", "nature"), False,
              "somewhere calm by the water to spend a slow afternoon"),
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
_GRID_CITIES = ["IST", "PAR", "ROM", "BCN", "AMS", "PRG", "VIE", "LIS"]
_GRID_CATEGORIES = ["museum", "landmark", "history", "nature", "nightlife",
                    "shopping", "food", "culture"]


# Four categories per city rather than all eight: that lands the combined set
# near 50 queries, which is where the power curve flattens, without doubling
# how long a re-run takes for evidence nobody needs.
_GRID_CATEGORIES_PER_CITY = 4


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

    The hand-written set covers 16 of the 80 city x category cells and leans
    hard on one archetype, which left the comparison underpowered: the queries
    whose grading does not lean on the retriever's own traversal numbered 11,
    and at that size a real effect is only detected about a third of the time.
    """
    idx = idx or GraphIndex()
    queries = []
    for position, city in enumerate(_GRID_CITIES):
        for slot in range(_GRID_CATEGORIES_PER_CITY):
            # Walking the category list with a per-city offset gives every
            # category the same number of cells instead of front-loading the
            # first four.
            category = _GRID_CATEGORIES[(position * _GRID_CATEGORIES_PER_CITY + slot) % len(_GRID_CATEGORIES)]
            eligible = _eligible_archetypes(category)
            archetype = eligible[(position + slot) % len(eligible)]

            # Alternate the two phrasings across the grid rather than splitting
            # by city or category, so neither family clusters in one city.
            near_transport = (position + slot) % 5 < 2
            phrase = _GRID_PHRASE[category]
            text = (f"{phrase} within easy reach of a transport hub" if near_transport
                    else f"the best {phrase} to visit in this city")
            query = TestQuery(city, archetype, (category,), near_transport, text, tier="grid")
            if gold_for(idx, query):  # a cell with no answer cannot be scored
                queries.append(query)
    return queries


def gold_for(idx: GraphIndex, query: TestQuery) -> set:
    """POIs that would answer `query`: the union of its accepted categories,
    narrowed to those near a hub only when the query asked for that."""
    pois = [poi for category in query.categories
            for poi in idx.city_pois(query.destination_id, category=category)]
    if not query.near_transport:
        return {poi["poi_id"] for poi in pois}

    hubs = idx.city_transport(query.destination_id)
    return {
        poi["poi_id"] for poi in pois
        if min((haversine_km(poi["lat"], poi["lon"], hub["lat"], hub["lon"])
                for hub in hubs), default=999) <= GOLD_MAX_KM
    }


def dependence_level(query: TestQuery) -> str:
    """How much of this query's grading leans on the retriever's own traversal.

    The graph retriever routes on literal keywords, so when a query names the
    gold's category next to a transport word it dispatches to the very
    traversal the gold set is built from -- at a tighter radius, which makes
    its results a guaranteed subset of the answer key. That is not evidence
    that graph traversal finds better answers, only that it agrees with
    itself. Reported per query so the split is visible rather than assumed
    (issue #48).
    """
    text = query.text.lower()
    matched = next((c for c in CATEGORY_KEYWORDS if c in text), None)
    names_transport = any(keyword in text for keyword in TRANSPORT_KEYWORDS)

    if matched not in query.categories:
        return "independent"          # the router never reaches for the gold's category
    if names_transport and query.near_transport and len(query.categories) == 1:
        return "subset"               # retrieval is a strict subset of the gold set
    return "superset"                 # same category filter, wider than the gold


TEST_QUERIES = HANDWRITTEN_QUERIES + build_grid_queries()


def recall_at_k(retrieved_ids: list, gold: set) -> float:
    if not gold:
        return None
    hit = len(set(retrieved_ids) & gold)
    return hit / len(gold)


def run_comparative_analysis(top_k: int = 8) -> pd.DataFrame:
    idx = GraphIndex()
    retriever = FusionRetriever()
    router = RouterAgent(idx)
    rows = []

    for query_id, test_query in enumerate(TEST_QUERIES):
        dest_id, archetype, query = test_query.destination_id, test_query.archetype, test_query.text
        gold = gold_for(idx, test_query)
        level = dependence_level(test_query)
        preferred_categories = set(CATEGORY_AFFINITY[archetype])

        for config in CONFIGS:
            # Timed around retrieve() only: routing runs on every config alike
            # and would bury the difference this table exists to show.
            started = time.perf_counter()
            results = retriever.retrieve(query, config=config, destination_id=dest_id, archetype=archetype, top_k=top_k)
            retrieval_ms = (time.perf_counter() - started) * 1000
            poi_results = [r for r in results if r.get("type") == "poi"]
            retrieved_ids = [r["poi_id"] for r in poi_results]

            recall = recall_at_k(retrieved_ids, gold)

            candidate_pois = [idx.g.nodes[pid] | {"poi_id": pid} for pid in retrieved_ids]
            if not candidate_pois:  # standard prompting has no retrieval -> unfiltered fallback
                candidate_pois = idx.city_pois(dest_id)[:top_k]

            relevant = sum(1 for p in candidate_pois if p.get("category") in preferred_categories)
            precision = relevant / len(candidate_pois) if candidate_pois else 0.0

            grounded_rate = 1.0 if poi_results else 0.0

            routing = router.run(dest_id, candidate_pois, n_days=1)
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
                "category": "+".join(test_query.categories),
                "near_transport": test_query.near_transport, "gold_size": len(gold),
                "config": config, "recall_at_k": recall, "archetype_precision": round(precision, 3),
                "grounded_entity_rate": grounded_rate, "n_candidate_pois": len(candidate_pois),
                "n_stops_day1": n_stops, "km_per_stop_day1": round(km_per_stop, 2) if km_per_stop else None,
                "retrieval_ms": round(retrieval_ms, 2),
            })

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("config")
        .agg(
            mean_recall_at_k=("recall_at_k", "mean"),
            mean_archetype_precision=("archetype_precision", "mean"),
            mean_grounded_entity_rate=("grounded_entity_rate", "mean"),
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
    ("archetype_precision", True),
    ("grounded_entity_rate", True),
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
                # is nothing to test: grounded_entity_rate is 1.0 by
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
    """Optional, opt-in: if ANTHROPIC_API_KEY is set, ask the live LLM to
    describe a POI in each test city with zero retrieved context (true
    'standard prompting'), then check whether every named entity in the
    response matches a real KB node. Skipped (returns None) without a key
    so the rest of the evaluation stays free/offline."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    from roamwise.agents.llm_client import AnthropicLLMClient
    idx = GraphIndex()
    known_names = {data["name"].lower() for _, data in idx.g.nodes(data=True) if data.get("type") == "POI"}
    llm = AnthropicLLMClient()
    probe_rows = []
    for dest_id, _, _, _ in TEST_QUERIES:
        city = idx.g.nodes[dest_id]["name"]
        prompt = f"List 5 specific points of interest to visit in {city}, one per line, name only."
        text = llm.complete(system="You are a travel assistant.", prompt=prompt)
        lines = [l.strip("-* ").lower() for l in text.splitlines() if l.strip()]
        matched = sum(any(name in line or line in name for name in known_names) for line in lines)
        probe_rows.append({"destination_id": dest_id, "named": len(lines), "matched_kb": matched})
    return pd.DataFrame(probe_rows)


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

    probe = run_llm_hallucination_probe()
    if probe is not None:
        probe.to_csv(HERE / "llm_hallucination_probe.csv", index=False)
        print("\nLive LLM hallucination probe:\n", probe.to_string())
    else:
        print("\n(No ANTHROPIC_API_KEY set -- skipped the optional live-LLM hallucination probe.)")
