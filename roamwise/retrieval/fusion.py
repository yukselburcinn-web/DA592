"""
Reciprocal Rank Fusion (RRF) across the three retrieval signals, and the
three retrieval configurations the proposal's comparative analysis calls for:

  fusion  = Semantic + Graph + Keyword   (primary architecture)
  hybrid  = Semantic + Keyword only      (ablation baseline, no Graph-RAG)
  standard = no retrieval at all         (lower-bound reference)

RRF score for a document = sum over retrievers r that returned it of
w_r / (k + rank_r(doc)), k=60 (the standard RRF constant). This is used instead
of a learned re-ranker because it needs no training data and is the
explicitly offered alternative in the proposal.

The per-retriever weight w_r is there because plain RRF -- every retriever
weighted 1.0 -- lets the *number* of retrievers in an evidence family decide
the answer. Semantic and Keyword are two readings of one corpus, so when a
query contains a category word they agree with each other for the same reason
rather than for two independent reasons: for "culture" they jointly ranked a
television channel above the Louvre, and their two correlated votes (0.0354)
outscored the one retriever that had it right (graph, rank 1, 0.0164). See
RETRIEVER_WEIGHTS below.
"""
from roamwise.retrieval.corpus import load_documents
from roamwise.retrieval import graph_search
from roamwise.retrieval.graph_search import GraphSearchIndex
from roamwise.retrieval.keyword_search import KeywordIndex
from roamwise.retrieval.semantic_search import SemanticIndex

RRF_K = 60

# Two evidence families, weighted equally -- not two retrievers out-voting one.
# Semantic and Keyword both rank the same document corpus (retrieval/corpus.py),
# so they are two readings of one signal and their agreement is not two
# independent confirmations. Graph traversal reads the structured knowledge
# graph instead, and is the only retriever that knows either the traveler's
# archetype or how well-known a place is. Leaving all three at 1.0 therefore
# does not weight evidence, it counts heads, and the document family wins every
# tie two votes to one: on "best culture enthusiast points of interest and
# experiences" that put `France 3`, a television channel, seventh and the Louvre
# seventeenth (#63). Giving graph the weight of the pair it is being compared
# against makes the two families count the same. Configs without graph
# (`hybrid`) are unaffected -- both their retrievers stay at 1.0.
# "chain" is the hour-aware traversal (#126, decision K4). Its weight was
# swept rather than picked, because a short dense list stacks its votes in RRF
# -- the same "head counting" problem #63 solved for the other three.
#
# Measured over 8 cells (2 cities x 4 archetypes) at the pool the orchestrator
# actually retrieves (`RETRIEVED_POIS_PER_DAY` x 3 days = 72), counting how
# many of the top 8 the chain surfaced:
#
#     weight   mean/8   range   in the pool
#      0.05     2.50     0-4       16.9
#      0.10     2.50     0-4       17.2
#      0.15     2.75     1-4       17.6   <- chosen
#      0.20     3.00     1-5       17.9
#      0.30     3.25     1-5       19.0
#      0.50     3.62     1-6       20.5
#      1.00     4.88     2-7       24.6
#      1.50     5.88     2-8       29.9   <- dominates
#
# KN-2's band is 2-4 of 8, and 0.15 is the only value whose *every* cell lands
# inside it rather than only its mean.
#
# It is far below the other three, and that is the shape of the list rather
# than a judgement about the evidence: `retrieve` asks each retriever for
# `top_k * 2`, so at a 72-POI pool the chain contributes 144 ranked documents
# where the graph contributes a category and an archetype list. Per-document
# weight has to be small for total influence to be comparable -- which is #63's
# finding restated, not a new one.
#
# This value replaces the 1.5 that phase 4 shipped. That number was swept at
# `top_k=8` -- eight retrieved POIs, not the seventy-two a three-day trip
# retrieves -- so it measured the top 8 of a pool the app never builds. At the
# real pool size 1.5 puts the chain at 5.88 of 8 and takes it to 8 of 8 in one
# cell, which is the domination KN-2 exists to catch. The correction is
# recorded here rather than quietly applied: the phase-4 measurement was wrong
# about its own conditions, and the sweep above is the one that decides.
#
# Inert while ROAMWISE_GRAPH_CHAIN is off: with no chain list, this key is
# never read.
RETRIEVER_WEIGHTS = {"graph": 2.0, "chain": 0.15, "semantic": 1.0, "keyword": 1.0}
DEFAULT_RETRIEVER_WEIGHT = 1.0


def reciprocal_rank_fusion(ranked_lists: list[list[dict]], top_k: int = 10) -> list[dict]:
    scores: dict[str, float] = {}
    doc_lookup: dict[str, dict] = {}
    sources: dict[str, set] = {}
    for list_name, ranked in ranked_lists:
        weight = RETRIEVER_WEIGHTS.get(list_name, DEFAULT_RETRIEVER_WEIGHT)
        for rank, doc in enumerate(ranked):
            doc_id = doc["doc_id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (RRF_K + rank + 1)
            doc_lookup[doc_id] = doc
            sources.setdefault(doc_id, set()).add(list_name)
    # Ties are the rule here, not the exception: every document returned by
    # exactly one retriever at the same rank scores identically, so a single
    # query routinely produces groups of documents on exactly 1/61. Sorting on
    # score alone left those groups in dict-insertion order -- i.e. decided by
    # which retriever happened to run first -- which is arbitrary and, being
    # arbitrary, silently favoured whichever family was listed first. Breaking
    # on popularity_score instead makes the tie-break say something: among
    # documents the retrievers rank equally, offer the better-known place. City
    # guides carry no popularity_score and sort last within their tie group,
    # which is right -- a guide paragraph is context, not a suggestion.
    ranked_ids = sorted(scores, key=lambda d: (-scores[d],
                                               -doc_lookup[d].get("popularity_score", 0.0),
                                               d))
    out = []
    for doc_id in ranked_ids[:top_k]:
        d = dict(doc_lookup[doc_id])
        d["rrf_score"] = scores[doc_id]
        d["retrieved_by"] = sorted(sources[doc_id])
        out.append(d)
    return out


class FusionRetriever:
    """Builds all three indices once and exposes `retrieve(query, config, ...)`
    where config in {"fusion", "hybrid", "standard"} -- this is the single
    object the evaluation harness swaps configs on for the comparative
    analysis in the final report."""

    def __init__(self):
        documents = load_documents()
        self.semantic = SemanticIndex(documents)
        self.keyword = KeywordIndex(documents)
        self.graph = GraphSearchIndex()

    def retrieve(self, query: str, config: str = "fusion", destination_id: str = None,
                 archetype: str = None, top_k: int = 8,
                 arrival_hub_id: str = None, start_date=None) -> list[dict]:
        """`arrival_hub_id` and `start_date` reach only the graph retrievers:
        one names where the traveler's day starts and the other which day it
        is, and the graph is the only component holding a relation from a
        starting point to anything, or one that opening hours can invalidate
        (#126). The semantic and keyword indexes read POI text, which says
        nothing about where anyone arrived or what day they arrived on."""
        if config == "standard":
            return []

        lists = [
            ("semantic", self.semantic.search(query, top_k=top_k * 2, destination_id=destination_id)),
            ("keyword", self.keyword.search(query, top_k=top_k * 2, destination_id=destination_id)),
        ]
        if config == "fusion":
            lists.append(("graph", self.graph.search(
                query, top_k=top_k * 2, destination_id=destination_id, archetype=archetype,
                arrival_hub_id=arrival_hub_id)))
            # A fourth list rather than more entries in the graph list: the
            # chain is a different kind of evidence and RRF has to be able to
            # weight it on its own (RETRIEVER_WEIGHTS["chain"], #126). Read
            # off the module rather than imported by value so a test -- and
            # the flag gate in phase 5 -- can flip it.
            if graph_search.CHAIN_ENABLED:
                lists.append(("chain", self.graph.chain_search(
                    destination_id=destination_id, top_k=top_k * 2,
                    arrival_hub_id=arrival_hub_id, start_date=start_date)))
        elif config != "hybrid":
            raise ValueError(f"unknown retrieval config: {config}")

        return reciprocal_rank_fusion(lists, top_k=top_k)


# Demo blocks below take their city from the catalogue rather than naming one:
# a hardcoded code prints nothing at all once that city stops shipping.
def _demo_city():
    import pandas as pd
    from pathlib import Path as _P
    d = _P(__file__).resolve().parents[1] / "data" / "destinations.csv"
    return pd.read_csv(d).destination_id.iloc[0]


if __name__ == "__main__":
    fr = FusionRetriever()
    query = "museums and landmarks near a transport hub for a culture-loving traveler"
    for config in ["fusion", "hybrid", "standard"]:
        print(f"\n=== {config} ===")
        for r in fr.retrieve(query, config=config, destination_id=_demo_city(), archetype="Culture Enthusiast", top_k=5):
            print(f"  {r.get('rrf_score', 0):.4f}  {r['doc_id']}  via={r.get('retrieved_by')}  {r['text'][:70]}")
