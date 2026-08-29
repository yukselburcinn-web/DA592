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
RETRIEVER_WEIGHTS = {"graph": 2.0, "semantic": 1.0, "keyword": 1.0}
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
                 arrival_hub_id: str = None) -> list[dict]:
        """`arrival_hub_id` reaches only the graph retriever: it names where
        the traveler's day starts, and the graph is the only component that
        holds a relation from a starting point to anything (#126). The
        semantic and keyword indexes read POI text, which says nothing about
        where anyone arrived."""
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
