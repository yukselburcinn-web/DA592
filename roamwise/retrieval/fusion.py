"""
Reciprocal Rank Fusion (RRF) across the three retrieval signals, and the
three retrieval configurations the proposal's comparative analysis calls for:

  fusion  = Semantic + Graph + Keyword   (primary architecture)
  hybrid  = Semantic + Keyword only      (ablation baseline, no Graph-RAG)
  standard = no retrieval at all         (lower-bound reference)

RRF score for a document = sum over retrievers r that returned it of
1 / (k + rank_r(doc)), k=60 (the standard RRF constant). This is used instead
of a learned re-ranker because it needs no training data and is the
explicitly offered alternative in the proposal.
"""
from retrieval.corpus import load_documents
from retrieval.graph_search import GraphSearchIndex
from retrieval.keyword_search import KeywordIndex
from retrieval.semantic_search import SemanticIndex

RRF_K = 60


def reciprocal_rank_fusion(ranked_lists: list[list[dict]], top_k: int = 10) -> list[dict]:
    scores: dict[str, float] = {}
    doc_lookup: dict[str, dict] = {}
    sources: dict[str, set] = {}
    for list_name, ranked in ranked_lists:
        for rank, doc in enumerate(ranked):
            doc_id = doc["doc_id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank + 1)
            doc_lookup[doc_id] = doc
            sources.setdefault(doc_id, set()).add(list_name)
    ranked_ids = sorted(scores, key=lambda d: -scores[d])
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
                 archetype: str = None, top_k: int = 8) -> list[dict]:
        if config == "standard":
            return []

        lists = [
            ("semantic", self.semantic.search(query, top_k=top_k * 2, destination_id=destination_id)),
            ("keyword", self.keyword.search(query, top_k=top_k * 2, destination_id=destination_id)),
        ]
        if config == "fusion":
            lists.append(("graph", self.graph.search(
                query, top_k=top_k * 2, destination_id=destination_id, archetype=archetype)))
        elif config != "hybrid":
            raise ValueError(f"unknown retrieval config: {config}")

        return reciprocal_rank_fusion(lists, top_k=top_k)


if __name__ == "__main__":
    fr = FusionRetriever()
    query = "museums and landmarks near a transport hub for a culture-loving traveler"
    for config in ["fusion", "hybrid", "standard"]:
        print(f"\n=== {config} ===")
        for r in fr.retrieve(query, config=config, destination_id="ROM", archetype="Culture Enthusiast", top_k=5):
            print(f"  {r.get('rrf_score', 0):.4f}  {r['doc_id']}  via={r.get('retrieved_by')}  {r['text'][:70]}")
