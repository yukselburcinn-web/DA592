"""Keyword (BM25) search component of the Fusion RAG layer -- sparse lexical
retrieval to catch exact-match entities (names, landmark titles) that the
latent-semantic layer can underweight."""
from rank_bm25 import BM25Okapi

from roamwise.retrieval.corpus import load_documents, tokenize


class KeywordIndex:
    def __init__(self, documents: list[dict] = None):
        self.documents = documents if documents is not None else load_documents()
        self.tokenized = [tokenize(d["text"]) for d in self.documents]
        self.bm25 = BM25Okapi(self.tokenized)

    def search(self, query: str, top_k: int = 10, destination_id: str = None) -> list[dict]:
        scores = self.bm25.get_scores(tokenize(query))
        order = scores.argsort()[::-1]
        results = []
        for i in order:
            doc = self.documents[i]
            if destination_id and doc.get("destination_id") != destination_id:
                continue
            if scores[i] <= 0:
                continue
            results.append({**doc, "score": float(scores[i])})
            if len(results) >= top_k:
                break
        return results


if __name__ == "__main__":
    idx = KeywordIndex()
    for r in idx.search("Sagrada Familia Gaudi", top_k=5):
        print(f"{r['score']:.3f}  {r['doc_id']}  {r['text'][:80]}")
