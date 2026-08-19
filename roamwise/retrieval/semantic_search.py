"""
Semantic search component of the Fusion RAG layer.

The proposal names FAISS/ChromaDB dense-vector retrieval over transformer
embeddings. This uses `sentence-transformers` (`all-MiniLM-L6-v2`, a small
~80MB model) for the embeddings and FAISS (`IndexFlatIP` over L2-normalized
vectors, i.e. exact cosine-similarity search) for the index -- the two
libraries the proposal names for this exact purpose (issue #4; see
REPORT.md §3.3 for why the project originally shipped a TF-IDF+LSA stand-in
instead, and the trade-offs of this swap).

Every caller goes through `SemanticIndex.search`, and the constructor still
takes an optional `documents` list -- this is the same public interface the
TF-IDF+LSA version had, so `retrieval/fusion.py` and everything upstream of
it needed zero changes.
"""
from pathlib import Path

import faiss
import numpy as np

from retrieval.corpus import load_documents

DEFAULT_MODEL = "all-MiniLM-L6-v2"


class SemanticIndex:
    def __init__(self, documents: list[dict] = None, model_name: str = DEFAULT_MODEL):
        from sentence_transformers import SentenceTransformer

        self.documents = documents if documents is not None else load_documents()
        self.model = SentenceTransformer(model_name)

        texts = [d["text"] for d in self.documents]
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        self.embeddings = np.asarray(embeddings, dtype="float32")

        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def search(self, query: str, top_k: int = 10, destination_id: str = None) -> list[dict]:
        q_vec = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)
        q_vec = np.asarray(q_vec, dtype="float32")

        # Search the full ranking (not just top_k) so post-hoc destination_id
        # filtering can't under-fill the result list -- same behavior the
        # previous TF-IDF+LSA implementation had via a full argsort.
        sims, order = self.index.search(q_vec, len(self.documents))
        sims, order = sims[0], order[0]

        results = []
        for score, i in zip(sims, order):
            doc = self.documents[i]
            if destination_id and doc.get("destination_id") != destination_id:
                continue
            results.append({**doc, "score": float(score)})
            if len(results) >= top_k:
                break
        return results


if __name__ == "__main__":
    idx = SemanticIndex()
    for r in idx.search("cheap places to eat near the old town at night", top_k=5):
        print(f"{r['score']:.3f}  {r['doc_id']}  {r['text'][:80]}")
