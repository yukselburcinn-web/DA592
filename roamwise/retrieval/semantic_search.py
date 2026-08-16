"""
Semantic search component of the Fusion RAG layer.

The proposal names FAISS/ChromaDB dense-vector retrieval over transformer
embeddings. This prototype uses TF-IDF projected into a latent-semantic
space (truncated SVD) instead of a downloaded transformer encoder: it needs
no multi-hundred-MB model file, trains in milliseconds, and still captures
synonymy/topic similarity beyond exact keyword overlap (e.g. "cheap eats"
retrieving a market POI) -- which is what distinguishes it from the BM25
keyword layer. Swapping in `sentence-transformers` + FAISS later only
requires reimplementing `SemanticIndex.encode`; every caller goes through
`SemanticIndex.search`.
"""
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from retrieval.corpus import load_documents


class SemanticIndex:
    def __init__(self, documents: list[dict] = None, n_components: int = 64):
        self.documents = documents if documents is not None else load_documents()
        texts = [d["text"] for d in self.documents]
        self.vectorizer = TfidfVectorizer(stop_words="english", max_df=0.9, min_df=1)
        tfidf = self.vectorizer.fit_transform(texts)
        n_components = min(n_components, tfidf.shape[1] - 1, tfidf.shape[0] - 1)
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.embeddings = self.svd.fit_transform(tfidf)

    def search(self, query: str, top_k: int = 10, destination_id: str = None) -> list[dict]:
        q_tfidf = self.vectorizer.transform([query])
        q_vec = self.svd.transform(q_tfidf)
        sims = cosine_similarity(q_vec, self.embeddings)[0]
        order = np.argsort(-sims)
        results = []
        for i in order:
            doc = self.documents[i]
            if destination_id and doc.get("destination_id") != destination_id:
                continue
            results.append({**doc, "score": float(sims[i])})
            if len(results) >= top_k:
                break
        return results


if __name__ == "__main__":
    idx = SemanticIndex()
    for r in idx.search("cheap places to eat near the old town at night", top_k=5):
        print(f"{r['score']:.3f}  {r['doc_id']}  {r['text'][:80]}")
