"""
Graph-RAG component of the Fusion RAG layer.

Unlike the semantic/keyword layers, this does not rank a flat document list --
it traverses the knowledge graph to answer relational, multi-hop questions
("landmarks within walking distance of a transport hub", "POIs this traveler
archetype prefers in this city") that document retrieval cannot express. A
lightweight keyword router inspects the query for the relation it implies and
dispatches to the matching GraphIndex traversal; results are wrapped into the
same doc-like shape (doc_id/text/score) as the other two retrievers so
fusion.py can merge all three with reciprocal rank fusion.
"""
from roamwise.knowledge_graph.build_graph import GraphIndex

ARCHETYPE_KEYWORDS = {
    "Culture Enthusiast": ["culture", "museum", "history", "art"],
    "Beach & Relax": ["beach", "relax", "chill"],
    "Budget Backpacker": ["budget", "cheap", "backpack"],
    "Luxury Traveler": ["luxury", "upscale", "high-end"],
    "Nightlife Seeker": ["nightlife", "bar", "club", "party"],
    "Nature & Adventure": ["nature", "adventure", "outdoor", "hike"],
    "Family Traveler": ["family", "kids", "children"],
}
CATEGORY_KEYWORDS = [
    "museum", "landmark", "history", "religion", "nature", "food", "nightlife",
    "shopping", "culture", "beach",
]
TRANSPORT_KEYWORDS = ["near", "walk", "close to", "airport", "station", "transport", "hub"]


class GraphSearchIndex:
    def __init__(self, graph_index: GraphIndex = None):
        self.idx = graph_index if graph_index is not None else GraphIndex()

    def search(self, query: str, top_k: int = 10, destination_id: str = None, archetype: str = None) -> list[dict]:
        if destination_id is None:
            return []
        q = query.lower()
        results: list[dict] = []

        matched_archetype = archetype
        if matched_archetype is None:
            for arch, kws in ARCHETYPE_KEYWORDS.items():
                if any(kw in q for kw in kws):
                    matched_archetype = arch
                    break

        matched_category = next((c for c in CATEGORY_KEYWORDS if c in q), None)
        near_transport = any(kw in q for kw in TRANSPORT_KEYWORDS)

        if matched_archetype:
            for poi in self.idx.archetype_preferred_pois(matched_archetype, destination_id, top_k=top_k):
                results.append(self._poi_to_doc(poi, f"preferred by {matched_archetype} travelers"))

        if near_transport and matched_category:
            for poi in self.idx.multi_hop_transport_to_poi(destination_id, matched_category):
                results.append(self._poi_to_doc(poi, f"{poi.get('nearest_hub_km')}km from nearest transport hub"))
        elif matched_category:
            for poi in self.idx.city_pois(destination_id, category=matched_category):
                results.append(self._poi_to_doc(poi, f"category match: {matched_category}"))

        if not results:
            # fall back: return the city's top-rated POIs as generic graph context
            for poi in sorted(self.idx.city_pois(destination_id), key=lambda p: -p["popularity_score"])[:top_k]:
                results.append(self._poi_to_doc(poi, "top-rated in city"))

        # dedupe by poi_id, preserve order, attach descending pseudo-scores for RRF
        seen = set()
        deduped = []
        for r in results:
            if r["poi_id"] in seen:
                continue
            seen.add(r["poi_id"])
            deduped.append(r)
        for rank, r in enumerate(deduped[:top_k]):
            r["score"] = 1.0 - rank / max(len(deduped), 1)
        return deduped[:top_k]

    @staticmethod
    def _poi_to_doc(poi: dict, reason: str) -> dict:
        return {
            "doc_id": f"poi::{poi['poi_id']}",
            "type": "poi",
            "destination_id": poi.get("destination_id"),
            "poi_id": poi["poi_id"],
            "name": poi["name"],
            "text": f"{poi['name']} ({poi.get('category')}): {poi.get('description', '')} [graph: {reason}]",
        }


# Demo blocks below take their city from the catalogue rather than naming one:
# a hardcoded code prints nothing at all once that city stops shipping.
def _demo_city():
    import pandas as pd
    from pathlib import Path as _P
    d = _P(__file__).resolve().parents[1] / "data" / "destinations.csv"
    return pd.read_csv(d).destination_id.iloc[0]


if __name__ == "__main__":
    idx = GraphSearchIndex()
    for r in idx.search("museums within walking distance of a transport hub", destination_id=_demo_city(), top_k=5):
        print(f"{r['score']:.2f}  {r['doc_id']}  {r['text'][:90]}")
