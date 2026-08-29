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
import re

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

# The router matched a category only when the query used the catalogue's own
# taxonomy word, which made it brittle in exactly the place it is asked to be
# robust: nobody types "religion". Both the evaluation grid and the query the
# orchestrator builds phrase that category as "places of worship", so a query
# *about* churches routed as though it named no category at all, fell back to
# the generic archetype list, and -- once that list began mixing categories by
# prominence (#63) -- ranked Culture Enthusiast's lowest-weighted category
# (religion, 0.6) near the bottom of its own answer key.
#
# Only phrasings this codebase actually produces or that a traveler plausibly
# types are listed. This is a routing aid, not a thesaurus: a term here makes
# the graph retriever *consider* a category, and a wrong guess costs recall on
# the category it displaces.
CATEGORY_SYNONYMS = {
    "religion": ["place of worship", "places of worship", "church", "cathedral",
                 "mosque", "synagogue", "temple"],
    "museum": ["gallery", "galleries"],
    "landmark": ["monument", "sight"],
    "history": ["historic", "historical", "heritage"],
    "nature": ["park", "parks", "garden", "gardens", "green space"],
    "nightlife": ["bar", "bars", "club", "clubs", "pub"],
    "food": ["restaurant", "eat", "dining", "meal"],
    "shopping": ["shop", "market", "boutique"],
}
TRANSPORT_KEYWORDS = ["near", "walk", "close to", "airport", "station", "transport", "hub"]


_TERM_PATTERNS: dict[str, "re.Pattern"] = {}


def _names(term: str, text: str) -> bool:
    """Does `text` use `term` as a word (bare or pluralised)?

    Whole words, not substrings. Matching on substrings is what a short
    synonym list cannot survive: "pub" is inside "public", so
    "accessible via late-night public transit" would route as a nightlife
    query; "bar" is inside "barrier", "park" inside "parking", "eat" inside
    "theatre". The trailing `s?` keeps the plural a traveler actually types
    ("museums", "landmarks") matching without needing both forms listed.
    """
    pattern = _TERM_PATTERNS.get(term)
    if pattern is None:
        pattern = _TERM_PATTERNS[term] = re.compile(rf"\b{re.escape(term)}s?\b")
    return pattern.search(text) is not None


def categories_in(text: str) -> list[str]:
    """Every catalogue category `text` names, by taxonomy word or synonym.

    Order follows CATEGORY_KEYWORDS rather than position in the text, so the
    result is stable for a given query whatever order the words appear in.
    """
    lowered = text.lower()
    return [category for category in CATEGORY_KEYWORDS
            if _names(category, lowered)
            or any(_names(synonym, lowered) for synonym in CATEGORY_SYNONYMS.get(category, ()))]


class GraphSearchIndex:
    def __init__(self, graph_index: GraphIndex = None):
        self.idx = graph_index if graph_index is not None else GraphIndex()

    def anchor_for(self, destination_id: str, arrival_hub_id: str = None) -> str:
        """Where the traveler's day starts, as a graph node id.

        The gateway they land at when they named one, the city centre
        otherwise. Both carry `SERVES` edges, so the traversal that reads this
        does not care which it got -- which is the point of resolving it here
        rather than branching at the call site.

        The centre is the right default rather than a placeholder: `Arriving
        at` defaults to "Already in the city", so most sessions name no hub at
        all, and the router already starts every day but the first from the
        centre (#32). It is where the trip actually begins.

        An id this city does not hold falls back to the centre rather than
        raising. That is the same rule `RouterAgent` applies to a stale hub id
        (a city switched after the gateway was picked) and the same rule an
        unknown travel mode gets: planning continues.
        """
        if arrival_hub_id and any(hub["transport_id"] == arrival_hub_id
                                  for hub in self.idx.city_transport(destination_id)):
            return arrival_hub_id
        return destination_id

    def search(self, query: str, top_k: int = 10, destination_id: str = None,
               archetype: str = None, arrival_hub_id: str = None) -> list[dict]:
        """`arrival_hub_id` is carried but not yet dispatched on: the anchored
        chain traversal it selects lands in #126 phase 4, behind a flag. It is
        wired ahead of that traversal because the wire is the part this repo
        keeps getting wrong -- `day_start_hour` (#59) and `start_date` on the
        LangGraph path (#76) were both parameters that existed on the agent
        below and could not be reached from the app above.
        """
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

        # Every category the query names, not just the first one in
        # CATEGORY_KEYWORDS order. The count is what distinguishes the two
        # kinds of question this retriever gets asked, and taking `next(...)`
        # collapsed them: "the best museums, landmarks, history sites, culture
        # venues and places of worship" reported itself as a museum query
        # purely because `museum` sorts first in the keyword list.
        matched_categories = categories_in(q)
        near_transport = any(kw in q for kw in TRANSPORT_KEYWORDS)

        # One named category is a constraint ("museums near a transport hub");
        # several are a preference profile, which is what the orchestrator
        # sends. A constraint outranks a profile prior, so it leads -- the
        # archetype list used to lead unconditionally, which was invisible only
        # while that list was a single category deep. Once it began mixing
        # categories by prominence (#63) a single-category query started
        # getting the archetype's other categories ahead of the one it asked
        # for, and recall against a single-category answer key fell with it.
        category_leads = len(matched_categories) == 1
        constraint = matched_categories[0] if matched_categories else None

        def category_matches():
            if constraint is None:
                return
            if near_transport:
                for poi in self.idx.multi_hop_transport_to_poi(destination_id, constraint):
                    yield self._poi_to_doc(
                        poi, f"{poi.get('nearest_hub_minutes')}min from nearest transport hub")
                return
            # Sorted rather than left in graph-edge order: this is a ranked
            # list feeding RRF, and an unranked category dump gave the
            # catalogue's storage order the weight of a relevance signal.
            for poi in sorted(self.idx.city_pois(destination_id, category=constraint),
                              key=lambda p: -p["popularity_score"]):
                yield self._poi_to_doc(poi, f"category match: {constraint}")

        def archetype_matches():
            if not matched_archetype:
                return
            for poi in self.idx.archetype_preferred_pois(matched_archetype, destination_id, top_k=top_k):
                yield self._poi_to_doc(poi, f"preferred by {matched_archetype} travelers")

        first, second = ((category_matches, archetype_matches) if category_leads
                         else (archetype_matches, category_matches))
        results.extend(first())
        results.extend(second())

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
            # Same field the corpus documents carry, so a POI fused from the
            # graph and from the document corpus tie-breaks identically
            # whichever retriever's copy of it fusion happens to keep (#63).
            "popularity_score": poi.get("popularity_score", 0.0),
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
