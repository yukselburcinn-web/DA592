"""Turning a traveler archetype into a retrieval query.

The orchestrator used to build one by interpolating the archetype's *label*:

    f"best {archetype.lower()} points of interest and experiences"
    -> "best culture enthusiast points of interest and experiences"

That is a label, not a search string, and both document retrievers fell for it
(#63). BM25 matched the literal token `culture` plus the filler words (`best`,
`points`, `interest`), so its top hit for a Paris culture traveler was
`France 3` -- a television channel whose description happens to say "culture" --
followed by `Cimetiere de Montrouge` and `Sorbonne Nouvelle-Paris 3`. Semantic
search fared no better: `Tiki playa`, `Madame Guan`, `Cafe Cox`, because short
bar and restaurant snippets sit closer to a vague tourism sentence than a
Wikipedia lead paragraph does. Between them the two retrievers placed *none* of
Paris's twelve best-known POIs in their top 48.

So the query is built from what the archetype actually wants -- the categories
on its `PREFERS` edges -- rather than from what it is called. The archetype
itself still travels to `GraphSearchIndex.search` as a parameter, so nothing
depends on its name surviving into the text.
"""
import pandas as pd

from roamwise.knowledge_graph.build_graph import CATEGORY_AFFINITY, DATA_DIR

# One natural phrase per catalogue category. These read as something a person
# would type because the corpus they are matched against is prose: POI
# documents are "{name} ({category}) in {city}: {description}", so the category
# word is present to be matched, and the plural noun phrase is what the
# sentence-embedding model was trained on.
CATEGORY_PHRASE = {
    "museum": "museums", "landmark": "landmarks", "history": "history sites",
    "nature": "nature spots", "nightlife": "nightlife venues", "shopping": "shopping",
    "food": "food markets and restaurants", "beach": "beaches",
    "culture": "culture venues", "religion": "places of worship",
}
_CATALOGUE_CATEGORIES = None


def _catalogue_categories() -> set:
    """Which categories the catalogue actually holds, cached for the process.

    A query term for an empty category is a term that can only match the wrong
    thing. Two of the seven archetypes asked for `beaches` in cities that hold
    zero beach POIs -- "the best beaches, nature spots, ..." for Beach & Relax
    and "the best nature spots, beaches and history sites" for Nature &
    Adventure (#79). Measured, dropping it is a small effect rather than a
    dramatic one: 21 to 23 of each 24-POI pool come back unchanged. It is
    removed because a query should say what the catalogue can answer, not
    because retrieval was collapsing.
    """
    global _CATALOGUE_CATEGORIES
    if _CATALOGUE_CATEGORIES is None:
        _CATALOGUE_CATEGORIES = set(
            pd.read_csv(DATA_DIR / "poi.csv", usecols=["category"])["category"].unique())
    return _CATALOGUE_CATEGORIES


def archetype_query(archetype: str) -> str:
    """The categories this archetype prefers, strongest first, as a phrase.

    Ordered by affinity weight so the leading terms are the ones the traveler
    cares most about -- BM25 and the embedding both read the whole string, but
    a query that opens with what matters is the one a person would write.

    Categories the catalogue does not hold are left out; see
    `_catalogue_categories`.
    """
    affinities = CATEGORY_AFFINITY.get(archetype)
    if not affinities:
        # An archetype with no profile in the graph still has to retrieve
        # something; a generic sightseeing query is the honest fallback.
        return "the best places to visit in this city"
    held = _catalogue_categories()
    phrases = [CATEGORY_PHRASE[category]
               for category, _ in sorted(affinities.items(), key=lambda kv: -kv[1])
               if category in CATEGORY_PHRASE and category in held]
    if not phrases:
        return "the best places to visit in this city"
    if len(phrases) == 1:
        return f"the best {phrases[0]} to visit in this city"
    return f"the best {', '.join(phrases[:-1])} and {phrases[-1]} to visit in this city"


if __name__ == "__main__":
    for name in sorted(CATEGORY_AFFINITY):
        print(f"{name:<20} {archetype_query(name)}")
