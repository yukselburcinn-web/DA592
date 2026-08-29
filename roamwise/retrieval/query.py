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
#
# `religion` names church buildings as well as places of worship, and the
# wording is chosen by measurement rather than by ear (#113). The tokenizer
# does no stemming, so a phrase matches only the words it literally contains,
# and the phrase was asking BM25 for words the corpus does not use: across the
# 654 POI documents `worship` occurs in 4 and `places` in 5, against `church`
# in 69 and `churches` in 9. Reachable religion POIs out of 84, fused, with the
# graph ranking of `build_graph.rank_preferred` in place:
#
#   places of worship                        14      church                  9 docs
#   churches                                 13      churches                9 docs
#   churches and places of worship           22
#   church buildings and places of worship   30   <- singular `church` matches
#
# "church buildings" is the clumsiest of the four to read and the only one that
# carries the singular token, which is the one the corpus actually uses. This
# string is never shown to a traveler -- it goes from the orchestrator straight
# into retrieval -- so it is worth 8 more reachable POIs.
#
# The graph ranking does the heavier lifting here (1 of 84 to 14 on its own,
# against this phrase's 1 to 4 on its own). Both are kept because they are
# independent faults, and this one is what the docstring above already asked
# for: the category word present to be matched.
# #113's principle, applied to every entry rather than only to `religion`
# (#123). `tokenize` does no stemming, so a phrase matches only the words it
# literally contains, and two more entries turned out to be in exactly the
# state `religion` had been in. Document frequency over the 654-POI corpus:
#
#     category    phrase             most common token   docs
#     landmark    landmarks          landmarks              2   <- invisible
#     museum      museums            museums               17   <- invisible
#     nightlife   nightlife venues   nightlife             42
#     culture     culture venues     culture               92
#     history     history sites      history               83
#     nature      nature spots       nature                76
#     shopping    shopping           shopping              32
#
# The singular is what the documents use -- `museum` 127, `landmark` 117 --
# and neither plural reached it. They went unnoticed because the graph carries
# both at 0.9 and 1.0 affinity; `religion` surfaced only because 0.6 could not
# carry it.
#
# The fix is not free, and that is the part worth writing down. The pool is
# strictly zero-sum at a fixed `top_k`: adding the singular forms is worth 27
# newly reachable POIs and costs 11, and left alone it takes `religion` from
# 30 back to 23 -- a quarter of what #113 had just won. The phrasings below
# ship together with `build_graph.PREFERENCE_QUOTA_EXPONENT`, swept jointly;
# see `evaluation/category_phrase_sweep.py` and that constant.
#
# `religion` moved too, and not because it was broken -- #113 had already fixed
# it. It moved because it is what the museum and landmark fixes were about to
# break. Naming the specific buildings the corpus names (`cathedral` 15,
# `basilica` 7, `chapel` 9 alongside `church` 69) is what let the category hold
# its ground against a stronger `museum`: without it, `religion` fell to 23 of
# the 30 POIs #113 won at the three-day pool and to 48 of 56 at five days. With
# it, 39 and 56.
#
# `history`, `nature`, `nightlife`, `culture`, `shopping` and `food` were
# audited too and left alone: each already carries a token the corpus uses
# often (83, 76, 42, 92, 32, 62 documents), and the sweep measured every
# candidate replacement as a net loss -- "history and historic sites" costs
# `history` 5 reachable POIs by pulling the pool towards `site`. Their dead
# tokens (`spots` 0, `venues` 3, `sites` 7, `markets` 1) are harmless rather
# than worth fixing: BM25 scores what matches, and a token nothing contains
# adds nothing instead of subtracting.
#
# None of these strings is ever shown to a traveler -- they go from the
# orchestrator straight into retrieval -- which is why a clumsy phrase that
# matches beats a fluent one that does not.
CATEGORY_PHRASE = {
    "museum": "museum collections", "landmark": "landmark sights",
    "history": "history sites",
    "nature": "nature spots", "nightlife": "nightlife venues", "shopping": "shopping",
    "food": "food markets and restaurants", "beach": "beaches",
    "culture": "culture venues",
    "religion": "church cathedral basilica and chapel buildings",
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
