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
import os
import re

from roamwise.knowledge_graph.build_graph import (REACHABLE_MAX_MIN, SERVES_MAX_MIN,
                                                  GraphIndex)
from roamwise.optimization.routing import _opening_intervals

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


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# The hour-aware chain, on by default since #126 phase 5 passed its gate.
#
# It shipped behind a flag, default off, because it fires on the archetype
# query the orchestrator actually sends -- the anchor is the city centre, not a
# keyword -- so it changes every plan rather than only the queries that name a
# transport word. There was no "opt-in, therefore zero regression" story to
# lean on, and the measurement that decides whether the new default is better
# had not been made.
#
# It has now. Over the full 12/15/18-hour grid (24 cells, 2 cities x 4
# archetypes x 3 day lengths, `evaluation/graph_rag_baseline.py --full`):
#
#                       stops/day   km/stop   pref match   cats/day   closed
#     flag off            9.417      0.637      0.5858       3.167       0
#     flag on             9.444      0.628      0.5859       3.194       0
#
# Nothing regresses: distance per stop falls slightly, preference match is flat
# to four decimals, category variety per day rises, and arrivals at a closed
# venue stay at zero with both meals filled on all 72 days. That is the gate
# #126 set (KN-3), so the default moves.
#
# The flag stays, and stays honest in both directions: `ROAMWISE_GRAPH_CHAIN=0`
# turns the traversal off without a code change, which is what makes the
# comparison above reproducible by anyone who doubts it.
CHAIN_ENABLED = _env_flag("ROAMWISE_GRAPH_CHAIN", default=True)

# Only POIs whose hours were actually observed take part. 277 of the
# catalogue's 654 rows carry a category default -- "a museum is usually open
# 10-18" -- which no one checked; against those, a sequencing constraint is
# satisfied for free and therefore says nothing. Measured: without this filter
# the chain reaches 48-57% of the catalogue and separates nothing; with it,
# 18-28%. Coverage of the filter itself is PAR 237/371 (64%), BER 140/283
# (49%) -- a real limit, written up in REPORT section 5 rather than left for a
# reader to find.
CHAIN_REAL_HOURS_SOURCES = ("osm", "gmaps")

# `matrix_min` is the median over departures between 08:00 and 20:00
# (`pipeline/build_transit_matrix.py`), so a second leg departing at 22:00
# would be priced at a daytime rate that no night service delivers. Chains
# leaving after the matrix's own sample window are dropped rather than shipped
# mispriced: the graph should decline to make a claim its data cannot support.
# #126 records this limit as a REPORT footnote and measures 96% of chains
# departing before 20:00; enforcing it makes that 100% by construction, and
# costs the 4% rather than pricing them wrong.
CHAIN_LATEST_DEPARTURE_HOUR = 20.0

# Nightlife is excluded from the *second* leg for the same reason: an evening
# venue is exactly the one whose journey the daytime matrix cannot price. It
# stays eligible as the first leg, where the traveler arrives during the
# sampled window.
CHAIN_EXCLUDED_SECOND_LEG = ("nightlife",)


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


def _clock(hour: float) -> str:
    """An itinerary-clock hour as HH:MM. 26.0 is 02:00 the next morning, which
    is how `_opening_intervals` represents a venue open past midnight; it is
    shown as 02:00 rather than 26:00 because that is what a sign on the door
    would say."""
    hour = hour % 24
    return f"{int(hour):02d}:{int(round((hour % 1) * 60)) % 60:02d}"


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

    def chain_search(self, destination_id: str = None, top_k: int = 10,
                     arrival_hub_id: str = None, start_date=None) -> list[dict]:
        """The hour-aware chain, as a ranked list of ordinary POI documents.

            anchor -[SERVES <=15min]-> POI_a -[REACHABLE <=10min]-> POI_b

        valid only when **POI_b is still open once POI_a closes** -- the
        proposal's "sequencing constraint-aware activities", stated over two
        real edges rather than recomputed in Python (#126).

        A separate ranked list rather than more entries in `search`'s: it is a
        different kind of evidence, and RRF has to be able to weight it
        separately (`fusion.RETRIEVER_WEIGHTS["chain"]`). What it is *not* is a
        new document type -- both POIs come out as plain `type: "poi"` docs
        inheriting the chain's rank, so nothing downstream of fusion learns a
        new shape (decision K2).

        **Not raw two-hop expansion.** Taking the union of the two hops was
        measured and rejected: from Gare du Nord it returns 186 of Paris's 371
        POIs, from Gare de Lyon 200 -- half the catalogue, which is #113's 3 km
        radius wearing a different unit. The hour constraint is what makes this
        a query rather than a dump.

        Ranked by how well-known the pair is, not by how fast the chain is.
        The minutes are the *constraint* -- a chain either satisfies it or is
        not returned -- while relevance is which places these are, and RRF
        reads rank as relevance. Ranking by journey time would put the
        catalogue's most obscure adjacent pair first.

        `start_date` is what lets `_opening_intervals` resolve the verbatim OSM
        tag, which is the only thing that knows a venue shuts on Mondays; with
        no date it falls back to the coarse open/close pair, same as the
        router. That function is used rather than reimplemented: OSM grammar,
        lunchtime closures and past-midnight hours are solved there (#70).
        """
        if destination_id is None:
            return []
        anchor_id = self.anchor_for(destination_id, arrival_hub_id)
        ranked = self.chains(destination_id, arrival_hub_id=arrival_hub_id,
                             start_date=start_date)
        anchor_name = self.idx.g.nodes[anchor_id]["name"] if anchor_id in self.idx.g else anchor_id

        results = []
        for poi_a, to_a, poi_b, a_to_b, closes_a in ranked:
            # The path itself, not a claim about it. This text reaches the
            # traveler through "What the plan was grounded in"
            # (`views/itinerary.py`), so provenance becomes something they can
            # follow rather than something the report asserts.
            path = (f"{anchor_name} \u2192{to_a:.1f}min\u2192 {poi_a['name']} "
                    f"(closes {_clock(closes_a)}) "
                    f"\u2192{a_to_b:.1f}min\u2192 {poi_b['name']}")
            results.append(self._poi_to_doc(poi_a, path))
            results.append(self._poi_to_doc(poi_b, path))

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

    def chains(self, destination_id: str, arrival_hub_id: str = None, start_date=None,
               enforce_hours: bool = True, max_serves_min: float = None,
               max_reachable_min: float = None) -> list[tuple]:
        """The ranked chains themselves: `(poi_a, minutes_to_a, poi_b,
        minutes_a_to_b, poi_a_closing_hour)`, best first.

        `chain_search` flattens these into documents; the evaluation reads them
        as chains. Both go through here so there is one definition of what a
        valid chain is -- the invalid-chain rate #126 reports would be
        worthless if the evaluator computed validity with its own copy of this
        rule and the two drifted.

        `enforce_hours=False` keeps every pair the two edges admit, hour
        compatibility ignored. That is not a mode anything ships; it is the
        denominator the invalid-chain rate is measured against, and the only
        honest way to state what the constraint is worth.

        The two threshold overrides exist for the same reason: the sweep that
        chose them (`evaluation/chain_threshold_sweep.py`) has to ask what a
        different pair would have returned. Left None they are the shipped
        constants. Tightening here filters edges the graph already holds, so
        the sweep can vary both without rebuilding -- but it can only tighten,
        never loosen past what `build_graph` stored.
        """
        anchor_id = self.anchor_for(destination_id, arrival_hub_id)

        # Opening hours are parsed from a text tag, and one anchor produces
        # thousands of chains over a few hundred POIs -- so each POI's day is
        # resolved once per call rather than once per chain it appears in.
        day_cache: dict[str, list] = {}

        def intervals(poi):
            cached = day_cache.get(poi["poi_id"])
            if cached is None:
                # Only what opens on the itinerary day itself: the coarse
                # fallback also describes tomorrow, and "come back tomorrow"
                # is not a sequence.
                cached = day_cache[poi["poi_id"]] = [
                    (s, e) for s, e in _opening_intervals(poi, day_date=start_date) if s < 24]
            return cached

        ranked = []
        walked = self.idx.chains_from(
            anchor_id,
            max_serves_min=SERVES_MAX_MIN if max_serves_min is None else max_serves_min,
            max_reachable_min=(REACHABLE_MAX_MIN if max_reachable_min is None
                               else max_reachable_min))
        for poi_a, to_a, poi_b, a_to_b in walked:
            if poi_b.get("category") in CHAIN_EXCLUDED_SECOND_LEG:
                continue
            if (poi_a.get("hours_source") not in CHAIN_REAL_HOURS_SOURCES
                    or poi_b.get("hours_source") not in CHAIN_REAL_HOURS_SOURCES):
                continue
            open_a = intervals(poi_a)
            if not open_a:
                continue
            closes_a = max(end for _, end in open_a)
            if enforce_hours:
                if closes_a > CHAIN_LATEST_DEPARTURE_HOUR:
                    continue
                arrival = closes_a + a_to_b / 60.0
                stay = poi_b.get("avg_visit_minutes", 0) / 60.0
                # Open on arrival *and* still open long enough to be visited.
                # Arriving five minutes before the doors shut is not "you can
                # do this one after that one", which is the whole claim of the
                # edge.
                if not any(start <= arrival and arrival + stay <= end
                           for start, end in intervals(poi_b)):
                    continue
            ranked.append((
                -max(poi_a.get("popularity_score", 0.0), poi_b.get("popularity_score", 0.0)),
                to_a + a_to_b, poi_a["poi_id"], poi_b["poi_id"],
                poi_a, to_a, poi_b, a_to_b, closes_a))

        ranked.sort(key=lambda row: row[:4])
        return [(poi_a, to_a, poi_b, a_to_b, closes_a)
                for _, _, _, _, poi_a, to_a, poi_b, a_to_b, closes_a in ranked]

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
