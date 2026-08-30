"""Should the chain's second leg be 10 or 15 minutes -- and at what weight?
(issue #144)

#126's phase-5 sweep shipped `REACHABLE_MAX_MIN = 10` while its own table
showed 15 was defensible: precision is flat (0.624 -> 0.631) and recall gains
28% relative (0.232 -> 0.298). What stopped it was phase 4's acceptance
criterion, "the returned POI set does not exceed 25% of the catalogue" -- and
Paris already sits at 28.3% at ten minutes.

The reason the decision was deferred rather than taken is that the missing
measurement is not coverage. The chain enters fusion as a fourth ranked list at
`RETRIEVER_WEIGHTS["chain"] = 0.15`, tuned to keep it in KN-2's 2-4 band of the
top eight. Growing the chain set moves that band too, so **threshold and weight
have to be swept together** -- the same lesson #123 learned about phrasing and
quota, and #143 confirmed about quota and pool size.

What it reports, per (threshold, weight):

  chain_in_top8   per archetype, and the worst of them -- KN-2's own statistic.
                  The band is 2-4: 0 means fusion drowns the chain, 5+ means
                  the chain is deciding the answer alone, which is the
                  head-counting failure #63 fixed for the other retrievers.
  share           of each city's catalogue, how much the chain can reach. This
                  is the number phase 4's 25% criterion is about.
  chain_recall    normalized recall on the chain tier of the comparative
                  analysis -- the queries the relation exists for -- for fusion
                  and for hybrid. The ratio between them is what says whether
                  the chain is still *discriminating*: raw recall falls as the
                  threshold rises, because the chain tier's answer key is the
                  traversal's own output and loosening the walk grows the key
                  along with the result.

Run:  python -m roamwise.evaluation.chain_threshold_weight_sweep
Writes `chain_threshold_weight_sweep.csv`. Nothing here mutates a shipped
constant permanently; both are patched per configuration and restored.
"""
import collections
import datetime
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd

from roamwise.evaluation import comparative_analysis as ca
from roamwise.knowledge_graph import build_graph
from roamwise.knowledge_graph.build_graph import CATEGORY_AFFINITY, GraphIndex
from roamwise.retrieval import fusion as fusion_module
from roamwise.retrieval.fusion import FusionRetriever
from roamwise.retrieval import graph_search as graph_search_module
from roamwise.retrieval.graph_search import GraphSearchIndex
from roamwise.retrieval.query import archetype_query

HERE = Path(__file__).parent
SWEEP_CSV = HERE / "chain_threshold_weight_sweep.csv"

CITIES = ["PAR", "BER"]
PLAN_CITY = "PAR"
N_DAYS = 3
RETRIEVED_POIS_PER_DAY = 24
CHAIN_RETRIEVER = "chain"
# The date #126's sweep used, kept so the numbers are comparable to
# `chain_threshold_sweep.csv` rather than only to each other.
CHAIN_DATE = datetime.date(2026, 9, 24)

THRESHOLDS = [10.0, 15.0]
# Around the shipped 0.15, and down to near-zero. The band is what decides, so
# the grid has to be wide enough to leave it in both directions -- and at 15
# minutes the question becomes whether *any* weight can hold it.
WEIGHTS = [0.02, 0.05, 0.10, 0.15, 0.25, 0.40]


def _chain_share(city: str) -> tuple[float, int]:
    """How much of a city's catalogue the chain can reach, and how many POIs.

    Reads the threshold from `graph_search`'s module global, which `main`
    patches -- `chain_search` takes no threshold argument, and the walk it
    delegates to resolves the constant at call time.
    """
    graph = GraphSearchIndex()
    catalogue = graph.idx.city_pois(city)
    if not catalogue:
        return None, 0
    reached = {d["poi_id"] for d in graph.chain_search(
        destination_id=city, top_k=len(catalogue), start_date=CHAIN_DATE)}
    return len(reached) / len(catalogue), len(reached)


def _kn2(retriever: FusionRetriever) -> dict:
    """KN-2's statistic: how many of the top eight came from the chain."""
    shares = {}
    for archetype in sorted(CATEGORY_AFFINITY):
        # No `arrival_hub_id` / `start_date`: `graph_rag_baseline.py`'s
        # retrieval_section does not pass them, and KN-2's band was set on that
        # call. Passing them here measures a different, more chain-heavy query
        # and the verdict would not be comparable to the checkpoint.
        results = retriever.retrieve(
            archetype_query(archetype), config="fusion", destination_id=PLAN_CITY,
            archetype=archetype, top_k=N_DAYS * RETRIEVED_POIS_PER_DAY)
        shares[archetype] = sum(1 for r in results[:8]
                                if CHAIN_RETRIEVER in r.get("retrieved_by", []))
    return shares


def _chain_tier_recall() -> dict:
    """Normalized recall on the chain tier, per configuration.

    Only the chain queries: the other two tiers do not exercise the relation
    and running all 67 would cost minutes per threshold to measure nothing
    this decision reads.
    """
    # `comparative_analysis` caches its own GraphSearchIndex for the chain
    # queries, so it holds the previous threshold's walk unless dropped -- the
    # same per-process caching that CLAUDE.md gotcha #14 warns about, one layer
    # further out. Without this the tier reports identical recall at both
    # thresholds, which is how this sweep first read.
    ca._chain_index.cache_clear()
    saved = ca.TEST_QUERIES
    ca.TEST_QUERIES = ca.build_chain_queries()
    try:
        df = ca.run_comparative_analysis()
    finally:
        ca.TEST_QUERIES = saved
    return {config: round(group.normalized_recall.mean(), 4)
            for config, group in df.groupby("config")}


def main():
    shipped_threshold = graph_search_module.REACHABLE_MAX_MIN
    shipped_weight = fusion_module.RETRIEVER_WEIGHTS[CHAIN_RETRIEVER]
    rows = []

    try:
        for threshold in THRESHOLDS:
            # Three things have to move together, and each one caught this
            # sweep out in turn.
            #
            # `build_graph` is where the REACHABLE edges are *built*:
            # `np.nonzero(block <= REACHABLE_MAX_MIN)` filters them into the
            # graph, so a threshold the graph was not built with cannot be
            # walked back in later.
            # `shared_graph` caches that graph per process (CLAUDE.md gotcha
            # #14), so it has to be dropped or the next configuration walks the
            # previous one's edges.
            # `graph_search` binds the name at import, and `chains()` reads
            # *that* copy when it filters -- patching only `build_graph` leaves
            # the number fusion actually applies untouched.
            build_graph.REACHABLE_MAX_MIN = threshold
            graph_search_module.REACHABLE_MAX_MIN = threshold
            build_graph.shared_graph.cache_clear()

            shares = {city: _chain_share(city) for city in CITIES}
            recall = _chain_tier_recall()
            for weight in WEIGHTS:
                fusion_module.RETRIEVER_WEIGHTS[CHAIN_RETRIEVER] = weight
                kn2 = _kn2(FusionRetriever())
                worst = max(kn2.values())
                rows.append({
                    "reachable_max_min": threshold, "chain_weight": weight,
                    "kn2_worst": worst,
                    "kn2_band": ("drowned" if worst == 0 else
                                 "dominates" if worst > 4 else
                                 "thin" if worst < 2 else "in band"),
                    **{f"top8_{a}": n for a, n in sorted(kn2.items())},
                    **{f"share_{c}": (None if shares[c][0] is None else round(shares[c][0], 3))
                       for c in CITIES},
                    **{f"pois_{c}": shares[c][1] for c in CITIES},
                    # Independent of the weight -- the tier is measured on the
                    # retriever, not on the fused top-8 -- but carried on every
                    # row so a reader does not have to join two files.
                    **{f"chain_recall_{k}": v for k, v in recall.items()},
                    "chain_recall_ratio": (round(recall["fusion"] / recall["hybrid"], 2)
                                           if recall.get("hybrid") else None),
                })
                print(f"REACHABLE {threshold:>4} | weight {weight:>4} | "
                      f"worst {worst}/8 ({rows[-1]['kn2_band']})", flush=True)
    finally:
        build_graph.REACHABLE_MAX_MIN = shipped_threshold
        graph_search_module.REACHABLE_MAX_MIN = shipped_threshold
        fusion_module.RETRIEVER_WEIGHTS[CHAIN_RETRIEVER] = shipped_weight
        build_graph.shared_graph.cache_clear()

    df = pd.DataFrame(rows)
    df.to_csv(SWEEP_CSV, index=False)
    print(f"\nWrote {SWEEP_CSV.relative_to(_REPO_ROOT)}\n")
    print(df[["reachable_max_min", "chain_weight", "kn2_worst", "kn2_band",
              "share_PAR", "share_BER"]].to_string(index=False))
    print("\nChain tier, normalized recall (weight-independent):")
    print(df.drop_duplicates("reachable_max_min")[
        ["reachable_max_min", "chain_recall_fusion", "chain_recall_hybrid",
         "chain_recall_ratio"]].to_string(index=False))


if __name__ == "__main__":
    main()
