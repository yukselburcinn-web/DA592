"""Where to set the chain's two thresholds, measured rather than assumed
(issue #126, phase 5).

The chain is two edges and each carries a limit:

    anchor -[SERVES <= s min]-> POI_a -[REACHABLE <= r min]-> POI_b

`s` says how far into the city the traveler's starting point reaches; `r` says
how far one place is from the next. Phase 1 and 2 shipped 15 and 10 on a
measurement of edge counts alone -- how many edges each threshold produces and
whether the relation still separates anything. That was enough to build on and
is not enough to ship on, because edge counts say nothing about whether the
places the chain returns are worth returning.

What this scores, per (s, r) cell:

  share       what fraction of the city's catalogue the chain reaches. The
              failure mode #126 measured and rejected is raw two-hop expansion,
              which returns 41-54% -- half the catalogue, dressed as a query.
              A threshold pair that climbs into that band has stopped asking a
              question.
  invalid     of the pairs the two edges admit, the share that are not doable
              in sequence -- the second place already shut. This is what the
              hour constraint is worth, and it needs no answer key, so no
              choice of gold can inflate it.
  precision   of the POIs the chain reaches, the share Wikivoyage recommends.
              This is the dilution test: loosening a threshold always reaches
              more places, and the question is whether the extra ones are worth
              seeing. Wikivoyage owes nothing to this project's graph (#48), so
              it cannot be satisfied by traversing harder.
  recall      of the city's Wikivoyage-recommended POIs, the share the chain
              reaches. Precision alone is maximised by returning almost
              nothing, so the two are read together.

The grid is swept by *filtering* a graph built at the widest cell rather than
rebuilding one per cell: `chains_from` takes both limits, and tightening them
selects among edges already stored. That makes the whole sweep one graph build.

Run:  python -m roamwise.evaluation.chain_threshold_sweep
"""
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import datetime
import itertools

import pandas as pd

from roamwise.knowledge_graph import build_graph as build_graph_module
from roamwise.knowledge_graph.build_graph import DATA_DIR, GraphIndex, build_graph
from roamwise.retrieval.graph_search import GraphSearchIndex

HERE = Path(__file__).parent
SWEEP_CSV = HERE / "chain_threshold_sweep.csv"
GOLD_CSV = HERE / "retrieval_gold.csv"

# The same Thursday `comparative_analysis.CHAIN_DATE` grades on: a chain is a
# claim about opening hours, so the day has to be fixed or the sweep is not
# reproducible.
CHAIN_DATE = datetime.date(2026, 9, 24)

# `SERVES` is the first leg -- getting into the city -- and is allowed to be
# generous; `REACHABLE` is the leg that has to discriminate. The grid stops at
# 20 minutes for the second because #126 already measured 20 as #113's 3 km
# radius in a new unit (40% of all pairs), and there is no reason to sweep a
# region a previous issue has already ruled out.
SERVES_GRID = [10.0, 15.0, 20.0, 25.0]
REACHABLE_GRID = [5.0, 10.0, 15.0, 20.0]

# What "raw two-hop expansion" looked like when #126 measured and rejected it:
# 41-54% of the catalogue from a single hub. A cell at or above this is not a
# tighter version of the same query, it is that failure.
RAW_EXPANSION_SHARE = 0.41


def main():
    cities = sorted(pd.read_csv(DATA_DIR / "destinations.csv").destination_id)
    recommended = set(pd.read_csv(GOLD_CSV).poi_id)

    # One graph, built at the widest cell in the grid. Everything below
    # tightens; nothing can loosen past what is stored here.
    shipped = (build_graph_module.SERVES_MAX_MIN, build_graph_module.REACHABLE_MAX_MIN)
    build_graph_module.SERVES_MAX_MIN = max(SERVES_GRID)
    build_graph_module.REACHABLE_MAX_MIN = max(REACHABLE_GRID)
    try:
        graph = build_graph()
    finally:
        (build_graph_module.SERVES_MAX_MIN,
         build_graph_module.REACHABLE_MAX_MIN) = shipped
    index = GraphSearchIndex(GraphIndex(graph))
    print(f"widest grid cell: SERVES <= {max(SERVES_GRID)} min, "
          f"REACHABLE <= {max(REACHABLE_GRID)} min -> {graph.number_of_edges()} edges",
          flush=True)

    rows = []
    for city in cities:
        catalogue = {p["poi_id"] for p in index.idx.city_pois(city)}
        city_recommended = catalogue & recommended
        for serves, reach in itertools.product(SERVES_GRID, REACHABLE_GRID):
            valid = index.chains(city, start_date=CHAIN_DATE,
                                 max_serves_min=serves, max_reachable_min=reach)
            admitted = index.chains(city, start_date=CHAIN_DATE, enforce_hours=False,
                                    max_serves_min=serves, max_reachable_min=reach)
            reached = {poi["poi_id"] for chain in valid for poi in (chain[0], chain[2])}
            hit = reached & city_recommended
            rows.append({
                "destination_id": city, "serves_max_min": serves,
                "reachable_max_min": reach,
                "chains": len(valid), "admitted_chains": len(admitted),
                "invalid_chain_rate": round(1 - len(valid) / len(admitted), 3) if admitted else None,
                "pois_reached": len(reached),
                "catalogue": len(catalogue),
                "share_of_catalogue": round(len(reached) / len(catalogue), 3),
                "recommended_precision": round(len(hit) / len(reached), 3) if reached else None,
                "recommended_recall": round(len(hit) / len(city_recommended), 3) if city_recommended else None,
                "raw_expansion_band": len(reached) / len(catalogue) >= RAW_EXPANSION_SHARE,
            })
            print(f"  {city}  serves<={serves:<5} reach<={reach:<5} "
                  f"chains {len(valid):5d}  share {rows[-1]['share_of_catalogue']:.0%}"
                  f"{'  <-- raw-expansion band' if rows[-1]['raw_expansion_band'] else ''}"
                  f"  precision {rows[-1]['recommended_precision']}"
                  f"  recall {rows[-1]['recommended_recall']}"
                  f"  invalid {rows[-1]['invalid_chain_rate']}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(SWEEP_CSV, index=False)
    print(f"\nwrote {SWEEP_CSV}")

    # Averaged over cities, because a threshold is one number for the whole
    # system and picking it from whichever city looks better is how a knob ends
    # up fitted to one dataset.
    summary = (df.groupby(["serves_max_min", "reachable_max_min"])
               .agg(share=("share_of_catalogue", "mean"),
                    precision=("recommended_precision", "mean"),
                    recall=("recommended_recall", "mean"),
                    invalid=("invalid_chain_rate", "mean"),
                    in_raw_band=("raw_expansion_band", "any"))
               .round(3).reset_index())
    print("\naveraged over cities:")
    print(summary.to_string(index=False))
    print(f"\nshipped: SERVES <= {shipped[0]}, REACHABLE <= {shipped[1]}")


if __name__ == "__main__":
    main()
