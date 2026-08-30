"""Can one quota exponent serve three trip lengths at once? (issue #143)

#123 shipped `PREFERENCE_QUOTA_EXPONENT = 0.5` and left one acceptance
criterion unmet: at the one-day pool the reachable set fell from 173 POIs to
168. Its sweep found no configuration that cost nothing anywhere -- the only
lossless row was the baseline -- and concluded the pool is zero-sum at a fixed
`top_k`. #143 asks the next question: the sweep held the exponent *constant
across trip lengths*, and one number may simply not fit three pools.

The mechanism says it should not. `build_graph.rank_preferred` interleaves
categories by `(i + 1) / weight ** exponent`, so a low exponent flattens the
ranking towards equal shares. At five days there are 120 slots and flattening
is free -- the saturated categories keep what they had and the weak ones get
the remainder. At one day there are 24, and flattening cuts `nature` (47 of 76
reachable) and `nightlife` (35 of 41) to hand slots to categories that have
nothing to add yet.

So this sweeps the exponent *per trip length*, and extends the range above 1.0,
which #123 never tried: it was looking for a counterweight to the phrasings and
had no reason to sharpen past proportional. If the best exponent moves with
`top_k`, the constant should become a function of it.

What it reports, per (exponent, trip length):

  total        reachable POIs, unioned over 2 cities x 7 archetypes -- the
               measure `retrieval_coverage.py` uses
  per category the same, split out, because #123's fault was invisible in the
               total
  iconic       of each city's twelve best-known POIs, how many are reachable.
               A total can be gamed by trading famous places for obscure ones.

Run:  python -m roamwise.evaluation.quota_topk_sweep
Writes `quota_topk_sweep.csv`. Nothing here mutates the shipped constant; the
exponent is patched per configuration and restored.
"""
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd

from roamwise.agents.fusion_rag_agent import FusionRAGAgent
from roamwise.knowledge_graph import build_graph
from roamwise.knowledge_graph.build_graph import CATEGORY_AFFINITY
from roamwise.retrieval.query import archetype_query

HERE = Path(__file__).parent
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SWEEP_CSV = HERE / "quota_topk_sweep.csv"

CITIES = ["PAR", "BER"]
TRIP_LENGTHS = [1, 3, 5]
# `retrieval_coverage.py`'s pool sizes, kept identical so the numbers here can
# be read against the committed coverage report rather than only against
# themselves.
RETRIEVED_POIS_PER_DAY = 24
ICONIC_PER_CITY = 12

# Above 1.0 as well as below. #123 swept 1.0 down to 0.4 because it needed a
# counterweight; the one-day pool wants the opposite and nobody had looked.
EXPONENTS = [0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]


def reachable(rag: FusionRAGAgent, top_k: int) -> set:
    """Every POI some traveler could be shown at this pool size.

    Union over archetypes and cities: `archetype_query` reads only the
    archetype, so this is the whole space rather than a sample of it.
    """
    found = set()
    for city in CITIES:
        for archetype in sorted(CATEGORY_AFFINITY):
            out = rag.run(archetype_query(archetype), destination_id=city,
                          archetype=archetype, config="fusion", top_k=top_k,
                          narrate=False)
            found |= {r["poi_id"] for r in out["results"] if r.get("type") == "poi"}
    return found


def _iconic_ids(poi: pd.DataFrame) -> set:
    """Each city's best-known POIs, by the catalogue's own popularity score."""
    return {row.poi_id
            for city in CITIES
            for row in (poi[poi.destination_id == city]
                        .nlargest(ICONIC_PER_CITY, "popularity_score")
                        .itertuples())}


def main():
    rag = FusionRAGAgent()
    poi = pd.read_csv(DATA_DIR / "poi.csv")
    category_of = dict(zip(poi.poi_id, poi.category))
    iconic = _iconic_ids(poi)
    shipped = build_graph.PREFERENCE_QUOTA_EXPONENT

    rows = []
    try:
        for exponent in EXPONENTS:
            # The graph index caches per process and reads the constant at
            # ranking time, so patching the module is enough -- no rebuild.
            build_graph.PREFERENCE_QUOTA_EXPONENT = exponent
            for n_days in TRIP_LENGTHS:
                top_k = n_days * RETRIEVED_POIS_PER_DAY
                found = reachable(rag, top_k)
                by_category = pd.Series(
                    [category_of.get(p) for p in found]).value_counts().to_dict()
                rows.append({
                    "exponent": exponent, "n_days": n_days, "top_k": top_k,
                    "total": len(found),
                    "iconic": len(found & iconic), "iconic_of": len(iconic),
                    **{f"cat_{c}": n for c, n in sorted(by_category.items())},
                })
                print(f"exponent {exponent:>4} | {n_days}d (top_k={top_k:>3}) | "
                      f"total {len(found):>3} | iconic {len(found & iconic)}/{len(iconic)}")
    finally:
        build_graph.PREFERENCE_QUOTA_EXPONENT = shipped

    df = pd.DataFrame(rows).fillna(0)
    for column in df.columns:
        if column.startswith("cat_"):
            df[column] = df[column].astype(int)
    df.to_csv(SWEEP_CSV, index=False)

    print(f"\nWrote {SWEEP_CSV.relative_to(_REPO_ROOT)}\n")
    print("Reachable POIs by exponent and trip length:")
    print(df.pivot(index="exponent", columns="n_days", values="total").to_string())
    print("\nIconic coverage (of 24):")
    print(df.pivot(index="exponent", columns="n_days", values="iconic").to_string())


if __name__ == "__main__":
    main()
