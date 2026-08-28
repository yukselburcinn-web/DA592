"""How much of the catalogue can a traveler ever be shown?

Issue #113. Every measurement in this project so far asked what a *plan*
contains; this one asks what the catalogue can even offer, which is a different
question and turned out to have a much worse answer for one category.

Two numbers, and they check each other:

  reachability   Union over every (city x archetype) retrieval pool: the set of
                 POIs some traveler could see. Per category, because the fault
                 this script was written for was invisible in the total --
                 63.8% of the catalogue was reachable while `religion` sat at
                 1 POI of 84.

  iconic@k       Of each city's twelve best-known POIs, how many are in that
                 union. Reachability alone can be gamed by trading famous
                 places for obscure ones, and this is the measure that catches
                 it: before #113, four of the twenty-four -- Notre-Dame de
                 Paris, the Pantheon, Sacre-Coeur and Berlin Cathedral -- were
                 unreachable at any trip length, all of them `religion`.

`top_k` follows the app rather than a round number: `orchestrator` retrieves
`RETRIEVED_POIS_PER_DAY` per day, so a three-day trip is what the middle row
of every table here describes.

Run:  python evaluation/retrieval_coverage.py
"""
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd

from roamwise.agents.fusion_rag_agent import FusionRAGAgent
from roamwise.agents.orchestrator import RETRIEVED_POIS_PER_DAY
from roamwise.knowledge_graph.build_graph import CATEGORY_AFFINITY, DATA_DIR
from roamwise.retrieval.query import archetype_query

HERE = Path(__file__).parent
CATEGORY_CSV = HERE / "retrieval_coverage_by_category.csv"
ICONIC_CSV = HERE / "retrieval_coverage_iconic.csv"

CITIES = ["PAR", "BER"]
TRIP_LENGTHS = [1, 3, 5]
# How many of a city's best-known POIs count as the ones a traveler would be
# surprised not to be offered. Twelve because that is the list `query.py`'s
# docstring already uses to describe the failure it was written for.
ICONIC_PER_CITY = 12


def reachable_pois(rag: FusionRAGAgent, top_k: int) -> set:
    """Every POI some traveler could be shown at this trip length.

    The union is over archetypes rather than over slider settings on purpose:
    `archetype_query` reads only the archetype, so the retrieval pool depends
    on the city and the archetype and on nothing else the traveler can move.
    Seven archetypes times two cities is therefore the whole space, not a
    sample of it."""
    found = set()
    for city in CITIES:
        for archetype in sorted(CATEGORY_AFFINITY):
            out = rag.run(archetype_query(archetype), destination_id=city,
                          archetype=archetype, config="fusion", top_k=top_k,
                          narrate=False)
            found |= {r["poi_id"] for r in out["results"] if r.get("type") == "poi"}
    return found


def main():
    rag = FusionRAGAgent()
    poi = pd.read_csv(DATA_DIR / "poi.csv")
    iconic = pd.concat([poi[poi.destination_id == c].nlargest(ICONIC_PER_CITY,
                                                              "popularity_score")
                        for c in CITIES])
    by_category, by_iconic = [], []
    for n_days in TRIP_LENGTHS:
        top_k = n_days * RETRIEVED_POIS_PER_DAY
        found = reachable_pois(rag, top_k)
        held = poi.assign(reachable=poi.poi_id.isin(found))
        for category, group in held.groupby("category"):
            by_category.append({
                "n_days": n_days, "top_k": top_k, "category": category,
                "in_catalogue": len(group),
                "reachable": int(group.reachable.sum()),
                "reachable_pct": round(100 * group.reachable.mean(), 1)})
        hit = iconic.poi_id.isin(found)
        by_iconic.append({
            "n_days": n_days, "top_k": top_k, "iconic": len(iconic),
            "reachable": int(hit.sum()),
            "reachable_pct": round(100 * hit.mean(), 1),
            "missing": "; ".join(iconic[~hit]["name"]) or "-"})
        print(f"  {n_days}-day trip (top_k={top_k}): "
              f"{len(found)}/{len(poi)} POIs, iconic {hit.sum()}/{len(iconic)}",
              flush=True)

    cats = pd.DataFrame(by_category)
    cats.to_csv(CATEGORY_CSV, index=False)
    icons = pd.DataFrame(by_iconic)
    icons.to_csv(ICONIC_CSV, index=False)
    print(f"\nwrote {CATEGORY_CSV}\nwrote {ICONIC_CSV}\n")
    print(cats.pivot(index="category", columns="n_days",
                     values="reachable_pct").to_string())
    print()
    print(icons[["n_days", "top_k", "reachable", "iconic", "reachable_pct",
                 "missing"]].to_string(index=False))


if __name__ == "__main__":
    main()
