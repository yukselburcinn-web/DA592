"""Build the retrieval evaluation's answer key from Wikivoyage.

`comparative_analysis.gold_for` used to define a correct answer as "every
catalogue POI of the queried category, near a hub if the query asked for it".
Both halves of that come from `GraphIndex.city_pois` and
`GraphIndex.multi_hop_transport_to_poi` -- the very traversals the graph
retriever dispatches to. So on any query naming its category next to a
transport word, graph retrieval was a strict subset of the answer key by
construction, and its recall measured that it agrees with itself rather than
that it finds good answers. `dependence_level` reported how much of the query
set was affected but could not fix it (issue #48).

Wikivoyage is written by travellers listing what is worth seeing, eating and
drinking in a city. It owes nothing to Wikidata sitelinks, to OpenStreetMap
tagging, or to this project's graph -- the three signals the catalogue and the
retriever are built from -- so membership of it is an independent statement
that a place is worth recommending. That is what an answer key needs to be.

The key is committed to `evaluation/retrieval_gold.csv` so the evaluation and
the test suite run offline; only rebuilding it needs the network.

    python retrieval_gold.py                 # report
    python retrieval_gold.py --write         # -> ../evaluation/retrieval_gold.csv
"""
import argparse

import pandas as pd

from common import CITIES, DATA, norm_name
from gold_list import assign, build_gold, catalogue_qid_index

ROOT = DATA.parent
OUT = ROOT / "evaluation" / "retrieval_gold.csv"

# Every Wikivoyage listing section that names somewhere a traveller goes.
# `gold_list.py`'s catalogue measurement keeps only see/do/buy because it is
# scoring sight coverage; a retrieval answer key also has to be able to say
# what a good restaurant or bar is, or `food` and `nightlife` queries have no
# correct answer at all.
LISTING_TYPES = {"see", "do", "buy", "eat", "drink"}


def build(codes):
    catalogue = pd.read_csv(DATA / "poi.csv")
    rows = []
    for code in codes:
        gold = build_gold(code, types=LISTING_TYPES)
        cat = catalogue[catalogue.destination_id == code].copy()
        cat["n"] = cat.name.map(norm_name)
        tiers, matched = assign(gold, cat, catalogue_qid_index(cat))
        for gold_index, catalogue_index in matched.items():
            poi = cat.iloc[catalogue_index]
            entry = gold.iloc[gold_index]
            rows.append({
                "poi_id": poi.poi_id,
                "destination_id": code,
                "poi_name": poi["name"],
                "category": poi.category,
                "wikivoyage_name": entry["name"],
                "wikivoyage_type": entry["type"],
                "matched_by": tiers[gold_index],
            })
        print(f"  {code}: {len(gold)} Wikivoyage listing -> {len(matched)} katalog POI eslesti "
              f"({len(matched) / len(cat):.0%} of {len(cat)})")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cities", nargs="*", default=list(CITIES))
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    key = build([c.upper() for c in (args.cities or CITIES)])
    print(f"\n{'=' * 66}\nRETRIEVAL GOLD: {len(key)} POI")
    print(f"{'=' * 66}")
    print("\nsehir x kategori:")
    print(pd.crosstab(key.category, key.destination_id).to_string())
    print("\nWikivoyage bolumu:", dict(key.wikivoyage_type.value_counts()))
    print("eslesme tipi:", dict(key.matched_by.value_counts()))

    if args.write:
        key.sort_values(["destination_id", "poi_id"]).to_csv(OUT, index=False)
        print(f"\nyazildi: {OUT.relative_to(ROOT.parent)}")
    else:
        print("\n(rapor modu -- yazmak icin --write)")


if __name__ == "__main__":
    main()
