"""Drop catalogue rows that are not places a traveller can visit.

The category a POI gets is read off its Wikidata type label, and several of
those keywords admit whole classes of entity that are documented like a
landmark but cannot be a stop on a day out -- universities, a hospital, metro
stations, demolished buildings, a television channel, a fire. See
`common.not_a_sight` for the rule and why each class is there.

This is deliberately a pass over the *committed* catalogue rather than a
rebuild. `build_catalogue.py` ranks candidates on a 60-day rolling pageview
window, so re-running it picks a different POI set with different
`popularity_score` values and cannot reproduce the shipped catalogue -- which
would invalidate REPORT's coverage numbers and every committed evaluation CSV
for reasons unrelated to this fix. The same rule is applied inside
`build_catalogue.classify_spine` so a future rebuild is correct at source; this
script brings the catalogue already on disk into line with it.

Rows are classified from Wikidata P31 (instance of) and P1435 (heritage
designation), fetched through the shared disk cache. A row with no QID cannot
be classified and is kept -- the rule only ever removes on positive evidence.

    python sight_filter.py                # report only
    python sight_filter.py --apply        # rewrite poi.csv + dropped_pois.csv
"""
import argparse
import sys

import pandas as pd

from common import DATA, WD_API, http, not_a_sight

BATCH = 50          # wbgetentities takes 50 ids per request


def fetch_types(qids):
    """{qid: (P31 type labels, carries P1435)} for every qid given."""
    claims = {}
    for i in range(0, len(qids), BATCH):
        chunk = qids[i:i + BATCH]
        data = http(WD_API, {"action": "wbgetentities", "format": "json",
                             "ids": "|".join(chunk), "props": "claims"})
        for qid, entity in data.get("entities", {}).items():
            c = entity.get("claims", {})
            claims[qid] = (
                [snak["mainsnak"]["datavalue"]["value"]["id"]
                 for snak in c.get("P31", [])
                 if snak["mainsnak"].get("datavalue")],
                bool(c.get("P1435")),
            )
        print(f"  P31/P1435: {min(i + BATCH, len(qids))}/{len(qids)}", end="\r")
    print()

    # The rule matches on type *labels*, the same way CATEGORY_RULES does, so
    # the type QIDs have to be resolved to English labels.
    type_ids = sorted({t for types, _ in claims.values() for t in types})
    labels = {}
    for i in range(0, len(type_ids), BATCH):
        chunk = type_ids[i:i + BATCH]
        data = http(WD_API, {"action": "wbgetentities", "format": "json",
                             "ids": "|".join(chunk), "props": "labels"})
        for qid, entity in data.get("entities", {}).items():
            labels[qid] = entity.get("labels", {}).get("en", {}).get("value", qid)
        print(f"  tip etiketleri: {min(i + BATCH, len(type_ids))}/{len(type_ids)}", end="\r")
    print()

    return {qid: ([labels.get(t, t) for t in types], heritage)
            for qid, (types, heritage) in claims.items()}


def classify(catalogue):
    """Add `drop_reason` and `wikidata_types` columns to a catalogue frame."""
    qids = sorted({q for q in catalogue.wikidata_qid.dropna().astype(str) if q.startswith("Q")})
    print(f"{len(catalogue)} POI, {len(qids)} tanesinde QID var")
    types = fetch_types(qids)

    def reason(row):
        qid = str(row.wikidata_qid)
        if qid not in types:      # no QID, or Wikidata has no claims for it
            return None
        labels, heritage = types[qid]
        return not_a_sight(labels, heritage)

    out = catalogue.copy()
    out["wikidata_types"] = out.wikidata_qid.map(
        lambda q: "; ".join(types.get(str(q), ([], False))[0]))
    out["drop_reason"] = out.apply(reason, axis=1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="poi.csv'yi yeniden yaz ve dropped_pois.csv uret")
    ap.add_argument("--catalogue", default=str(DATA / "poi.csv"))
    args = ap.parse_args()

    catalogue = pd.read_csv(args.catalogue)
    classified = classify(catalogue)
    dropped = classified[classified.drop_reason.notna()]
    kept = classified[classified.drop_reason.isna()]

    print(f"\n{'=' * 66}")
    print(f"ELENEN: {len(dropped)}/{len(catalogue)} = {len(dropped) / len(catalogue):.1%}")
    print(f"{'=' * 66}")
    for reason, group in dropped.groupby("drop_reason"):
        print(f"\n  {reason}  ({len(group)})")
        print(f"    kategori: {dict(group.category.value_counts())}")
        for r in group.nlargest(6, "popularity_score").itertuples():
            print(f"      {r.destination_id}  {r['name'
                  ] if isinstance(r, dict) else r.name:<44.44}  {r.popularity_score:.2f}")

    print(f"\nkalan: {len(kept)}  ({dict(kept.destination_id.value_counts())})")
    print("kategori dagilimi:")
    before = catalogue.category.value_counts()
    after = kept.category.value_counts()
    for cat in sorted(before.index):
        print(f"   {cat:<10} {before.get(cat, 0):>4} -> {after.get(cat, 0):>4}")

    if not args.apply:
        print("\n(rapor modu -- yazmak icin --apply)")
        return

    columns = list(catalogue.columns)
    kept[columns].to_csv(DATA / "poi.csv", index=False)
    dropped[columns + ["drop_reason", "wikidata_types"]].to_csv(
        DATA / "dropped_pois.csv", index=False)
    print(f"\nyazildi: poi.csv ({len(kept)} satir), dropped_pois.csv ({len(dropped)} satir)")


if __name__ == "__main__":
    main()
