"""Tune the selection parameters against the Wikivoyage gold list.

Everything the sweep needs is already on disk from a catalogue run, so a
combination costs only the greedy selection plus the matching -- a few hundred
milliseconds. That makes it cheap to answer the question the Paris measurement
raised: the pipeline lost centre coverage to gain outer-ring coverage, and it
is not obvious which weighting of the two is best until both are scored.

Primary score is `see+do` recall. `buy` is left out of the objective on
purpose: those gold entries are individual boutiques that a fame-ranked
catalogue cannot hold, so optimising for them would just push the catalogue
toward noise.

    python sweep.py PAR BER
"""
import argparse
import itertools
import math
import sys
from collections import defaultdict

import pandas as pd

import build_catalogue as bc
from common import CITIES, DATA, haversine_km, norm_name

MATCH_M = 250


def load_city(code):
    """Spine + long tail + candidate pageviews, straight from the HTTP cache."""
    city = CITIES[code]
    osm_by_qid = bc.fetch_osm_wikidata_tags(city)
    by_item = bc.fetch_spine(city, osm_by_qid)
    sights, quarters, _, _ = bc.classify_spine(by_item, city)
    cands = sights + quarters
    pv = bc.fetch_pageviews_batch([r.get("wikipedia_title") for r in cands])
    for r in cands:
        r["rank_pageviews"] = pv.get(r.get("wikipedia_title") or "", 0)
    tail = bc.fetch_longtail(city)
    gold = pd.read_csv(DATA / f"gold_{code}.csv")
    return city, cands, tail, gold


def select(candidates, tail, city, target, longtail_slots, beta, damp_k, damp_p,
           ceiling_frac, quarter_slots, w_s=0.8, w_p=0.6):
    ceiling = int(ceiling_frac * target)
    slots = target - longtail_slots
    fame = lambda r: bc.blended_fame(r, w_s, w_p)
    remaining = sorted(candidates, key=lambda r: -fame(r))
    picked, per_cat, n_q = [], defaultdict(int), 0
    while len(picked) < slots and remaining:
        best, best_score = None, -1.0
        for rec in remaining:
            c = per_cat[rec["category"]]
            if c >= ceiling or (rec["is_quarter"] and n_q >= quarter_slots):
                continue
            radial = 1 + beta * (rec["km"] / city["radius_km"])
            score = fame(rec) * radial / (1 + damp_k * c) ** damp_p
            if score > best_score:
                best, best_score = rec, score
        if best is None:
            break
        picked.append(best)
        per_cat[best["category"]] += 1
        n_q += best["is_quarter"]
        remaining.remove(best)
    return picked + bc.select_longtail(tail, longtail_slots)


def score(chosen, gold):
    """One-to-one recall, same rule the reported measurement uses."""
    cat = [{"lat": r["lat"], "lon": r["lon"], "qid": r["qid"] or "",
            "n": norm_name(r["name"])} for r in chosen]
    qid_of = {}
    for i, r in enumerate(cat):
        if r["qid"]:
            qid_of.setdefault(r["qid"].upper(), i)

    proposals = []
    for gi, g in enumerate(gold.itertuples()):
        gn = norm_name(g.name)
        q = str(g.qid).upper()
        if q and q != "NAN" and q in qid_of:
            proposals.append((0, 0.0, gi, qid_of[q]))
        for ci, r in enumerate(cat):
            d = haversine_km(g.clat, g.clon, r["lat"], r["lon"]) * 1000
            if d <= MATCH_M:
                proposals.append((1, d, gi, ci))
            elif len(gn) > 6 and (gn in r["n"] or r["n"] in gn):
                proposals.append((2, d, gi, ci))

    proposals.sort(key=lambda p: (p[0], p[1]))
    tg, tc, hit = set(), set(), set()
    for _, _, gi, ci in proposals:
        if gi in tg or ci in tc:
            continue
        tg.add(gi)
        tc.add(ci)
        hit.add(gi)

    seedo = set(gold[gold.type.isin({"see", "do"})].index)
    near = set(gold[gold.km <= 2].index)
    far = set(gold[gold.km > 4].index)
    return {
        "toplam": 100 * len(hit) / len(gold),
        "see_do": 100 * len(hit & seedo) / max(1, len(seedo)),
        "yakin": 100 * len(hit & near) / max(1, len(near)),
        "uzak": 100 * len(hit & far) / max(1, len(far)),
    }


# The structural parameters were settled by the first sweep (see
# data/sweep_results.csv): beta 0 beats every radial boost, the long tail wants
# ~7% of slots, and the category ceiling never binds. What is left open is how
# to weight the two fame signals against each other.
GRID = {
    "w_s": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "w_p": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "beta": [0.0],
    "damp_k": [0.25],
    "damp_p": [0.7],
    "ceiling_frac": [0.22],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cities", nargs="*", default=["PAR", "BER"])
    ap.add_argument("--target", type=int, default=0,
                    help="0 = her sehrin kendi hedefi (CITIES)")
    args = ap.parse_args()
    codes = [c.upper() for c in args.cities]

    loaded = {}
    for code in codes:
        if code not in CITIES:
            sys.exit(f"bilinmeyen sehir: {code}")
        print(f"{code} yukleniyor (onbellekten)...")
        loaded[code] = load_city(code)

    keys = list(GRID)
    combos = list(itertools.product(*(GRID[k] for k in keys)))
    tinfo = args.target or "sehir bazli"
    print(f"\n{len(combos)} kombinasyon x {len(codes)} sehir, hedef {tinfo}\n")

    rows = []
    for combo in combos:
        p = dict(zip(keys, combo))
        per_city = {}
        for code in codes:
            city, cands, tail, gold = loaded[code]
            target = args.target or city.get("target", bc.TARGET)
            lt = max(8, round(bc.LONGTAIL_FRAC * target))
            chosen = select(cands, tail, city, target, lt,
                            p["beta"], p["damp_k"], p["damp_p"],
                            p["ceiling_frac"], bc.QUARTER_SLOTS,
                            p["w_s"], p["w_p"])
            per_city[code] = score(chosen, gold)
        row = dict(p)
        for m in ("see_do", "toplam", "yakin", "uzak"):
            row[m] = sum(per_city[c][m] for c in codes) / len(codes)
        for code in codes:
            row[f"{code}_see_do"] = per_city[code]["see_do"]
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("see_do", ascending=False)
    out = DATA / "sweep_results.csv"
    df.to_csv(out, index=False)

    show = ["w_s", "w_p", "see_do", "toplam", "yakin", "uzak"] + \
        [f"{c}_see_do" for c in codes]
    pd.set_option("display.width", 200)
    print("--- en iyi 12 (see+do ortalamasina gore) ---")
    print(df[show].head(12).to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    print("\n--- en kotu 3 ---")
    print(df[show].tail(3).to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
