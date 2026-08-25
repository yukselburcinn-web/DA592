"""Build a Wikivoyage gold list per city, then measure a catalogue against it.

The gold list is the independent yardstick: Wikivoyage is written by travellers
listing what is worth seeing, so it owes nothing to Wikidata sitelinks or OSM
tagging -- the two signals the catalogue is built from. Coverage against it is
therefore a real recall measurement rather than a restatement of our own inputs.

District subpages are discovered from the API rather than hardcoded, so a new
city needs only a row in common.CITIES.

    python gold_list.py PAR BER --catalogue ../data/poi.csv
"""
import argparse
import re
import sys

import pandas as pd

from common import (CITIES, DATA, WD_API, WV_API, haversine_km, http, norm_name)

SIGHT_TYPES = {"see", "do", "buy"}
KNOWN_HEADS = {"see", "do", "buy", "eat", "drink", "sleep", "go", "listing",
               "marker", "vicinity"}
MATCH_M = 250          # a catalogue POI this close to a gold entry is that entry


# ---------------------------------------------------------------------------
# Wikitext parsing
# ---------------------------------------------------------------------------
def iter_templates(text):
    """Yield the inner body of every {{...}} template, brace-matched so
    multi-line listings and nested templates survive."""
    i = 0
    while True:
        start = text.find("{{", i)
        if start == -1:
            return
        depth, j = 0, start
        while j < len(text) - 1:
            if text[j:j + 2] == "{{":
                depth += 1
                j += 2
            elif text[j:j + 2] == "}}":
                depth -= 1
                j += 2
                if depth == 0:
                    break
            else:
                j += 1
        yield text[start + 2:j - 2]
        i = start + 2


def split_params(body):
    """Split on top-level pipes only -- nested templates and links contain
    pipes of their own."""
    parts, depth, cur, k = [], 0, [], 0
    while k < len(body):
        two = body[k:k + 2]
        if two in ("{{", "[["):
            depth += 1
            cur.append(two)
            k += 2
        elif two in ("}}", "]]"):
            depth -= 1
            cur.append(two)
            k += 2
        elif body[k] == "|" and depth == 0:
            parts.append("".join(cur))
            cur, k = [], k + 1
        else:
            cur.append(body[k])
            k += 1
    parts.append("".join(cur))
    return parts


def parse_listing(body):
    parts = split_params(body)
    head = parts[0].strip().lower()
    if head not in KNOWN_HEADS:
        return None
    fields = {}
    for p in parts[1:]:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        fields[k.strip().lower()] = v.strip()

    ltype = (fields.get("type", "") if head in ("listing", "marker")
             else fields.get("type", head)).strip().lower()
    name = re.sub(r"\[\[([^\]|]*\|)?([^\]]*)\]\]", r"\2", fields.get("name", "")).strip()
    name = re.sub(r"[''\"]{2,}", "", name).strip()
    if not name:
        return None
    qid = (fields.get("wikidata") or "").strip().upper()
    return {
        "name": name,
        "type": ltype,
        "lat": fields.get("lat") or None,
        "lon": fields.get("long") or fields.get("lon") or None,
        "qid": qid if re.fullmatch(r"Q\d+", qid) else None,
    }


# ---------------------------------------------------------------------------
# Page discovery
# ---------------------------------------------------------------------------
def discover_pages(root):
    """The city article plus every district subpage under it, plus the
    "<city> with children" companion article when it exists."""
    pages = [root]
    # `apfilterredir=nonredirects` matters: Wikivoyage carries a redirect per
    # historical district name ("Berlin/Kreuzberg", "Berlin/Central East" and
    # "Berlin/East Central" all point at one article), and parsing with
    # redirects=1 would fetch and count that article once per alias.
    d = http(WV_API, {"action": "query", "format": "json", "list": "allpages",
                      "apprefix": f"{root}/", "apnamespace": "0",
                      "apfilterredir": "nonredirects", "aplimit": "200"})
    pages += [p["title"] for p in d.get("query", {}).get("allpages", [])]
    extra = f"{root} with children"
    d2 = http(WV_API, {"action": "query", "format": "json", "titles": extra})
    if not any(k == "-1" for k in d2.get("query", {}).get("pages", {})):
        pages.append(extra)
    return pages


DEDUPE_M = 40
GENERIC_TOKENS = {"musee", "museum", "galerie", "gallery", "eglise", "church",
                  "paris", "berlin", "national", "nationale", "national",
                  "theatre", "theater", "palais", "palace", "maison", "haus",
                  "centre", "center", "berliner", "deutsche", "deutsches"}


def dedupe_gold(gold):
    """Collapse the same sight listed twice under different names.

    Wikivoyage carries both the local and the English name across its district
    pages -- "Carnavalet" and "Musee Carnavalet", "Picasso Museum" and "Musee
    Picasso" -- and a name-only dedupe keeps both, which inflates the
    denominator of every coverage number. Two entries merge when they share a
    Wikidata id, or when they sit within 40 m *and* one name contains the
    other; proximity alone would merge genuinely distinct neighbours in a dense
    centre.
    """
    keep, seen_qids = [], set()
    for g in gold.itertuples():
        if g.qid and g.qid in seen_qids:
            continue
        gn = norm_name(g.name)
        dup = False
        for k in keep:
            if haversine_km(g.clat, g.clon, k["clat"], k["clon"]) * 1000 > DEDUPE_M:
                continue
            kn = norm_name(k["name"])
            if not gn or not kn:
                continue
            if gn in kn or kn in gn:
                dup = True
                break
            # "Musee Picasso" / "Picasso Museum" contain neither the other, so
            # fall back to a shared distinctive token -- a proper noun, not the
            # generic words half these names are built from.
            shared = ({t for t in gn.split() if len(t) >= 5} &
                      {t for t in kn.split() if len(t) >= 5}) - GENERIC_TOKENS
            if shared:
                dup = True
                break
        if dup:
            continue
        if g.qid:
            seen_qids.add(g.qid)
        keep.append({"name": g.name, "type": g.type, "qid": g.qid,
                     "clat": g.clat, "clon": g.clon, "km": g.km})
    return pd.DataFrame(keep)


def build_gold(code, types=SIGHT_TYPES):
    """Wikivoyage listings for a city, deduped and inside its radius.

    `types` is which listing sections to keep. The catalogue measurement wants
    SIGHT_TYPES; `retrieval_gold.py` also wants `eat` and `drink`, because a
    retrieval answer key has to be able to say what a good restaurant or bar
    is (#48).
    """
    city = CITIES[code]
    print(f"\n{'=' * 68}\n{city['city']} gold list\n{'=' * 68}")
    pages = discover_pages(city["wv_root"])
    print(f"{len(pages)} Wikivoyage sayfasi bulundu")

    rows = []
    for page in pages:
        d = http(WV_API, {"action": "parse", "format": "json", "prop": "wikitext",
                          "page": page, "redirects": "1"})
        if "parse" not in d:
            print(f"  ! {page}: sayfa yok")
            continue
        n = 0
        for body in iter_templates(d["parse"]["wikitext"]["*"]):
            rec = parse_listing(body)
            if rec:
                rec["page"] = page
                rows.append(rec)
                n += 1
        print(f"  {page:<44} {n:>4} listing")

    df = pd.DataFrame(rows)
    print(f"\n{len(df)} ham listing; tipler: {df.type.value_counts().head(8).to_dict()}")

    # Many listings carry a QID but no inline coordinate; Wikidata has it.
    need = sorted({r.qid for r in df.itertuples() if r.qid and not r.lat})
    print(f"{len(need)} listing koordinatsiz ama QID'li -> Wikidata'dan cozuluyor")
    coords = {}
    for i in range(0, len(need), 50):
        d = http(WD_API, {"action": "wbgetentities", "format": "json",
                          "ids": "|".join(need[i:i + 50]), "props": "claims"})
        for qid, ent in d.get("entities", {}).items():
            cl = ent.get("claims", {}).get("P625")
            if cl:
                v = cl[0]["mainsnak"].get("datavalue", {}).get("value", {})
                if "latitude" in v:
                    coords[qid] = (v["latitude"], v["longitude"])
    print(f"  {len(coords)}/{len(need)} cozuldu")

    def coord_of(r):
        if r["lat"] and r["lon"]:
            try:
                return float(r["lat"]), float(r["lon"])
            except ValueError:
                pass
        return coords.get(r["qid"], (None, None))

    df[["clat", "clon"]] = df.apply(lambda r: pd.Series(coord_of(r)), axis=1)

    gold = df[df.type.isin(types) & df.clat.notna()].copy()
    gold["km"] = gold.apply(
        lambda r: haversine_km(city["lat"], city["lon"], r.clat, r.clon), axis=1)
    gold = gold[gold.km <= city["radius_km"]].drop_duplicates(subset="name")
    gold = dedupe_gold(gold).reset_index(drop=True)

    out = DATA / f"gold_{code}.csv"
    gold.to_csv(out, index=False)
    print(f"\nGOLD: {len(gold)} yer (merkeze <= {city['radius_km']} km) -> {out.name}")
    print(f"  tip: {gold.type.value_counts().to_dict()}")
    print(f"  mesafe 0-2/2-4/4-7 km: {(gold.km <= 2).sum()}/"
          f"{((gold.km > 2) & (gold.km <= 4)).sum()}/{(gold.km > 4).sum()}")
    return gold


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------
def assign(gold, cat, qid_of):
    """One-to-one match of gold entries to catalogue rows.

    Returns ({gold_index: tier_name}, {gold_index: catalogue_positional_index}).

    A 250 m radius in a dense centre puts several gold entries inside one
    catalogue POI's circle, and counting each of them as covered credits us for
    places we do not hold -- it is what produced a 106% "hit rate". Each
    catalogue POI may stand for exactly one gold entry, cheapest tier first.
    """
    proposals = []          # (tier, distance_m, gold_index, catalogue_index)
    for gi, g in enumerate(gold.itertuples()):
        gn = norm_name(g.name)
        if g.qid and g.qid in qid_of:
            proposals.append((0, 0.0, gi, qid_of[g.qid]))
        for ci, r in enumerate(cat.itertuples()):
            d = haversine_km(g.clat, g.clon, r.lat, r.lon) * 1000
            if d <= MATCH_M:
                proposals.append((1, d, gi, ci))
            elif len(gn) > 6 and (gn in r.n or r.n in gn):
                proposals.append((2, d, gi, ci))

    proposals.sort(key=lambda p: (p[0], p[1]))
    taken_gold, taken_cat, assigned, matched_rows = set(), set(), {}, {}
    for tier, d, gi, ci in proposals:
        if gi in taken_gold or ci in taken_cat:
            continue
        taken_gold.add(gi)
        taken_cat.add(ci)
        assigned[gi] = ("qid", "koordinat", "isim")[tier]
        matched_rows[gi] = ci
    return assigned, matched_rows


def catalogue_qid_index(cat):
    """{QID: first positional index carrying it} for a catalogue frame."""
    qid_of = {}
    for i, r in enumerate(cat.itertuples()):
        q = str(r.wikidata_qid).upper()
        if q and q != "NAN":
            qid_of.setdefault(q, i)
    return qid_of


def measure(code, gold, cat):
    """Three-tier match: QID, then <=250 m coordinate, then name.

    Tiers matter because neither side is canonical. The catalogue carries local
    names where Wikivoyage is English ("Pergamonmuseum" vs "Pergamon Museum"),
    so a name match alone is unreliable; a coordinate match alone confuses
    neighbours in a dense centre. QID is exact when both sides have one.
    """
    city = CITIES[code]
    cat = cat[cat.destination_id == code].copy()
    if cat.empty:
        print(f"  ! katalogda {code} yok")
        return None
    cat["n"] = cat.name.map(norm_name)
    cat_qids = set(cat.wikidata_qid.dropna().astype(str).str.upper()) - {""}

    qid_of = catalogue_qid_index(cat)

    assigned, matched_rows = assign(gold, cat, qid_of)

    hits, misses = [], []
    for gi, g in enumerate(gold.itertuples()):
        row = (g.name, assigned.get(gi), g.type, round(g.km, 2), g.qid or "")
        (hits if gi in assigned else misses).append(row)

    rec = 100 * len(hits) / len(gold)
    # A 150-slot catalogue cannot cover a 275-entry gold list, so the raw
    # percentage is not comparable across cities with differently sized gold
    # lists -- report what share of the *reachable* maximum we actually took.
    ceiling = 100 * min(len(cat), len(gold)) / len(gold)
    print(f"\n{'=' * 68}")
    print(f"{city['city']} KAPSAMA: {len(hits)}/{len(gold)} = {rec:.1f}%   "
          f"kacan: {len(misses)}")
    print(f"  tavan (katalog {len(cat)} slot / gold {len(gold)}): {ceiling:.1f}%"
          f"   -> tavanin {100 * rec / ceiling:.1f}%'i alindi")
    print(f"  isabet orani: katalogdaki {len(cat)} POI'nin {len(hits)}'i gold'da "
          f"= {100 * len(hits) / len(cat):.1f}%")
    print(f"{'=' * 68}")
    print("  eslesme:", pd.Series([h[1] for h in hits]).value_counts().to_dict())

    # `buy` is mostly individual boutiques -- a fame-ranked catalogue will never
    # hold them and arguably should not, so they drag the headline number down
    # without saying anything about the pipeline. The QID-bearing subset is the
    # part Wikidata can even represent, which is the fairest test of the spine.
    for label, mask in (("see+do (gezilecek)", gold.type.isin({"see", "do"})),
                        ("buy (dukkanlar)", gold.type == "buy"),
                        ("QID tasiyanlar", gold.qid.notna() & (gold.qid != ""))):
        idx = set(gold[mask].index)
        tot = len(idx)
        if tot:
            got = sum(1 for gi in assigned if gi in idx)
            print(f"  {label:<20} {got:>3}/{tot:<3} = {100 * got / tot:.0f}%")

    for lo, hi, label in ((0, 2, "0-2 km"), (2, 4, "2-4 km"), (4, 99, "4-7 km")):
        band = [h for h in hits if lo < h[3] <= hi or (lo == 0 and h[3] <= hi)]
        tot = len(gold[(gold.km > lo) & (gold.km <= hi)]) if lo else len(gold[gold.km <= hi])
        if tot:
            print(f"  {label}: {len(band)}/{tot} = {100 * len(band) / tot:.0f}%")

    md = pd.DataFrame(misses, columns=["name", "how", "type", "km", "qid"])
    md = md.drop(columns="how").sort_values("km")
    md.to_csv(DATA / f"missed_{code}.csv", index=False)
    print(f"\n  --- kacan {len(md)} yer (ilk 25) ---")
    for r in md.head(25).itertuples():
        print(f"    [{r.type:<3}] {r.name[:46]:<48} {r.km:>5} km  {r.qid}")
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cities", nargs="*", default=list(CITIES))
    ap.add_argument("--catalogue", default=str(DATA / "poi.csv"),
                    help="olculecek katalog CSV'si (poi.csv semasi)")
    args = ap.parse_args()
    codes = [c.upper() for c in (args.cities or CITIES)]
    for c in codes:
        if c not in CITIES:
            sys.exit(f"bilinmeyen sehir: {c}")

    cat = None
    try:
        cat = pd.read_csv(args.catalogue)
        print(f"katalog: {args.catalogue}  ({len(cat)} satir)")
    except FileNotFoundError:
        print(f"katalog yok ({args.catalogue}) -- sadece gold list uretilecek")

    for code in codes:
        gold = build_gold(code)
        if cat is not None:
            measure(code, gold, cat)


if __name__ == "__main__":
    main()
