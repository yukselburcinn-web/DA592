"""Build a fame-first POI catalogue for one or more cities, in poi.csv schema.

Wikidata is the spine -- it knows what is famous and carries coordinates --
and OSM supplies both the everyday long tail Wikidata does not cover
(restaurants, bars) and the operational tags Wikidata does not carry (opening
hours, fees). Selection uses a damped fame score instead of a round-robin
quota, so a famous museum is never traded for an anonymous bar.

What this adds over the single-city Istanbul pilot:

  1. OSM reverse join. The pilot took Wikidata rows straight to output, which
     left every spine POI on a category-default opening hour. We now pull every
     OSM object carrying a `wikidata` tag in the bbox and join on QID, which
     recovers real hours for half the Paris catalogue and 39% of Berlin's.
  2. A famous-quarter gate. Wikivoyage treats Montmartre and Kreuzberg as
     places you go; the pilot's concept blacklist dropped every neighbourhood
     outright. They are now admitted through a separate high-fame threshold.
  3. Per-city targets and swept parameters, scored against a Wikivoyage gold
     list by sweep.py rather than chosen by hand. Two of those measurements
     overturned an assumption: the radial distance boost the pilot proposed
     makes recall worse (see DIST_BETA), and so does ranking on pageviews
     (see FAME_W_PAGEVIEWS).

Everything network-bound is batched and disk-cached, so a re-run after a
tuning change costs nothing.

Writes <CODE>_poi.csv per city plus a combined poi.csv into ../data/.
Reads nothing from the repo.

    python build_catalogue.py PAR BER
"""
import argparse
import collections
import math
import re
import sys
import urllib.parse
from collections import defaultdict

import pandas as pd

from common import (CATEGORY_RULES, not_a_sight, CITIES, DATA, NETWORK_ERRORS, OVERPASS, QLEVER, WD_API,
                    WP_API, WP_REST, bbox_for, haversine_km, http, normalize_qid)

# ---------------------------------------------------------------------------
# Tunables. Phase 2 measures against a Wikivoyage gold list and adjusts these.
# ---------------------------------------------------------------------------
# Set by sweep.py against the Wikivoyage gold lists for both cities; see
# data/sweep_results.csv for the full 144-combination grid.
TARGET = 150                 # fallback when a city carries no explicit target
LONGTAIL_FRAC = 10 / 150     # share of slots for food + nightlife (OSM long tail)
SITELINK_FLOOR = 3           # below this a Wikidata item is noise, not a sight
CEILING_FRAC = 0.22          # no category may exceed this share of a catalogue
DAMP_K, DAMP_P = 0.25, 0.7   # category damping: /(1 + K*taken)^P

# Selection weights, swept against both gold lists. Documentation (sitelinks)
# carries most of the signal and a light touch of attention (pageviews) adds a
# little on top: see+do recall is 76.2% on sitelinks alone, 75.4% on pageviews
# alone, 77.0% blended -- against 59.1% with no ranking at all.
#
# A caution for anyone re-running this. An earlier sweep concluded pageviews
# *hurt* selection, monotonically, down to 61.3% for pageviews alone. That was
# an artefact: `prop=pageviews` paginates, the continuation token was not being
# followed, and ~80% of candidates silently carried a zero. A signal that is
# mostly zero will of course wreck a ranking. Complete the data before trusting
# any weight conclusion.
#
# The failure modes a *pure* pageview ranking shows are real, and are why the
# weight is small rather than large: news-driven spikes (the 2024 Olympics
# cauldron), places people look up but do not visit (a university, a political
# party's headquarters), rooms inside something already in the catalogue (the
# Galerie d'Apollon, in the Louvre), and monuments that no longer exist (the
# Elephant of the Bastille, demolished 1846). Blended at 0.2 the sitelink term
# filters those out.
FAME_W_SITELINKS, FAME_W_PAGEVIEWS = 1.0, 0.2

# Radial boost, off. It was added to fix the 4-7 km band that stayed at 50%
# coverage on Istanbul, and it does work -- at 0.55 the outer ring goes from
# 32% to 42%. But it is a bad trade in these two cities: most gold sights are
# central (Paris has 98 of 250 inside 2 km), so buying 9 points outside costs
# 16 points in the centre, and see+do recall falls from 50.0% to 43.6%. Kept as
# a parameter rather than deleted because the balance is city-shaped -- a
# sprawling city could well want it back.
DIST_BETA = 0.0

QUARTER_SITELINK_FLOOR = 22  # a neighbourhood must be this documented to count
QUARTER_SLOTS = 8            # and no more than this many per city

# Types that name a concept or an administrative unit, not a place you visit.
# `neighborhood`/`quarter` are here so the ordinary ones stay out; the famous
# ones come back through the quarter gate below.
TYPE_BLACKLIST = ["ancient city", "polis", "historical country", "battle", "council",
                  "human settlement", "city", "province", "district", "neighborhood",
                  "neighbourhood", "quarter", "borough", "municipality", "empire",
                  "administrative territorial entity", "commune of france",
                  "urban area", "capital", "metropolis", "locality",
                  # Museum pieces, not destinations. An outdoor statue almost
                  # always also carries `monument` or `memorial`, and the
                  # clean-type path below keeps it on that.
                  "painting", "sculpture", "work of art", "artwork", "artifact"]

# Wikidata types admitted as a "famous quarter" -- Montmartre, Kreuzberg.
# `arrondissement` is deliberately absent: Paris's numbered arrondissements are
# administrative units that no one visits as such, and they are documented well
# enough to crowd out the named quarters that people actually go to.
QUARTER_TYPES = ["neighborhood", "neighbourhood", "quarter", "district of berlin",
                 "quarter of paris", "ortsteil", "locality of berlin"]

# category: (avg_visit_minutes, price_level, open_hour, close_hour)
CATEGORY_DEFAULTS = {
    "museum": (120, 1, 9, 18),
    "landmark": (60, 1, 8, 20),
    "religion": (30, 0, 7, 19),
    "history": (60, 0, 8, 19),
    "culture": (90, 1, 9, 20),
    "shopping": (60, 1, 10, 20),
    "food": (60, 1, 10, 23),
    "nightlife": (120, 1, 18, 2),
    "nature": (90, 0, 0, 24),
    "beach": (150, 0, 0, 24),
}

MAX_DESCRIPTION_CHARS = 500


# ---------------------------------------------------------------------------
# 1. Wikidata spine
#
# QLever is the primary endpoint rather than WDQS. WDQS is the canonical
# service but it is a heavily shared one: during this build it was in a
# declared outage, rate-limiting to 1 req/min and answering 429 before it even
# looked at the query. QLever serves the same Wikidata dump, answers the same
# query in ~3 s, and is independent of that outage.
#
# The OSM route below stays as a fallback that shares nothing with either --
# it reaches Wikidata through the Action API instead of SPARQL -- so the spine
# survives any single service being down.
#
# Two QLever quirks the query has to work around:
#   * `wikibase:sitelinks` is an untyped literal there, where WDQS types it as
#     xsd:integer. `FILTER(?sitelinks >= 3)` silently matches nothing; the
#     comparison has to cast.
#   * There is no `wikibase:around` service, so the radius is a `geof:distance`
#     filter. It returns kilometres.
# ---------------------------------------------------------------------------
QLEVER_PREFIXES = """PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX schema: <http://schema.org/>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX geo: <http://www.opengis.net/ont/geosparql#>
PREFIX geof: <http://www.opengis.net/def/function/geosparql/>
"""


def _spine_query(city):
    """No `?type wdt:P279* ?base` whitelist.

    That transitive closure was admitting only items under one of eight place
    classes, and it is what made the WDQS version time out -- an unbounded
    closure evaluated per item over a 7 km radius. It is also redundant: the
    category is read off the *type label* downstream (CATEGORY_RULES), and that
    rule set is itself the whitelist. Filtering locally is cheaper and drops
    the arbitrariness of picking eight classes.
    """
    langs = ", ".join(f'"{l}"' for l in city["langs"].split(","))
    return QLEVER_PREFIXES + f"""
SELECT ?item ?label ?llang ?sitelinks ?typeLabel ?article ?desc ?coord ?dissolved ?closed WHERE {{
  ?item wdt:P625 ?coord .
  FILTER(geof:distance(?coord, "POINT({city['lon']} {city['lat']})"^^geo:wktLiteral)
         < {city['radius_km']})
  ?item wikibase:sitelinks ?sitelinks .
  FILTER(xsd:integer(?sitelinks) >= {SITELINK_FLOOR})
  ?item rdfs:label ?label . FILTER(lang(?label) IN ({langs}))
  BIND(lang(?label) AS ?llang)
  ?item wdt:P31 ?type . ?type rdfs:label ?typeLabel . FILTER(lang(?typeLabel) = "en")
  OPTIONAL {{ ?article schema:about ?item ; schema:isPartOf <https://en.wikipedia.org/> }}
  OPTIONAL {{ ?item schema:description ?desc . FILTER(lang(?desc) = "en") }}
  # P576 dissolved/abolished/demolished, P3999 date of official closure. Only
  # acted on for the venue categories -- see SERVICE_CATEGORIES below.
  OPTIONAL {{ ?item wdt:P576 ?dissolved }}
  OPTIONAL {{ ?item wdt:P3999 ?closed }}
  # P1435 heritage designation. `not_a_sight` treats a state listing as proof
  # that people go and look at the thing, and lets it overrule a type that
  # would otherwise disqualify the row (#65).
  OPTIONAL {{ ?item wdt:P1435 ?heritage }}
}}
"""


POINT_RE = re.compile(r"POINT\(([-\d.eE]+)\s+([-\d.eE]+)\)")


def fetch_spine_qlever(city):
    query = _spine_query(city)
    try:
        raw = http(QLEVER, data=urllib.parse.urlencode({"query": query}),
                   cache_key="qlever-" + query, timeout=300,
                   attempts=3, max_wait=45,
                   accept="application/sparql-results+json")
    except NETWORK_ERRORS as e:
        print(f"   ! QLever basarisiz ({type(e).__name__}); OSM yoluna dusuluyor")
        return {}

    by_item = {}
    for b in raw["results"]["bindings"]:
        qid = b["item"]["value"].rsplit("/", 1)[-1]
        m = POINT_RE.search(b["coord"]["value"])
        if not m:
            continue
        rec = by_item.setdefault(qid, {
            "qid": qid,
            "name": b["label"]["value"],
            "name_lang": b.get("llang", {}).get("value", ""),
            "lon": float(m.group(1)),
            "lat": float(m.group(2)),
            "sitelinks": int(b["sitelinks"]["value"]),
            "types": set(),
            "wikipedia_title": "",
            "wd_description": "",
            "ended": "",
            "heritage": False,
            "source": "qlever",
        })
        # Prefer the English label; the local one is only a fallback for items
        # that have no English label at all.
        if b.get("llang", {}).get("value") == "en" and rec["name_lang"] != "en":
            rec["name"], rec["name_lang"] = b["label"]["value"], "en"
        rec["types"].add(b["typeLabel"]["value"].lower())
        if "article" in b and not rec["wikipedia_title"]:
            rec["wikipedia_title"] = urllib_unquote(
                b["article"]["value"].rsplit("/wiki/", 1)[-1])
        if "desc" in b and not rec["wd_description"]:
            rec["wd_description"] = b["desc"]["value"]
        for key in ("dissolved", "closed"):
            if key in b and not rec["ended"]:
                rec["ended"] = b[key]["value"][:10].lstrip("+")
        rec["heritage"] = rec["heritage"] or "heritage" in b
    return by_item


def fetch_spine_osm(city, osm_by_qid):
    """Build the same records from the QIDs OSM already gave us, via the
    Wikidata Action API -- a different service from WDQS, with its own much
    more generous limits.

    Entities arrive with type QIDs rather than type labels, so the distinct
    types get a second, far smaller resolution pass.
    """
    qids = sorted(osm_by_qid)
    entities = {}
    for i in range(0, len(qids), 40):
        chunk = qids[i:i + 40]
        try:
            d = http(WD_API, {"action": "wbgetentities", "format": "json",
                              "ids": "|".join(chunk),
                              "props": "labels|claims|sitelinks",
                              "languages": city["langs"]}, pause=0.2)
        except NETWORK_ERRORS:
            continue
        entities.update(d.get("entities", {}))

    # Resolve the type QIDs to labels in one pass.
    type_qids = set()
    for ent in entities.values():
        for c in ent.get("claims", {}).get("P31", []):
            v = c["mainsnak"].get("datavalue", {}).get("value", {})
            if isinstance(v, dict) and v.get("id"):
                type_qids.add(v["id"])
    type_labels = {}
    tq = sorted(type_qids)
    for i in range(0, len(tq), 50):
        try:
            d = http(WD_API, {"action": "wbgetentities", "format": "json",
                              "ids": "|".join(tq[i:i + 50]), "props": "labels",
                              "languages": "en"}, pause=0.2)
        except NETWORK_ERRORS:
            continue
        for qid, ent in d.get("entities", {}).items():
            lab = ent.get("labels", {}).get("en", {}).get("value")
            if lab:
                type_labels[qid] = lab.lower()

    out = {}
    for qid, ent in entities.items():
        if "missing" in ent:
            continue
        sitelinks = ent.get("sitelinks", {}) or {}
        if len(sitelinks) < SITELINK_FLOOR:
            continue
        labels = ent.get("labels", {}) or {}
        name = next((labels[l]["value"] for l in city["langs"].split(",")
                     if l in labels), None)
        if not name:
            continue

        coord = None
        for c in ent.get("claims", {}).get("P625", []):
            v = c["mainsnak"].get("datavalue", {}).get("value", {})
            if isinstance(v, dict) and "latitude" in v:
                coord = (v["latitude"], v["longitude"])
                break
        if coord is None:
            continue

        types = set()
        for c in ent.get("claims", {}).get("P31", []):
            v = c["mainsnak"].get("datavalue", {}).get("value", {})
            if isinstance(v, dict) and type_labels.get(v.get("id")):
                types.add(type_labels[v["id"]])
        if not types:
            continue

        out[qid] = {
            "qid": qid, "name": name, "lat": coord[0], "lon": coord[1],
            "sitelinks": len(sitelinks), "types": types,
            "wikipedia_title": sitelinks.get("enwiki", {}).get("title", ""),
            "source": "osm",
        }
    return out


def fetch_spine(city, osm_by_qid):
    wd = fetch_spine_qlever(city)
    if wd:
        print(f"   QLever {len(wd)} varlik")
        return wd
    osm = fetch_spine_osm(city, osm_by_qid)
    print(f"   OSM/ActionAPI yedegi: {len(osm)} varlik")
    return osm


def urllib_unquote(s):
    return urllib.parse.unquote(s).replace("_", " ")


def _word_re(keys):
    """Match a keyword only as a whole word.

    Bare substring matching silently mis-sorted whole categories: "pub" matched
    inside "public university" and "public library", and "bar" inside "border
    barrier", so the Bibliotheque nationale, Humboldt-Universitat and the
    Berlin Wall all landed in `nightlife`. The blacklist had the same flaw --
    "city" matches inside "capacity".
    """
    return re.compile(r"\b(?:" + "|".join(re.escape(k) for k in keys) + r")\b")


CATEGORY_PATTERNS = [(cat, _word_re(keys)) for cat, keys in CATEGORY_RULES]
BLACKLIST_RE = _word_re(TYPE_BLACKLIST)
QUARTER_RE = _word_re(QUARTER_TYPES)


# Categories whose POIs are a service the traveller consumes rather than a
# sight they look at. The distinction decides what a Wikidata closure date
# means: a restaurant that shut in 1913 cannot serve lunch, but the Berlin Wall
# (P576 = 1989), the Bastille (1789) and the Kaiser-Wilhelm-Gedaechtniskirche
# (1943) are among the best-known things to go and see in their cities. P576
# says the *entity* ended, not that the site is gone -- so it is only grounds
# for dropping a row in these three categories.
SERVICE_CATEGORIES = {"food", "nightlife", "shopping"}


def categorise(types):
    joined = " ; ".join(types)
    for cat, pattern in CATEGORY_PATTERNS:
        if pattern.search(joined):
            return cat
    return None


def is_quarter(types):
    return bool(QUARTER_RE.search(" ; ".join(types)))


def classify_spine(by_item, city):
    """Split the raw Wikidata hits into real sights, famous quarters, and drops.

    An item that carries a blacklisted type is only kept if it *also* carries a
    real place type -- "Île de la Cité" is both an island and a quarter, and
    the island is the reason to go there.
    """
    sights, quarters = [], []
    dropped_concept = dropped_uncat = dropped_closed = 0
    dropped_not_a_sight = collections.Counter()

    for rec in by_item.values():
        if re.fullmatch(r"Q\d+", rec["name"]):       # unlabelled entity
            continue
        rec["km"] = haversine_km(city["lat"], city["lon"], rec["lat"], rec["lon"])
        if rec["km"] > city["radius_km"]:
            continue

        # Entities documented like a landmark that are not places to visit --
        # universities, a hospital, metro stations, a television channel, a
        # fire. Applied here rather than to the finished catalogue so a rebuild
        # is correct at source; `pipeline/sight_filter.py` runs the same rule
        # over the already-committed poi.csv (#65).
        reason = not_a_sight(rec["types"], rec.get("heritage"))
        if reason:
            dropped_not_a_sight[reason] += 1
            continue

        blacklisted = [t for t in rec["types"] if BLACKLIST_RE.search(t)]
        clean_types = rec["types"] - set(blacklisted)
        cat = categorise(clean_types) if blacklisted else categorise(rec["types"])

        if blacklisted and not cat:
            # No real place type behind the administrative label. A well known
            # quarter still earns a slot; an ordinary one does not.
            if is_quarter(rec["types"]) and rec["sitelinks"] >= QUARTER_SITELINK_FLOOR:
                rec["category"] = "culture"
                rec["is_quarter"] = True
                quarters.append(rec)
            else:
                dropped_concept += 1
            continue

        if not cat:
            dropped_uncat += 1
            continue

        if cat in SERVICE_CATEGORIES and rec.get("ended"):
            # A venue Wikidata records as closed. The router books meal stops
            # out of `food`, so leaving these in put the Reich Ministry of Food
            # and Agriculture (typed "ministry of food", abolished 1945) and
            # cafes shut since 1910 into itineraries as places to eat.
            dropped_closed += 1
            continue

        rec["category"] = cat
        rec["is_quarter"] = False
        sights.append(rec)

    if dropped_not_a_sight:
        print(f"   gezilebilir yer degil: {dict(dropped_not_a_sight)}")
    return sights, quarters, dropped_concept, dropped_uncat, dropped_closed


# ---------------------------------------------------------------------------
# 2. OSM: operational tags for the spine, plus the food/nightlife long tail
# ---------------------------------------------------------------------------
def fetch_osm_wikidata_tags(city):
    """Every OSM object in the bbox that carries a `wikidata` tag, keyed by QID.

    One bbox query rather than a QID-regex query: the regex form would need a
    ~1500-alternative pattern that Overpass evaluates per object, while this is
    a single index-friendly sweep we join locally.
    """
    b = bbox_for(city["lat"], city["lon"], city["radius_km"])
    q = f"""[out:json][timeout:300];
nwr["wikidata"]({b[0]:.5f},{b[1]:.5f},{b[2]:.5f},{b[3]:.5f});
out center tags;
"""
    data = http(OVERPASS, data="data=" + urllib.parse.quote(q),
                cache_key="overpass-wd-" + q, timeout=300)
    by_qid = {}
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        qid = normalize_qid(tags.get("wikidata"))
        if not qid:
            continue
        # Prefer whichever object carries the most operational detail.
        score = sum(1 for k in ("opening_hours", "fee", "charge", "website", "phone")
                    if k in tags)
        if qid not in by_qid or score > by_qid[qid]["_score"]:
            by_qid[qid] = {"tags": tags, "_score": score}
    return {q: v["tags"] for q, v in by_qid.items()}


def fetch_longtail(city):
    b = bbox_for(city["lat"], city["lon"], city["radius_km"])
    q = f"""[out:json][timeout:180];
(
  node["amenity"~"^(restaurant|cafe|bar|pub|nightclub)$"]["name"]({b[0]:.5f},{b[1]:.5f},{b[2]:.5f},{b[3]:.5f});
  node["amenity"="marketplace"]["name"]({b[0]:.5f},{b[1]:.5f},{b[2]:.5f},{b[3]:.5f});
);
out body 4000;
"""
    data = http(OVERPASS, data="data=" + urllib.parse.quote(q),
                cache_key="overpass-tail-" + q, timeout=300)
    tail = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        name = tags.get("name")
        if not name:
            continue
        am = tags.get("amenity")
        cat = ("nightlife" if am in ("bar", "pub", "nightclub")
               else "shopping" if am == "marketplace" else "food")
        km = haversine_km(city["lat"], city["lon"], el["lat"], el["lon"])
        if km > city["radius_km"]:
            continue
        tail.append({"qid": None, "name": name, "lat": el["lat"], "lon": el["lon"],
                     "sitelinks": 0, "category": cat, "km": km, "is_quarter": False,
                     "wikipedia_title": "", "tags": tags,
                     # These rows are OSM objects to begin with, so their
                     # on-the-ground evidence is the record itself.
                     "has_osm": True})
    return tail


# ---------------------------------------------------------------------------
# 3. Selection
# ---------------------------------------------------------------------------
# Larger than any reachable fame score (sitelinks top out near log1p(140) and
# the pageview term near 2.8), so this acts as a tier boundary rather than as a
# weight to be traded off. Within the service categories, "is on the ground
# today" is not one signal among several -- a venue that does not exist cannot
# be a good answer however famous it is.
SERVICE_NO_OSM_PENALTY = 20.0


def blended_fame(rec, w_s=None, w_p=None):
    """Log-scale blend of documentation and attention.

    Service-category rows with no OpenStreetMap counterpart are demoted below
    every matched one. Wikidata records closure dates unevenly -- Cafe de la
    Regence carries P576, but Cafe Josty, Cafe Bauer and Moka Efti, all long
    gone, carry nothing at all and are typed simply "cafe". Sitelink fame then
    ranked them above Konnopke's Imbiss and Curry 36, which are open, and the
    router booked lunches at them. An OSM match separates the two cleanly:
    across every venue checked by hand, all ten defunct ones were unmatched and
    six of seven operating ones matched.
    """
    w_s = FAME_W_SITELINKS if w_s is None else w_s
    w_p = FAME_W_PAGEVIEWS if w_p is None else w_p
    score = (w_s * math.log1p(rec.get("sitelinks") or 0)
             + w_p * math.log1p(rec.get("rank_pageviews") or 0))
    if rec["category"] in SERVICE_CATEGORIES and not rec.get("has_osm"):
        score -= SERVICE_NO_OSM_PENALTY
    return score


def select_spine(candidates, slots, radius_km, quarter_slots):
    """Damped greedy: at every step take the highest damped fame score, where
    damping grows with how much of that category is already in. Fame still
    decides within a category, but the 30th church is damped far below the
    first theatre, and a hard ceiling stops any single category owning the
    catalogue -- on Istanbul, unceilinged fame gave mosques 65 of 150 slots.

    The radial term is the fix for the outer ring: without it the greedy loop
    exhausts its budget downtown, because that is simply where the famous
    things are.
    """
    ceiling = int(CEILING_FRAC * (slots + int(LONGTAIL_FRAC * slots)))
    remaining = sorted(candidates, key=lambda r: -blended_fame(r))
    picked, per_cat, n_quarter = [], defaultdict(int), 0

    while len(picked) < slots and remaining:
        best, best_score = None, -1.0
        for rec in remaining:
            c = per_cat[rec["category"]]
            if c >= ceiling:
                continue
            if rec["is_quarter"] and n_quarter >= quarter_slots:
                continue
            radial = 1 + DIST_BETA * (rec["km"] / radius_km)
            score = blended_fame(rec) * radial / (1 + DAMP_K * c) ** DAMP_P
            if score > best_score:
                best, best_score = rec, score
        if best is None:
            break
        best["_score"] = best_score
        picked.append(best)
        per_cat[best["category"]] += 1
        n_quarter += best["is_quarter"]
        remaining.remove(best)
    return picked


def select_longtail(tail, slots):
    """Split the budget so the router gets both meal stops and nightlife, and
    prefer venues that carry real opening hours over ones we would have to
    guess for."""
    out, seen = [], set()
    per = {"food": slots // 2, "nightlife": slots - slots // 2}
    for cat, n_slots in per.items():
        pool = sorted((r for r in tail if r["category"] == cat),
                      key=lambda r: (not r["tags"].get("opening_hours"), r["km"]))
        n = 0
        for r in pool:
            if n >= n_slots:
                break
            if r["name"].lower() in seen:
                continue
            seen.add(r["name"].lower())
            out.append(r)
            n += 1
    return out


# ---------------------------------------------------------------------------
# 4. Enrichment for the chosen rows only
# ---------------------------------------------------------------------------
def fetch_descriptions(qids):
    out = {}
    qids = [q for q in qids if q]
    for i in range(0, len(qids), 50):
        chunk = qids[i:i + 50]
        d = http(WD_API, {"action": "wbgetentities", "format": "json",
                          "ids": "|".join(chunk), "props": "descriptions",
                          "languages": "en"})
        for qid, ent in d.get("entities", {}).items():
            desc = ent.get("descriptions", {}).get("en", {}).get("value")
            if desc:
                out[qid] = desc
    return out


def fetch_summary(title):
    if not title:
        return None
    try:
        d = http(f"{WP_REST}/page/summary/{urllib.parse.quote(title.replace(' ', '_'))}",
                 pause=0.15, timeout=45)
    except NETWORK_ERRORS:
        return None
    text = re.sub(r"\s+", " ", (d.get("extract") or "")).strip()
    return text or None


def fetch_summaries_batch(titles):
    """Lead-section extracts for many articles at once, keyed by title.

    `prop=extracts` takes 20 titles per request against the REST summary
    endpoint's one, which is the difference between ~35 requests and ~700 for
    a two-city build. The REST endpoint answers 429 with `Retry-After: 65`
    under that load, and those sleeps were most of the wall clock.
    """
    out = {}
    titles = [t for t in titles if t]
    for i in range(0, len(titles), 20):
        chunk = titles[i:i + 20]
        try:
            d = http(WP_API, {"action": "query", "format": "json",
                              "prop": "extracts", "exintro": "1",
                              "explaintext": "1", "exlimit": "20",
                              "redirects": "1",
                              "titles": "|".join(chunk)}, pause=0.2)
        except NETWORK_ERRORS:
            continue
        for p in d.get("query", {}).get("pages", {}).values():
            title, extract = p.get("title"), p.get("extract")
            if title and extract:
                out[title] = re.sub(r"\s+", " ", extract).strip()
        for n in d.get("query", {}).get("normalized", []) + \
                d.get("query", {}).get("redirects", []):
            if n["to"] in out:
                out[n["from"]] = out[n["to"]]
    return out


def fetch_pageviews_batch(titles):
    """Recent daily pageviews for many articles at once, keyed by title.

    `prop=pageviews` takes 50 titles per request, so the whole candidate pool
    costs ~67 requests instead of one per article. That is what makes it
    affordable to rank *candidates* on attention rather than only annotating
    the ones already chosen -- the per-article REST endpoint would need 3341
    calls and rate-limits hard.

    The window is 60 days (the API's maximum), which is fine for ranking, where
    only the relative order matters. The `monthly_pageviews` column stored in
    the catalogue still comes from the 12-month REST series, which is steadier
    and does not inherit whatever season this was built in.
    """
    out = {}
    titles = [t for t in titles if t]
    for i in range(0, len(titles), 50):
        chunk = titles[i:i + 50]
        params = {"action": "query", "format": "json", "prop": "pageviews",
                  "pvipdays": "60", "titles": "|".join(chunk)}
        # MediaWiki paginates this prop: a 50-title request answers for only
        # part of the batch and hands back a `continue` token for the rest.
        # Ignoring it silently dropped ~80% of the candidates -- the request
        # still returned 200 with a well-formed page list, so nothing looked
        # wrong until the coverage count was read.
        while True:
            try:
                d = http(WP_API, params, pause=0.2)
            except NETWORK_ERRORS:
                break
            for p in d.get("query", {}).get("pages", {}).values():
                title = p.get("title")
                pv = p.get("pageviews")
                if not title or pv is None:
                    continue
                vals = [v for v in pv.values() if v]
                out[title] = int(30 * sum(vals) / len(vals)) if vals else 0
            # `titles` are normalised server-side (underscores, capitalisation);
            # map the query form back so lookups by our own title still hit.
            for n in d.get("query", {}).get("normalized", []):
                if n["to"] in out:
                    out[n["from"]] = out[n["to"]]
            cont = d.get("continue")
            if not cont:
                break
            params = {**params, **cont}
    return out


def fetch_pageviews(title):
    """Mean monthly views over the last full year, via the REST metrics API."""
    if not title:
        return None
    safe = urllib.parse.quote(title.replace(" ", "_"), safe="")
    url = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
           f"en.wikipedia/all-access/user/{safe}/monthly/2025080100/2026073100")
    try:
        d = http(url, pause=0.15, timeout=45)
    except NETWORK_ERRORS:
        return None
    items = d.get("items", [])
    if not items:
        return None
    return int(sum(i["views"] for i in items) / len(items))


def trim_description(text):
    if len(text) <= MAX_DESCRIPTION_CHARS:
        return text
    out = ""
    for sentence in re.split(r"(?<=[.!?]) ", text):
        if out and len(out) + len(sentence) + 1 > MAX_DESCRIPTION_CHARS:
            break
        out = f"{out} {sentence}".strip()
    return out or text[:MAX_DESCRIPTION_CHARS].rsplit(" ", 1)[0]


def build_description(name, city, category, wd_desc):
    if wd_desc:
        article = "an" if wd_desc[0].lower() in "aeiou" else "a"
        if city.lower() in wd_desc.lower():
            return f"{name} is {article} {wd_desc}."
        return f"{name} is {article} {wd_desc}, in {city}."
    generic = {
        "museum": f"A notable museum in {city}.",
        "landmark": f"A well-known landmark in {city}.",
        "religion": f"A historic place of worship in {city}.",
        "history": f"A historic site in {city}.",
        "culture": f"A cultural venue in {city}.",
        "shopping": f"A popular shopping spot in {city}.",
        "food": f"A well-regarded food destination in {city}.",
        "nightlife": f"A lively nightlife spot in {city}.",
        "nature": f"A green space in {city}.",
        "beach": f"A popular beach in {city}.",
    }
    return generic.get(category, f"A point of interest in {city}.")


def parse_opening_hours(raw):
    if not raw:
        return None
    m = re.search(r"(\d{1,2}):\d{2}\s*-\s*(\d{1,2}):\d{2}", str(raw))
    if not m:
        return None
    o, c = int(m.group(1)), int(m.group(2))
    return (o, c) if 0 <= o <= 24 and 0 <= c <= 24 else None


def fame_score(sitelinks, pageviews):
    """Blend the two independent attention signals on a log scale. Pageviews
    measure what people actually look up; sitelinks measure how widely a place
    is documented. Using both means a locally famous but thinly documented
    place, or a heavily documented one nobody reads, still lands sensibly."""
    return 0.6 * math.log1p(pageviews or 0) + 0.8 * math.log1p(sitelinks or 0)


# ---------------------------------------------------------------------------
# 5. Row assembly
# ---------------------------------------------------------------------------
COLUMNS = ["poi_id", "destination_id", "name", "category", "lat", "lon",
           "avg_visit_minutes", "price_level", "popularity_score", "description",
           "open_hour", "close_hour", "wikidata_qid", "wikipedia_title",
           "sitelink_count", "monthly_pageviews", "description_source",
           "hours_source", "price_source"]


def build_rows(city, chosen, osm_by_qid, descriptions, start_id):
    scores = [fame_score(r["sitelinks"], r.get("pageviews")) for r in chosen]
    rows = []
    for i, rec in enumerate(chosen):
        cat = rec["category"]
        avg_min, default_price, default_open, default_close = CATEGORY_DEFAULTS[cat]
        tags = rec.get("tags") or osm_by_qid.get(rec["qid"] or "", {})

        fee = tags.get("fee")
        if fee == "no":
            price_level, price_source = 0, "osm"
        elif fee == "yes" or tags.get("charge"):
            price_level, price_source = max(default_price, 1), "osm"
        else:
            price_level, price_source = default_price, "category_default"

        parsed = parse_opening_hours(tags.get("opening_hours"))
        hours = parsed or (default_open, default_close)
        hours_source = "osm" if parsed else "category_default"

        summary = rec.get("summary")
        # QLever hands back schema:description with the spine query, so the
        # Action API lookup only has to cover whatever it missed.
        wd_desc = rec.get("wd_description") or descriptions.get(rec["qid"] or "")
        if summary:
            description, description_source = trim_description(summary), "wikipedia"
        elif wd_desc:
            description, description_source = \
                build_description(rec["name"], city["city"], cat, wd_desc), "wikidata"
        else:
            description, description_source = \
                build_description(rec["name"], city["city"], cat, None), "template"

        score = fame_score(rec["sitelinks"], rec.get("pageviews"))
        below = sum(1 for s in scores if s <= score)
        popularity = round(2.5 + 2.5 * (below / len(scores)), 2)

        rows.append({
            "poi_id": f"POI{start_id + i:04d}",
            "destination_id": city["code"],
            "name": rec["name"],
            "category": cat,
            "lat": round(rec["lat"], 6),
            "lon": round(rec["lon"], 6),
            "avg_visit_minutes": avg_min,
            "price_level": price_level,
            "popularity_score": popularity,
            "description": description,
            "open_hour": hours[0],
            "close_hour": hours[1],
            "wikidata_qid": rec["qid"] or "",
            "wikipedia_title": rec.get("wikipedia_title") or "",
            "sitelink_count": rec["sitelinks"],
            "monthly_pageviews": rec.get("pageviews") if rec.get("pageviews") is not None else "",
            "description_source": description_source,
            "hours_source": hours_source,
            "price_source": price_source,
        })
    return rows


def build_city(code, start_id, stable_pageviews=False):
    city = CITIES[code]
    print(f"\n{'=' * 68}\n{city['city']} ({code})  merkez {city['lat']},{city['lon']}  "
          f"yaricap {city['radius_km']} km\n{'=' * 68}")

    print("1) OSM wikidata etiketli nesneler (saat/ucret + omurga yedegi)...")
    osm_by_qid = fetch_osm_wikidata_tags(city)
    print(f"   {len(osm_by_qid)} OSM nesnesi, wikidata etiketli")

    print("2) Wikidata omurgasi...")
    by_item = fetch_spine(city, osm_by_qid)
    sights, quarters, n_concept, n_uncat, n_closed = classify_spine(by_item, city)
    print(f"   {len(by_item)} aday -> {len(sights)} yer + {len(quarters)} unlu semt")
    print(f"   {n_concept} kavram elendi, {n_uncat} kategorize edilemedi, "
          f"{n_closed} kapanmis mekan elendi")
    for rec in sights + quarters:
        rec["has_osm"] = rec["qid"] in osm_by_qid
    matched = sum(1 for r in sights + quarters if r["has_osm"])
    print(f"   omurganin {matched}/{len(sights) + len(quarters)}'i OSM etiketine eslesti")

    print("3) OSM uzun kuyrugu (yeme-icme / gece hayati)...")
    tail = fetch_longtail(city)
    print(f"   {len(tail)} aday")

    print("4) Aday pageview'leri (toplu, siralama icin)...")
    candidates = sights + quarters
    pv = fetch_pageviews_batch([r.get("wikipedia_title") for r in candidates])
    for r in candidates:
        r["rank_pageviews"] = pv.get(r.get("wikipedia_title") or "", 0)
    have = sum(1 for r in candidates if r["rank_pageviews"])
    print(f"   {len(candidates)} adayin {have}'inde pageview var")

    target = city.get("target", TARGET)
    longtail_slots = max(8, round(LONGTAIL_FRAC * target))
    print(f"5) Secim (hedef {target}, uzun kuyruk {longtail_slots})...")
    spine_sel = select_spine(candidates, target - longtail_slots,
                             city["radius_km"], QUARTER_SLOTS)
    tail_sel = select_longtail(tail, longtail_slots)
    chosen = spine_sel + tail_sel
    n_q = sum(1 for r in spine_sel if r["is_quarter"])
    print(f"   {len(spine_sel)} omurga ({n_q} semt) + {len(tail_sel)} uzun kuyruk "
          f"= {len(chosen)}")

    print("6) Zenginlestirme (sadece secilenler)...")
    qids = [r["qid"] for r in chosen if r["qid"] and not r.get("wd_description")]
    descriptions = fetch_descriptions(qids)

    summaries = fetch_summaries_batch([r.get("wikipedia_title") for r in chosen])
    n_sum = n_pv = 0
    for rec in chosen:
        title = rec.get("wikipedia_title")
        if not title:
            continue
        rec["summary"] = summaries.get(title)
        if stable_pageviews:
            rec["pageviews"] = fetch_pageviews(title)
        else:
            # The 60-day figure the candidate pass already fetched. The
            # 12-month REST series is steadier -- it does not inherit whatever
            # season the build ran in -- but it is one request per article,
            # which is what made this step take most of an hour. `--stable-
            # pageviews` opts back into it when that matters more than time.
            rec["pageviews"] = rec.get("rank_pageviews") or None
        n_sum += rec["summary"] is not None
        n_pv += rec["pageviews"] is not None
    window = "12 ay" if stable_pageviews else "60 gun"
    print(f"   {len(descriptions)} wikidata aciklamasi, {n_sum} wikipedia ozeti, "
          f"{n_pv} pageview ({window})")

    rows = build_rows(city, chosen, osm_by_qid, descriptions, start_id)
    df = pd.DataFrame(rows)[COLUMNS]
    out = DATA / f"{code}_poi.csv"
    df.to_csv(out, index=False)

    print(f"\n   -> {out.name}  {len(df)} POI")
    print("   kategori:", df.category.value_counts().to_dict())
    print(f"   saat kaynagi osm: {(df.hours_source == 'osm').mean():.1%}   "
          f"ucret kaynagi osm: {(df.price_source == 'osm').mean():.1%}")
    print(f"   aciklama: {df.description_source.value_counts().to_dict()}")
    km = df.apply(lambda r: haversine_km(city["lat"], city["lon"], r.lat, r.lon), axis=1)
    print(f"   mesafe dagilimi 0-2/2-4/4-7 km: "
          f"{(km <= 2).sum()}/{((km > 2) & (km <= 4)).sum()}/{(km > 4).sum()}")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cities", nargs="*", default=list(CITIES),
                    help="sehir kodlari (varsayilan: hepsi)")
    ap.add_argument("--stable-pageviews", action="store_true",
                    help="monthly_pageviews icin 12 aylik REST serisi "
                         "(mevsimden bagimsiz ama makale basina 1 istek)")
    args = ap.parse_args()
    codes = [c.upper() for c in (args.cities or CITIES)]
    for c in codes:
        if c not in CITIES:
            sys.exit(f"bilinmeyen sehir: {c} (mevcut: {', '.join(CITIES)})")

    frames, next_id = [], 1
    for code in codes:
        df = build_city(code, next_id, args.stable_pageviews)
        frames.append(df)
        next_id += len(df)

    combined = pd.concat(frames, ignore_index=True)
    out = DATA / "poi.csv"
    combined.to_csv(out, index=False)
    print(f"\n{'=' * 68}\nBIRLESIK: {out}  {len(combined)} POI, "
          f"{combined.destination_id.nunique()} sehir")


if __name__ == "__main__":
    main()
