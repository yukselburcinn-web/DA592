"""Shared plumbing for the catalogue pipeline: city registry, cached HTTP, geo.

Every network call in this package goes through `http` and lands in a disk
cache keyed by the full request, so a re-run after a tuning change costs
nothing and never re-triggers a rate limit. Delete `.cache/` only if you mean
to re-fetch everything.
"""
import hashlib
import json
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

UA = {"User-Agent": "RoamWise-Research/1.0 (DA592 student project; contact via GitHub)"}

WDQS = "https://query.wikidata.org/sparql"
# QLever serves the same Wikidata dump over SPARQL and is not subject to WDQS's
# outages or its 1 req/min emergency throttle. Note the host: the
# qlever.cs.uni-freiburg.de path 308-redirects here, and urllib does not follow
# a 308 on POST.
QLEVER = "https://qlever.dev/api/wikidata"
OVERPASS = "https://overpass-api.de/api/interpreter"
WD_API = "https://www.wikidata.org/w/api.php"
WV_API = "https://en.wikivoyage.org/w/api.php"
WP_API = "https://en.wikipedia.org/w/api.php"
WP_REST = "https://en.wikipedia.org/api/rest_v1"

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = ROOT / ".cache"
DATA.mkdir(exist_ok=True)
CACHE.mkdir(exist_ok=True)


# `wv_root` is the Wikivoyage article title -- district subpages are discovered
# from it at run time rather than hardcoded, so adding a city means adding a row
# here and nothing else.
#
# This is the single city registry: `build_destinations.py` emits
# destinations.csv from it, and build_catalogue / build_transport / build_demand
# / city_guide all take their city list from here. Adding a destination is one
# row plus three things that cannot be derived:
#   * `target`  -- candidate pools differ by more than 2x between cities, so
#                  measure it with sweep.py rather than copying a number.
#   * an `EDITORIAL` sentence in city_guide.py.
#   * a NUTS 2 region in build_demand.py's DESTINATION_REGION -- and check that
#     the region's seasonal shape is the city's. Berlin is its own NUTS 2
#     region; most cities are not, and a region that carries a coastline or an
#     alpine season will teach the forecaster the wrong curve.
CITIES = {
    "PAR": {
        "code": "PAR", "city": "Paris", "country": "France",
        "lat": 48.8566, "lon": 2.3522, "radius_km": 7.0,
        "wv_root": "Paris", "budget_level": 3, "target": 400,
        "tags": ["culture", "art", "food", "shopping"],
        "langs": "en,fr",
    },
    "BER": {
        "code": "BER", "city": "Berlin", "country": "Germany",
        "lat": 52.5200, "lon": 13.4050, "radius_km": 7.0,
        "wv_root": "Berlin", "budget_level": 2, "target": 300,
        "tags": ["history", "culture", "nightlife", "art"],
        "langs": "en,de",
    },
}



# The fixed 10-category vocabulary CATEGORY_AFFINITY keys off. Order matters:
# the first rule matching any Wikidata type label wins.
CATEGORY_RULES = [
    ("beach", ["beach", "lido"]),
    ("religion", ["mosque", "church", "synagogue", "place of worship", "cathedral",
                  "basilica", "monastery", "chapel", "shrine", "tomb", "mausoleum",
                  "abbey", "temple"]),
    ("museum", ["museum", "art gallery", "gallery", "kunsthalle"]),
    # "archaeological site", not bare "archaeological": the loose form matched
    # "archaeological artifact" and pulled museum pieces (the Venus de Milo)
    # into the catalogue as though they were places.
    ("history", ["archaeological site", "ruin", "ancient", "cistern", "defensive wall",
                 "city wall", "fortification", "citadel", "aqueduct", "historic site",
                 "necropolis", "memorial", "cemetery", "bunker", "catacomb"]),
    ("shopping", ["bazaar", "market", "shopping", "mall", "department store", "arcade"]),
    ("nightlife", ["nightclub", "bar", "pub", "cabaret", "discotheque"]),
    ("food", ["restaurant", "cafe", "café", "food", "brasserie", "brewery"]),
    ("nature", ["park", "garden", "forest", "island", "hill", "grove", "bay", "lake",
                "nature reserve", "arboretum", "botanical", "canal", "zoo"]),
    # "university" is deliberately absent. It admitted every degree-granting
    # institution in both cities as a cultural sight -- Sorbonne University,
    # Sciences Po, TU Berlin, ESCP Business School, the College de France, and
    # the Charite, which is typed "university hospital". 26 of the 104 rows in
    # `culture` were institutions of this kind, and they reached itineraries: a
    # Budget Backpacker's Paris culture day opened Sorbonne -> Sciences Po
    # (#65). A university building worth seeing almost always also carries a
    # heritage listing or an architectural type, which is what keeps it.
    ("culture", ["theatre", "theater", "opera", "concert", "cultural", "stadium",
                 "aquarium", "library", "arena", "cinema", "philharmonic",
                 "planetarium"]),
    # "station" is likewise absent, and for the same reason: it made a landmark
    # of 15 metro and mainline stations, and -- because the match is on a type
    # *label* -- of a "television station" and a "radio station" too, which is
    # how the France 3 television channel came to be a Paris landmark. A
    # station that is a sight in its own right is one that also carries
    # `palace`, `museum` or a heritage listing, and those still admit it.
    ("landmark", ["palace", "tower", "bridge", "gate", "fountain", "lighthouse",
                  "monument", "obelisk", "square", "architectural", "building",
                  "structure", "pavilion", "villa", "mansion", "column",
                  "castle", "hôtel particulier", "observation"]),
]


# ---------------------------------------------------------------------------
# What counts as a place a traveller can visit
# ---------------------------------------------------------------------------
# `CATEGORY_RULES` in build_catalogue.py reads a POI's category off its
# Wikidata type label, and several of those keywords admit whole classes of
# entity that are documented like a landmark but cannot be a stop on a day
# out. `university` put Sorbonne University, Sciences Po, TU Berlin, ESCP
# Business School and the Charite (a "university hospital") into `culture`;
# `station` put eighteen metro and mainline stations into `landmark`. 71 of
# 700 catalogue rows -- 10.1% -- were entities of this kind, and they reached
# real itineraries: a Budget Backpacker's Paris culture day opened Sorbonne
# University -> Sciences Po, and a Luxury Traveler's first day contained
# "2019 fire at Notre-Dame de Paris", which is an event (#65).
#
# This is a *rule* fix, not a data fix: every one of those rows is correctly
# transcribed from Wikidata. What was wrong is the rule that called them
# sights.
NOT_A_SIGHT_TYPES = {
    "transit station": ["station", "stop"],
    "university or hospital": ["university", "college", "business school",
                               "higher education", "grand établissement",
                               "medical school", "hospital"],
    "event, not a place": ["fire", "occurrence", "event", "disaster",
                           "aviation accident"],
    "demolished": ["destroyed building", "demolished"],
    "organisation, not a place": ["nonprofit organization", "business",
                                  "enterprise", "broadcaster", "television channel",
                                  "record label", "publisher"],
}

# A type that would otherwise disqualify a row is overruled by evidence that
# people go and look at the thing. Both signals are external to this project:
# P1435 is a state heritage listing, and `tourist attraction` is Wikidata's own
# statement about the entity.
#
# This matters more than the exclusions do. Without it the rule takes the
# Berlin Wall ("destroyed building or structure"), the Bastille ("demolished"),
# Montmartre and the Tuileries Garden -- which is precisely the mistake
# REPORT.md section 5 already argues against for closure dates: P576 and its
# neighbours record that an *entity* ended, not that its site is gone. Of 700
# rows, 287 carry one of these two rescues.
SIGHT_RESCUE_TYPES = ["tourist attraction"]

# `station` inside "television station" is a television channel, not a train;
# `palace`/`museum` on a station row means the building is the reason to go.
_STATION_OVERRIDES = ["palace", "museum", "television", "radio"]


def not_a_sight(type_labels, has_heritage_listing=False):
    """Why this entity is not a place to visit, or None if it is one.

    `type_labels` is the entity's Wikidata P31 labels; `has_heritage_listing`
    is whether it carries P1435. Matching is on the type *label*, the same way
    CATEGORY_RULES reads it, so "public research university" matches
    "university".

    A disqualifying type only counts when it is the entity's whole story. This
    is the same test `build_catalogue.classify_spine` already applies to
    TYPE_BLACKLIST -- "an item that carries a blacklisted type is only kept if
    it *also* carries a real place type" -- and without it the rule takes real
    places carrying an administrative second type: the Berlinische Galerie and
    the Anne Frank Zentrum are each "art museum; nonprofit organization", the
    Palais de la Decouverte is "museum; event venue", the Parc floral de Paris
    is "urban park; event venue; public garden", the Saint-Ouen flea market is
    "flea market; business; shopping center". Strip the disqualifying labels
    and ask whether a CATEGORY_RULES place type is still standing; if one is,
    that is the reason to go and the row stays.

    It is what makes `demolished` safe to include at all. REPORT.md section 5
    already argues the point for closure dates: an entity ending is not its
    site being gone, and the Berlin Wall, the Bastille and the
    Kaiser-Wilhelm-Gedaechtniskirche are among the best-known things to go and
    see in their cities. Here the Tuileries Palace keeps "palace" and the
    rebuilt Berlin Palace keeps "city palace", so both stay; the Gibbet of
    Montfaucon is only ever "gallows; destroyed building" and goes.

    The bias is deliberately toward keeping. A wrongly kept row is one odd
    suggestion; a wrongly dropped one is a hole nothing downstream can see.
    """
    labels = [str(t).lower() for t in type_labels]
    if has_heritage_listing or _any_type(labels, SIGHT_RESCUE_TYPES):
        return None

    matched, disqualifying = None, set()
    for reason, keywords in NOT_A_SIGHT_TYPES.items():
        hits = [label for label in labels if _any_type([label], keywords)]
        if reason == "transit station":
            hits = [h for h in hits if not _any_type([h], _STATION_OVERRIDES)]
        if hits and matched is None:
            matched = reason
        disqualifying |= set(hits)
    if matched is None:
        return None

    surviving = [label for label in labels if label not in disqualifying]
    if any(pattern.search(label) for label in surviving for _, pattern in _PLACE_PATTERNS):
        return None
    return matched


_PLACE_PATTERNS = [
    (category, re.compile(r"\b(?:" + "|".join(re.escape(k) for k in keys) + r")\b"))
    for category, keys in CATEGORY_RULES
]


def _any_type(labels, keywords):
    return any(keyword in label for label in labels for keyword in keywords)


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def bbox_for(lat, lon, radius_km):
    dlat = radius_km * 1000 / 111_320
    dlon = radius_km * 1000 / (111_320 * math.cos(math.radians(lat)))
    return lat - dlat, lon - dlon, lat + dlat, lon + dlon


def _cache_path(key):
    return CACHE / f"{hashlib.sha1(key.encode()).hexdigest()}.json"


def http(url, params=None, data=None, cache_key=None, pause=0.3, timeout=180,
         attempts=6, max_wait=None, accept="application/json"):
    """GET (params) or POST (data), JSON in and out, cached on disk.

    Retries on the codes these endpoints actually use for backpressure --
    Overpass answers 429/504 under load and WDQS 500s on a slow query -- with a
    linear backoff. Anything else raises: a 400 means the query is wrong and
    retrying it just wastes the endpoint's time.
    """
    key = cache_key or f"{url}|{json.dumps(params, sort_keys=True)}|{data}"
    path = _cache_path(key)
    if path.exists():
        return json.loads(path.read_text())

    for attempt in range(attempts):
        try:
            if data is not None:
                req = urllib.request.Request(
                    url, data=data.encode(),
                    headers={**UA, "Accept": accept})
            else:
                full = f"{url}?{urllib.parse.urlencode(params)}" if params else url
                req = urllib.request.Request(
                    full, headers={**UA, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                payload = json.load(r)
            # Write via a temp file in the same directory and rename, so an
            # interrupted run can never leave a half-written entry that the
            # next run would read back as truncated JSON.
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload))
            tmp.replace(path)
            time.sleep(pause)
            return payload
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < attempts - 1:
                # WDQS answers 429 with "1 req / min" during an outage, so a
                # short backoff just burns attempts against the same wall.
                # Honour Retry-After when it is sent, and never go under a
                # minute on a 429.
                if e.code == 429:
                    wait = max(65, int(e.headers.get("Retry-After") or 0))
                else:
                    wait = 8 * (attempt + 1)
                # An optional source passes max_wait so a long Retry-After
                # (WDQS sends 1000s during an outage) makes us give up and take
                # the fallback path instead of stalling the whole run.
                if max_wait is not None and wait > max_wait:
                    raise
                print(f"      {e.code}, {wait}s bekleniyor...")
                time.sleep(wait)
                continue
            raise
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            if attempt == attempts - 1:
                raise
            print(f"      {type(e).__name__}, tekrar deneniyor...")
            time.sleep(8 * (attempt + 1))
    raise RuntimeError(f"istek basarisiz: {url}")


# What a caller may legitimately shrug off and retry or skip. Catching bare
# `Exception` around a request hides real defects: a missing import inside one
# of these helpers raised NameError on every batch and was swallowed as though
# the network had failed, so the whole pageview pass silently returned nothing.
NETWORK_ERRORS = (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
                  json.JSONDecodeError, ConnectionError, OSError)

QID_RE = re.compile(r"^Q\d+$")


def normalize_qid(raw):
    """OSM's `wikidata` tag is hand-applied free text: it can hold several ids
    ("Q1;Q2") or plain junk. Keep the first well-formed id or nothing."""
    if not raw:
        return None
    for part in re.split(r"[;,\s]+", str(raw).strip()):
        if QID_RE.match(part):
            return part
    return None


def norm_name(s):
    """Fold to bare ascii lowercase words for name matching. Catalogue names
    can be local ("Bahnhof Zoo") where Wikivoyage is English, so a name match
    is only ever used as the last of three tiers."""
    import unicodedata
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()
