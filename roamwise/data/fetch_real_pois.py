"""
SUPERSEDED by `pipeline/build_catalogue.py`. Do not run to set the project up.

This writes the old EIGHT-city `poi.csv` (IST, PAR, ROM, BCN, AMS, PRG, VIE,
LIS) at 150 rows each. What ships is Paris and Berlin at 2.5x the depth, and
it is committed -- setup is `streamlit run app.py` and nothing else (see the
README's Quickstart). Running this overwrites the shipped catalogue and leaves
the app half-migrated: two city_guides against eight destinations raises
KeyError in retrieval/corpus.py.

To rebuild the catalogue, use `cd pipeline && python build_catalogue.py PAR BER`
-- that is what produced the shipped file, and unlike this script it ranks
candidates by fame, records a source for every field, and reads its city list
from `common.CITIES`. Kept here for history.

Original description follows.

Replace the procedurally-generated `poi.csv` with real points of interest,
pulled live from OpenStreetMap (Overpass API) and enriched with Wikidata.

Run after `generate_data.py` (which still produces destinations.csv,
transport.csv, user_survey.csv and city_guides/ -- those stay curated/
synthetic, this script only touches poi.csv):

    python fetch_real_pois.py

Requires network access to overpass-api.de and query.wikidata.org. If either
call fails (rate limit, no network), the script raises rather than silently
falling back to synthetic data -- re-run generate_data.py first if you need
an offline poi.csv.

Data flow:
  1. Overpass: for each city, fetch named OSM elements tagged as a tourist
     attraction, museum, historic site, place of worship, park, beach,
     market or nightlife venue within a radius of the city center.
  2. Map each OSM tag combination to RoamWise's fixed category vocabulary
     (the same 10 categories the rest of the codebase -- knowledge graph
     affinities, retrieval, orchestrator -- already keys off of).
  3. Wikidata SPARQL: batch-fetch an English description and sitelink count
     (an editorial-attention proxy, used as popularity_score) for every OSM
     element that carries a `wikidata` tag.
  4. Pick a diverse, popularity-ranked subset per city and write poi.csv
     with the exact same columns generate_data.py produced, so every
     downstream module (knowledge_graph, retrieval, agents, tests) keeps
     working unmodified.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).parent
CACHE_DIR = HERE / ".cache"
OVERPASS_URL = "https://overpass.openstreetmap.fr/api/interpreter"
WIKIDATA_URL = "https://query.wikidata.org/sparql"
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/"
PAGEVIEWS_URL = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
                 "en.wikipedia/all-access/user/{title}/monthly/{start}/{end}")
PAGEVIEW_WINDOW = ("2025010100", "2025123100")  # a full, complete calendar year
HEADERS = {"User-Agent": "RoamWise-DataFetch/1.0 (student term project; contact via GitHub repo)"}

RADIUS_M = 7000
POIS_PER_CITY = 150

# How far a Wikidata entity may sit from the OSM feature before the link counts
# as wrong rather than imprecise.
MAX_ENTITY_DRIFT_KM = 5.0


def haversine_km(lat1, lon1, lat2, lon2):
    import math

    radius = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def cached_json(url: str, cache_key: str, params: dict = None, pause: float = 0.4):
    """GET returning parsed JSON, cached on disk under .cache/.

    The enrichment below makes thousands of small requests, most of which
    return the same answer every run. Caching keeps a re-run to seconds rather
    than half an hour, and keeps the load off shared public endpoints. Returns
    None on failure -- an enrichment gap must degrade one POI, not abort a
    pull that is most of the way through eight cities."""
    import json

    cache_file = CACHE_DIR / (re.sub(r"[^A-Za-z0-9_.-]", "_", cache_key)[:160] + ".json")
    if cache_file.exists():
        try:
            payload = json.loads(cache_file.read_text())
        except json.JSONDecodeError:
            cache_file.unlink()
        else:
            # a cached "this does not exist" is as useful as a cached answer:
            # without it every re-run re-asks for the same missing articles
            return None if payload == {"__absent__": True} else payload

    def remember_absent():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps({"__absent__": True}))
        return None

    delay = 5.0
    for attempt in range(4):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=45)
            resp.raise_for_status()
            payload = resp.json()
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(payload))
            time.sleep(pause)
            return payload
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            # A 404 means the article simply does not exist -- retrying it three
            # times behind an exponential backoff burns ~35 seconds to learn
            # nothing. Only 429 and server errors are worth another attempt.
            if status and status < 500 and status != 429:
                return remember_absent()
            if attempt == 3:
                return None
            time.sleep(delay)
            delay *= 2
        except (requests.exceptions.RequestException, ValueError):
            if attempt == 3:
                return None
            time.sleep(delay)
            delay *= 2
    return None

# OSM tag -> RoamWise category. Order matters: first matching rule wins.
TAG_CATEGORY_RULES = [
    (("natural", "beach"), "beach"),
    (("amenity", "place_of_worship"), "religion"),
    (("tourism", "museum"), "museum"),
    (("tourism", "gallery"), "museum"),
    (("amenity", "marketplace"), "food"),
    (("amenity", "food_court"), "food"),
    (("shop", "mall"), "shopping"),
    (("shop", "department_store"), "shopping"),
    (("amenity", "bar"), "nightlife"),
    (("amenity", "nightclub"), "nightlife"),
    (("amenity", "pub"), "nightlife"),
    (("leisure", "park"), "nature"),
    (("leisure", "garden"), "nature"),
    (("amenity", "theatre"), "culture"),
    (("amenity", "arts_centre"), "culture"),
    (("tourism", "zoo"), "culture"),
    (("tourism", "theme_park"), "culture"),
    (("historic", "archaeological_site"), "history"),
    (("historic", "ruins"), "history"),
    (("historic", "memorial"), "history"),
    (("historic", None), "landmark"),
    (("tourism", "viewpoint"), "landmark"),
    (("tourism", "attraction"), "landmark"),
]

# Fallback defaults when OSM has no useful signal for these fields.
CATEGORY_DEFAULTS = {
    # category: (avg_visit_minutes, price_level, open_hour, close_hour)
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

DESTINATIONS = [
    ("IST", "Istanbul", 41.0082, 28.9784),
    ("PAR", "Paris", 48.8566, 2.3522),
    ("ROM", "Rome", 41.9028, 12.4964),
    ("BCN", "Barcelona", 41.3874, 2.1686),
    ("AMS", "Amsterdam", 52.3676, 4.9041),
    ("PRG", "Prague", 50.0755, 14.4378),
    ("VIE", "Vienna", 48.2082, 16.3738),
    ("LIS", "Lisbon", 38.7223, -9.1393),
]


# Grouped by OSM key so the query issues one bbox-filtered statement per key
# (with a regex alternation over values) instead of one per (key, value) pair
# -- Overpass's `around` distance filter is expensive per-statement, so we use
# a plain bounding box instead, which is index-friendly and far faster for a
# union query like this one.
TAG_GROUPS = {
    "natural": ["beach"],
    "amenity": ["place_of_worship", "marketplace", "food_court", "bar", "nightclub", "pub", "theatre", "arts_centre"],
    "tourism": ["museum", "gallery", "attraction", "zoo", "theme_park", "viewpoint"],
    "shop": ["mall", "department_store"],
    "leisure": ["park", "garden"],
}


def bbox_for(lat: float, lon: float, radius_m: int) -> tuple[float, float, float, float]:
    import math

    dlat = radius_m / 111_320
    dlon = radius_m / (111_320 * math.cos(math.radians(lat)))
    return lat - dlat, lon - dlon, lat + dlat, lon + dlon


def _group_query(clause: str, bbox: tuple[float, float, float, float]) -> str:
    south, west, north, east = bbox
    return f"""
[out:json][timeout:30][bbox:{south},{west},{north},{east}];
({clause});
out center tags;
"""


def _run_query(query: str, attempts: int = 5) -> list[dict]:
    last_error = None
    for attempt in range(attempts):
        try:
            resp = requests.post(OVERPASS_URL, data={"data": query}, headers=HEADERS, timeout=90)
            resp.raise_for_status()
            data = resp.json()
            if "remark" in data:
                raise RuntimeError(f"Overpass returned an error: {data['remark']}")
            return data["elements"]
        except (requests.exceptions.RequestException, RuntimeError) as exc:
            last_error = exc
            time.sleep(8 * (attempt + 1))
    raise RuntimeError(f"Overpass query failed after {attempts} attempts: {last_error}")


def fetch_overpass(lat: float, lon: float) -> list[dict]:
    """One small request per tag group instead of a single combined query --
    the shared public Overpass instance reliably times out on the combined
    version for OSM-dense city centers, but handles each group in isolation
    within a few seconds."""
    bbox = bbox_for(lat, lon, RADIUS_M)
    clauses = {"historic": 'nwr["name"]["historic"];'}
    for key, values in TAG_GROUPS.items():
        alternation = "|".join(values)
        clauses[key] = f'nwr["name"]["{key}"~"^({alternation})$"];'

    elements = []
    for key, clause in clauses.items():
        elements.extend(_run_query(_group_query(clause, bbox)))
        time.sleep(1.5)
    return elements


def classify(tags: dict) -> str | None:
    for (key, value), category in TAG_CATEGORY_RULES:
        if key in tags and (value is None or tags[key] == value):
            return category
    return None


def dedupe_elements(elements: list[dict]) -> list[dict]:
    seen = {}
    for el in elements:
        tags = el.get("tags", {})
        name = tags.get("name", "").strip()
        if not name:
            continue
        category = classify(tags)
        if category is None:
            continue
        key = name.lower()
        # Prefer the element that already carries a wikidata tag (cleaner match downstream).
        if key not in seen or ("wikidata" in tags and "wikidata" not in seen[key].get("tags", {})):
            seen[key] = el
    return list(seen.values())


def fame_score(info: dict) -> float:
    """Blend the two independent attention signals on a log scale. Pageviews
    measure what people actually look up; sitelink count measures how widely a
    place is documented. Using both means a locally famous place that is thinly
    documented, or a heavily documented one nobody reads, still lands sensibly."""
    import math

    views = math.log1p(info.get("pageviews") or 0)
    links = math.log1p(info.get("sitelinks") or 0)
    return 0.6 * views + 0.8 * links


def select_diverse(elements: list[dict], n: int, ranking: dict[str, dict]) -> list[dict]:
    """Round-robin across categories so one flooded tag (e.g. place_of_worship)
    cannot crowd out everything else -- but rank *within* each category by how
    well known the place is.

    The ranking is the point. This used to sort each category only by whether
    the element carried a `wikidata` tag and then take whatever order Overpass
    happened to return, which meant the 150 POIs kept per city were an
    arbitrary sample: Prague's catalogue came back without Prague Castle,
    Charles Bridge or the Astronomical Clock, and Paris without the Louvre,
    the Musée d'Orsay or the Arc de Triomphe."""
    def score(el):
        qid = normalize_qid(el.get("tags", {}).get("wikidata"))
        return fame_score(ranking.get(qid, {})) if qid else 0.0

    by_category: dict[str, list[dict]] = {}
    for el in elements:
        category = classify(el.get("tags", {}))
        by_category.setdefault(category, []).append(el)
    for cat_elements in by_category.values():
        cat_elements.sort(key=score, reverse=True)

    picked, order = [], list(by_category.keys())
    idx = 0
    while len(picked) < n and any(by_category.values()):
        cat = order[idx % len(order)]
        if by_category[cat]:
            picked.append(by_category[cat].pop(0))
        idx += 1
        if idx > 10_000:
            break
    return picked[:n]


QID_RE = re.compile(r"^Q\d+$")


def normalize_qid(raw: str | None) -> str | None:
    """OSM's `wikidata` tag is hand-applied free text: it can hold several ids
    ("Q1;Q2") or plain junk. One malformed value makes a whole batch query
    fail, so keep the first well-formed id or nothing."""
    if not raw:
        return None
    for part in re.split(r"[;,\s]+", raw.strip()):
        if QID_RE.match(part):
            return part
    return None


def fetch_wikidata_info(qids: list[str]) -> dict[str, dict]:
    """Sitelink count, English description and English Wikipedia title per
    entity, in chunks. This now runs over *every* candidate rather than over an
    already-chosen subset: the selection below ranks on these numbers, so
    fetching them afterwards -- as this script used to -- meant the 150 kept
    per city were an arbitrary sample that the popularity score then ranked
    after the fact."""
    qids = sorted({q for q in (normalize_qid(q) for q in qids) if q})
    out: dict[str, dict] = {}
    for i in range(0, len(qids), 400):
        chunk = qids[i:i + 400]
        values = " ".join(f"wd:{q}" for q in chunk)
        query = f"""
        SELECT ?item ?itemDescription ?sitelinks ?articleName WHERE {{
          VALUES ?item {{ {values} }}
          ?item wikibase:sitelinks ?sitelinks .
          OPTIONAL {{ ?item schema:description ?itemDescription . FILTER(lang(?itemDescription) = "en") }}
          OPTIONAL {{
            ?article schema:about ?item ;
                     schema:isPartOf <https://en.wikipedia.org/> ;
                     schema:name ?articleName .
          }}
        }}
        """
        payload = cached_json(WIKIDATA_URL, f"wd_{chunk[0]}_{len(chunk)}",
                              params={"query": query, "format": "json"}, pause=1.0)
        if not payload:
            print(f"    ! Wikidata chunk starting {chunk[0]} failed; continuing without it")
            continue
        for row in payload["results"]["bindings"]:
            qid = row["item"]["value"].rsplit("/", 1)[-1]
            out[qid] = {
                "description": row.get("itemDescription", {}).get("value"),
                "sitelinks": int(row["sitelinks"]["value"]),
                "title": row.get("articleName", {}).get("value"),
            }
    return out


def fetch_article_coords(titles: list[str]) -> dict[str, tuple[float, float]]:
    """Which of these Wikipedia articles are about a *place*?

    OSM's `wikidata` tag often points at a concept rather than the feature: the
    red panda enclosure at Prague Zoo carries the QID for the *species*, a
    statue of Diana the QID for the *goddess*, the metre marker in Paris the
    QID for the *unit of length*. Because a species or a worldwide institution
    collects far more sitelinks than any single building, these outrank real
    landmarks -- Prague's top-scoring POI was the Wikidata entity for the
    Church of Jesus Christ of Latter-day Saints, the organisation.

    An article about a place carries coordinates; an article about a concept
    does not. Asking Wikipedia rather than Wikidata's SPARQL endpoint is
    deliberate: `prop=coordinates` takes 50 titles per call and answers in
    milliseconds."""
    import urllib.parse

    coords: dict[str, tuple[float, float]] = {}
    for i in range(0, len(titles), 50):
        chunk = titles[i:i + 50]
        payload = cached_json(WIKIPEDIA_API,
                              f"wpcoord_{re.sub(r'[^A-Za-z0-9]', '', chunk[0])[:40]}_{len(chunk)}",
                              # colimit defaults to 10 results per query, so a
                              # 50-title batch silently returns coordinates for
                              # only the first handful and the rest look like
                              # concepts. "max" is required, not cosmetic.
                              params={"action": "query", "prop": "coordinates",
                                      "titles": "|".join(chunk), "colimit": "max",
                                      "format": "json", "formatversion": "2"},
                              pause=0.3)
        if not payload:
            continue
        for page in (payload.get("query") or {}).get("pages") or []:
            position = (page.get("coordinates") or [None])[0]
            if position:
                coords[page["title"]] = (position["lat"], position["lon"])
        # the API normalises titles; map back so lookups by our string still hit
        for norm in (payload.get("query") or {}).get("normalized") or []:
            if norm["to"] in coords:
                coords[norm["from"]] = coords[norm["to"]]
    return coords


def fetch_pageviews(title: str) -> int | None:
    """Mean monthly English-Wikipedia pageviews over a complete calendar year.

    Sitelink count measures how widely documented a place is; pageviews measure
    whether anyone actually looks it up. They disagree often enough to be worth
    having both -- a cathedral documented in 40 languages that nobody reads
    should not outrank the landmark every visitor searches for."""
    import urllib.parse

    encoded = urllib.parse.quote(title.replace(" ", "_"), safe="")
    payload = cached_json(PAGEVIEWS_URL.format(title=encoded, start=PAGEVIEW_WINDOW[0],
                                               end=PAGEVIEW_WINDOW[1]),
                          f"pv_{encoded}", pause=0.35)
    if not payload or not payload.get("items"):
        return None
    views = [item["views"] for item in payload["items"]]
    return int(sum(views) / len(views))


def fetch_summary(title: str) -> str | None:
    """First paragraphs of the English Wikipedia article, trimmed to whole
    sentences. Replaces the templated `build_description` output ("X is a
    türbe in Turkey, in Istanbul") with prose a retrieval layer can actually
    match against."""
    import urllib.parse

    encoded = urllib.parse.quote(title.replace(" ", "_"), safe="")
    payload = cached_json(WIKIPEDIA_SUMMARY + encoded, f"sum_{encoded}", pause=0.35)
    if not payload:
        return None
    extract = re.sub(r"\s+", " ", (payload.get("extract") or "")).strip()
    return extract or None


MAX_DESCRIPTION_CHARS = 500


def trim_description(text: str) -> str:
    """Wikipedia extracts run past a thousand characters; keep whole sentences
    up to a budget so POI cards and agent prompts stay readable."""
    if len(text) <= MAX_DESCRIPTION_CHARS:
        return text
    out = ""
    for sentence in re.split(r"(?<=[.!?]) ", text):
        if out and len(out) + len(sentence) + 1 > MAX_DESCRIPTION_CHARS:
            break
        out = f"{out} {sentence}".strip()
    return out or text[:MAX_DESCRIPTION_CHARS].rsplit(" ", 1)[0]


def parse_opening_hours(raw: str | None) -> tuple[int, int] | None:
    if not raw:
        return None
    match = re.search(r"(\d{1,2}):\d{2}\s*-\s*(\d{1,2}):\d{2}", raw)
    if not match:
        return None
    open_h, close_h = int(match.group(1)), int(match.group(2))
    if 0 <= open_h <= 24 and 0 <= close_h <= 24:
        return open_h, close_h
    return None


def popularity_from_sitelinks(sitelinks: int | None) -> float:
    import math

    if not sitelinks:
        return 4.0
    return round(min(4.9, 3.9 + 0.15 * math.log1p(sitelinks)), 1)


def build_description(name: str, city: str, category: str, wikidata_desc: str | None) -> str:
    if wikidata_desc:
        article = "an" if wikidata_desc[0].lower() in "aeiou" else "a"
        normalized_desc = wikidata_desc.replace("İ", "I").lower()
        if city.lower() in normalized_desc:
            return f"{name} is {article} {wikidata_desc}."
        return f"{name} is {article} {wikidata_desc}, in {city}."
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


def popularity_from_fame(info: dict, city_scores: list[float]) -> float:
    """A within-city percentile of the fame score, mapped onto the 2.5-5.0 range
    the rest of the codebase already expects. The previous formula ran off
    sitelink count alone and compressed every POI into eight distinct values
    between 4.0 and 4.7, which gave the ranking almost nothing to work with."""
    if not city_scores:
        return 4.0
    score = fame_score(info)
    below = sum(1 for s in city_scores if s <= score)
    return round(2.5 + 2.5 * (below / len(city_scores)), 2)


def poi_row(poi_id: int, dest_id: str, city: str, el: dict, wikidata_info: dict[str, dict],
            city_scores: list[float]) -> dict:
    tags = el.get("tags", {})
    name = tags["name"]
    category = classify(tags)
    lat = el["lat"] if el["type"] == "node" else el["center"]["lat"]
    lon = el["lon"] if el["type"] == "node" else el["center"]["lon"]

    qid = normalize_qid(tags.get("wikidata"))
    wd = wikidata_info.get(qid, {}) if qid else {}

    avg_minutes, default_price, default_open, default_close = CATEGORY_DEFAULTS[category]
    fee = tags.get("fee")
    if fee == "no":
        price_level, price_source = 0, "osm"
    elif fee == "yes" or tags.get("charge"):
        price_level, price_source = max(default_price, 1), "osm"
    else:
        price_level, price_source = default_price, "category_default"

    parsed_hours = parse_opening_hours(tags.get("opening_hours"))
    hours = parsed_hours or (default_open, default_close)
    hours_source = "osm" if parsed_hours else "category_default"

    summary = wd.get("summary")
    if summary:
        description, description_source = trim_description(summary), "wikipedia"
    elif wd.get("description"):
        description, description_source = build_description(name, city, category, wd["description"]), "wikidata"
    else:
        description, description_source = build_description(name, city, category, None), "template"

    views = wd.get("pageviews")
    return {
        "poi_id": f"POI{poi_id:04d}",
        "destination_id": dest_id,
        "name": name,
        "category": category,
        "lat": round(lat, 6),
        "lon": round(lon, 6),
        "avg_visit_minutes": avg_minutes,
        "price_level": price_level,
        "popularity_score": popularity_from_fame(wd, city_scores),
        "description": description,
        "open_hour": hours[0],
        "close_hour": hours[1],
        # provenance -- lets a reader tell an observed value from a fallback
        "wikidata_qid": qid or "",
        "wikipedia_title": wd.get("title") or "",
        "sitelink_count": wd.get("sitelinks") or 0,
        "monthly_pageviews": views if views is not None else "",
        "description_source": description_source,
        "hours_source": hours_source,
        "price_source": price_source,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Refresh poi.csv from OSM/Wikidata/Wikipedia.")
    parser.add_argument("--limit-cities", default="",
                        help="comma-separated destination_ids, for a partial run while iterating")
    parser.add_argument("--out", default="poi.csv", help="output filename inside this directory")
    args = parser.parse_args()

    wanted = {c.strip() for c in args.limit_cities.split(",") if c.strip()}
    destinations = [d for d in DESTINATIONS if not wanted or d[0] in wanted]

    all_rows = []
    poi_id = 1
    for dest_id, city, lat, lon in destinations:
        print(f"[{dest_id}] querying Overpass ({city})...")
        elements = dedupe_elements(fetch_overpass(lat, lon))

        # Enrich every candidate before choosing, not after: the choice depends
        # on these numbers.
        qids = [t for t in (normalize_qid(el.get("tags", {}).get("wikidata")) for el in elements) if t]
        info = fetch_wikidata_info(qids)
        print(f"[{dest_id}] {len(elements)} candidates, {len(info)} enriched from Wikidata")

        # Place gate: drop entities that are concepts rather than places, and
        # links that point somewhere else entirely.
        titled = {qid: v["title"] for qid, v in info.items() if v.get("title")}
        coords = fetch_article_coords(sorted(set(titled.values())))
        element_positions = {}
        for el in elements:
            qid = normalize_qid(el.get("tags", {}).get("wikidata"))
            if qid:
                element_positions[qid] = (el["lat"] if el["type"] == "node" else el["center"]["lat"],
                                          el["lon"] if el["type"] == "node" else el["center"]["lon"])
        dropped_concept = dropped_far = 0
        for qid, title in titled.items():
            position = coords.get(title)
            if position is None:
                info.pop(qid, None)
                dropped_concept += 1
            elif qid in element_positions and \
                    haversine_km(*element_positions[qid], *position) > MAX_ENTITY_DRIFT_KM:
                info.pop(qid, None)
                dropped_far += 1
        elements = [el for el in elements
                    if not (normalize_qid(el.get("tags", {}).get("wikidata")) or "") in
                    (set(titled) - set(info))]
        print(f"[{dest_id}] place gate: dropped {dropped_concept} non-places, {dropped_far} mislinked")

        # Selection ranks on sitelink count, which we already have for every
        # candidate. Pageviews would refine the ordering, but one request per
        # candidate over a few thousand candidates runs into the pageviews
        # API's rate limit and costs hours; they are fetched below for the
        # chosen POIs instead, where they sharpen popularity_score.
        chosen = select_diverse(elements, POIS_PER_CITY, info)

        # Pageviews and rich descriptions for the final cut only.
        for el in chosen:
            qid = normalize_qid(el.get("tags", {}).get("wikidata"))
            entry = info.get(qid) if qid else None
            if entry and entry.get("title"):
                entry["pageviews"] = fetch_pageviews(entry["title"])
                entry["summary"] = fetch_summary(entry["title"])

        city_scores = [fame_score(info[normalize_qid(el["tags"]["wikidata"])])
                       for el in chosen
                       if normalize_qid(el.get("tags", {}).get("wikidata")) in info]
        for el in chosen:
            all_rows.append(poi_row(poi_id, dest_id, city, el, info, city_scores))
            poi_id += 1
        print(f"[{dest_id}] selected {len(chosen)}")

        time.sleep(1)  # be polite to the shared public Overpass/Wikidata endpoints

    poi_df = pd.DataFrame(all_rows)
    poi_df.to_csv(HERE / args.out, index=False)
    print(f"\nWrote {args.out} ({len(poi_df)} rows across {poi_df.destination_id.nunique()} cities)")
    for column in ["description_source", "hours_source", "price_source"]:
        print(f"  {column}: {poi_df[column].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
