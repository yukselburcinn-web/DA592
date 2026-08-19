"""
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
OVERPASS_URL = "https://overpass.openstreetmap.fr/api/interpreter"
WIKIDATA_URL = "https://query.wikidata.org/sparql"
HEADERS = {"User-Agent": "RoamWise-DataFetch/1.0 (student term project; contact via GitHub repo)"}

RADIUS_M = 7000
POIS_PER_CITY = 10

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


def select_diverse(elements: list[dict], n: int) -> list[dict]:
    """Round-robin across categories so one flooded tag (e.g. place_of_worship)
    can't crowd out everything else, then fill any remaining slots."""
    by_category: dict[str, list[dict]] = {}
    for el in elements:
        category = classify(el.get("tags", {}))
        by_category.setdefault(category, []).append(el)
    for cat_elements in by_category.values():
        cat_elements.sort(key=lambda e: "wikidata" not in e.get("tags", {}))

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


def fetch_wikidata_info(qids: list[str]) -> dict[str, dict]:
    if not qids:
        return {}
    values = " ".join(f"wd:{q}" for q in sorted(set(qids)))
    query = f"""
    SELECT ?item ?itemDescription ?sitelinks WHERE {{
      VALUES ?item {{ {values} }}
      ?item wikibase:sitelinks ?sitelinks .
      OPTIONAL {{ ?item schema:description ?itemDescription . FILTER(lang(?itemDescription) = "en") }}
    }}
    """
    last_error = None
    for attempt in range(4):
        try:
            resp = requests.get(
                WIKIDATA_URL, params={"query": query, "format": "json"}, headers=HEADERS, timeout=60
            )
            resp.raise_for_status()
            break
        except requests.exceptions.RequestException as exc:
            last_error = exc
            time.sleep(5 * (attempt + 1))
    else:
        raise RuntimeError(f"Wikidata query failed after retries: {last_error}")

    out = {}
    for row in resp.json()["results"]["bindings"]:
        qid = row["item"]["value"].rsplit("/", 1)[-1]
        out[qid] = {
            "description": row.get("itemDescription", {}).get("value"),
            "sitelinks": int(row["sitelinks"]["value"]),
        }
    return out


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


def poi_row(poi_id: int, dest_id: str, city: str, el: dict, wikidata_info: dict[str, dict]) -> dict:
    tags = el.get("tags", {})
    name = tags["name"]
    category = classify(tags)
    lat = el["lat"] if el["type"] == "node" else el["center"]["lat"]
    lon = el["lon"] if el["type"] == "node" else el["center"]["lon"]

    qid = tags.get("wikidata")
    wd = wikidata_info.get(qid, {}) if qid else {}

    avg_minutes, default_price, default_open, default_close = CATEGORY_DEFAULTS[category]
    fee = tags.get("fee")
    price_level = 0 if fee == "no" else default_price

    hours = parse_opening_hours(tags.get("opening_hours")) or (default_open, default_close)

    return {
        "poi_id": f"POI{poi_id:04d}",
        "destination_id": dest_id,
        "name": name,
        "category": category,
        "lat": round(lat, 6),
        "lon": round(lon, 6),
        "avg_visit_minutes": avg_minutes,
        "price_level": price_level,
        "popularity_score": popularity_from_sitelinks(wd.get("sitelinks")),
        "description": build_description(name, city, category, wd.get("description")),
        "open_hour": hours[0],
        "close_hour": hours[1],
    }


def main():
    all_rows = []
    poi_id = 1
    for dest_id, city, lat, lon in DESTINATIONS:
        print(f"[{dest_id}] querying Overpass ({city})...")
        elements = fetch_overpass(lat, lon)
        elements = dedupe_elements(elements)
        chosen = select_diverse(elements, POIS_PER_CITY)

        qids = [el["tags"]["wikidata"] for el in chosen if "wikidata" in el.get("tags", {})]
        print(f"[{dest_id}] {len(elements)} candidates -> {len(chosen)} selected, {len(qids)} with Wikidata IDs")
        wikidata_info = fetch_wikidata_info(qids)

        for el in chosen:
            all_rows.append(poi_row(poi_id, dest_id, city, el, wikidata_info))
            poi_id += 1

        time.sleep(1)  # be polite to the shared public Overpass/Wikidata endpoints

    poi_df = pd.DataFrame(all_rows)
    poi_df.to_csv(HERE / "poi.csv", index=False)
    print(f"\nWrote poi.csv ({len(poi_df)} rows across {poi_df.destination_id.nunique()} cities)")


if __name__ == "__main__":
    main()
