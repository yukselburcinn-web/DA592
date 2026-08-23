"""Build transport.csv from OSM, ranked by Wikidata notability.

"Every transport point" is the wrong target, and the graph says why.
`multi_hop_transport_to_poi` answers "which sights sit near a transport hub"
with a 3 km radius (knowledge_graph/build_graph.py). Paris has 365 rail
stations inside 45 km and 334 metro stations inside the catalogue's own bbox;
feed those in and every POI is within 3 km of a hub, so the relation stops
telling anything apart. The repo's two-rows-per-city goes wrong the other way:
Charles de Gaulle is 23 km out, so its 3 km circle is empty and only the one
station does any work.

The useful set is the gateways -- where a traveller actually arrives. Two OSM
tags find the candidates and Wikidata ranks them:

  * Airports: `aerodrome=international`. Filtering on an IATA code alone
    returned 7 "airports" for Paris, five of them air bases or general
    aviation fields.
  * Stations: `railway=station` + `train=yes` + a `wikidata` tag. `usage=main`
    looks like the right tag and is not -- it classifies the *track* as a main
    line, so it returned Pantin, Rosny-sous-Bois and Tournan while missing
    every Paris terminus and Berlin Hauptbahnhof outright.
  * Ranking is sitelink count, the same fame signal the POI catalogue uses.
    It orders these exactly right: Paris comes back Gare du Nord, Gare de
    Lyon, Montparnasse, Gare de l'Est, Saint-Lazare, Austerlitz -- the six
    termini, in order -- and Berlin leads with Hauptbahnhof.

    python build_transport.py PAR BER --write
"""
import argparse
import math
import sys
import urllib.parse

import pandas as pd

from common import (CITIES, DATA, OVERPASS, QLEVER, haversine_km, http,
                    normalize_qid)

SEARCH_KM = 45.0            # airports sit well outside the catalogue's 7 km
MAX_STATIONS = 6             # the termini; below them come commuter stops
HUB_RADIUS_KM = 3.0          # what build_graph.py calls "near a hub"


def _bbox(city, km):
    dlat = km * 1000 / 111_320
    dlon = km * 1000 / (111_320 * math.cos(math.radians(city["lat"])))
    return (city["lat"] - dlat, city["lon"] - dlon,
            city["lat"] + dlat, city["lon"] + dlon)


def fetch_candidates(city):
    b = _bbox(city, SEARCH_KM)
    box = f"{b[0]:.4f},{b[1]:.4f},{b[2]:.4f},{b[3]:.4f}"
    q = f"""[out:json][timeout:180];
(
  nwr["aeroway"="aerodrome"]["aerodrome"="international"]["name"]({box});
  nwr["railway"="station"]["train"="yes"]["wikidata"]["name"]({box});
  nwr["amenity"="bus_station"]["name"]["wikidata"]({box});
);
out center tags;
"""
    data = http(OVERPASS, data="data=" + urllib.parse.quote(q),
                cache_key="hubs2-" + q, timeout=300)

    out = {}
    for el in data.get("elements", []):
        t = el.get("tags", {})
        lat = el.get("lat") or el.get("center", {}).get("lat")
        lon = el.get("lon") or el.get("center", {}).get("lon")
        if lat is None or lon is None:
            continue
        if t.get("aeroway") == "aerodrome":
            ttype = "airport"
        elif t.get("railway") == "station":
            ttype = "train_station"
        else:
            ttype = "bus_station"
        name = t.get("name:en") or t["name"]
        km = haversine_km(city["lat"], city["lon"], lat, lon)
        # Dedupe on the Wikidata id, not the name. OSM carries one station
        # under several names and geometries -- "Berlin Hauptbahnhof (tief)"
        # and "Berlin Central Station" are the same building, as are the three
        # Ostkreuz platform groups -- and a name key let every variant through.
        qid = normalize_qid(t.get("wikidata"))
        key = qid or (ttype, name.lower())
        if key not in out or km < out[key]["km"]:
            out[key] = {"name": name, "type": ttype, "lat": round(lat, 6),
                        "lon": round(lon, 6), "km": km,
                        "iata": t.get("iata", ""), "qid": qid, "sitelinks": 0}
    return list(out.values())


def add_sitelinks(hubs):
    qids = sorted({h["qid"] for h in hubs if h["qid"]})
    if not qids:
        return hubs
    values = " ".join(f"wd:{q}" for q in qids)
    query = f"""PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wikibase: <http://wikiba.se/ontology#>
SELECT ?item ?sitelinks WHERE {{
  VALUES ?item {{ {values} }}
  ?item wikibase:sitelinks ?sitelinks .
}}"""
    r = http(QLEVER, data=urllib.parse.urlencode({"query": query}),
             cache_key="hubsl-" + query,
             accept="application/sparql-results+json", timeout=120)
    by_qid = {b["item"]["value"].rsplit("/", 1)[-1]: int(b["sitelinks"]["value"])
              for b in r["results"]["bindings"]}
    for h in hubs:
        h["sitelinks"] = by_qid.get(h["qid"] or "", 0)
    return hubs


def select(hubs):
    """Airports and coach terminals are few enough to keep whole; the station
    list is cut to the top N by notability.

    A count, not a sitelink threshold, because the threshold does not travel:
    at >= 15 sitelinks Paris kept 30 stations and Berlin 21, and the cut
    landed in a different place in each city's distribution. Six is where the
    termini stop and the commuter interchanges begin in both.
    """
    stations = sorted((h for h in hubs if h["type"] == "train_station"),
                      key=lambda h: -h["sitelinks"])[:MAX_STATIONS]
    gateways = [h for h in hubs if h["type"] != "train_station"]
    return sorted(gateways + stations, key=lambda h: (h["type"], -h["sitelinks"]))


def report_reach(code, hubs, poi):
    """What share of the catalogue this actually makes 'near a hub'.

    The number to watch: at 100% the graph relation carries no information,
    and with the repo's two rows it was near zero. Neither extreme is useful.
    """
    city_poi = poi[poi.destination_id == code]
    if city_poi.empty or not hubs:
        return
    near = sum(
        any(haversine_km(r.lat, r.lon, h["lat"], h["lon"]) <= HUB_RADIUS_KM
            for h in hubs)
        for r in city_poi.itertuples())
    pct = 100 * near / len(city_poi)
    print(f"    {len(city_poi)} POI'nin {near}'i bir hub'a {HUB_RADIUS_KM} km "
          f"icinde = %{pct:.0f}"
          + ("   <-- iliski ayirt etmiyor" if pct >= 90 else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cities", nargs="*", default=list(CITIES))
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    try:
        poi = pd.read_csv(DATA / "poi.csv")
    except FileNotFoundError:
        poi = pd.DataFrame(columns=["destination_id", "lat", "lon"])

    rows, n = [], 0
    for code in [c.upper() for c in args.cities]:
        if code not in CITIES:
            sys.exit(f"bilinmeyen sehir: {code}")
        city = CITIES[code]
        hubs = select(add_sitelinks(fetch_candidates(city)))
        print(f"\n===== {city['city']} ({code}): {len(hubs)} gecis noktasi =====")
        for h in hubs:
            n += 1
            print(f"  {h['type']:<14} sl{h['sitelinks']:>3} {h['iata']:<4} "
                  f"{h['name'][:40]:<42} {h['km']:>5.1f} km")
            rows.append({"transport_id": f"TR{n:03d}", "destination_id": code,
                         "name": h["name"], "type": h["type"],
                         "lat": h["lat"], "lon": h["lon"]})
        report_reach(code, hubs, poi)

    df = pd.DataFrame(rows)[["transport_id", "destination_id", "name", "type",
                             "lat", "lon"]]
    print(f"\ntoplam {len(df)} satir: {df.type.value_counts().to_dict()}")
    if args.write:
        out = DATA / "transport.csv"
        df.to_csv(out, index=False)
        print(f"-> {out}")


if __name__ == "__main__":
    main()
