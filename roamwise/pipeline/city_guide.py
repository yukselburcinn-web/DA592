"""Derive each city guide from the catalogue instead of writing it by hand.

The guide is retrieval material: `retrieval/corpus.py` loads it alongside the
per-POI descriptions and searches both. What it has to add is the city-level
structure a per-POI description cannot -- which areas cluster together, how far
apart they sit, where the food and nightlife are, what it costs. All of that is
computable from the catalogue, which means the text is true by construction and
every place it names is one the planner can actually route to.

That last point is not hypothetical. The hand-written PAR.txt in the repo
recommends the Champs-Elysees, which is not in the catalogue at all, so
retrieval could surface a guide pointing at a stop the optimiser cannot use.

Zones come from KMeans on POI coordinates -- the same technique POIZoner
applies downstream -- so the guide describes the city the way the day planner
will see it.

    python city_guide.py PAR BER --write
"""
import argparse
import sys

import pandas as pd
from sklearn.cluster import KMeans

from common import CITIES, DATA, haversine_km

N_ZONES = 4
WALKABLE_KM = 1.6      # a radius people cross on foot without thinking about it

# The one part that is NOT derived from the catalogue, and is marked as such so
# nobody later mistakes it for a measurement. Everything above comes out of
# poi.csv; these two sentences are editorial knowledge the data has no
# equivalent for -- closing days, opening seasons, when an evening starts.
#
# They are deliberately scheduling constraints rather than colour. A day
# planner can act on "the Louvre is shut on Tuesdays"; it can do nothing with
# "Paris is romantic". Every place named here must exist in the catalogue --
# check_grounding() reports what it found, so a rename downstream shows up.
#
# TODO: once an LLM API key is available (agents/llm_client.py defaults to the
# offline TemplateLLMClient today), evaluate generating these from the
# catalogue plus a retrieved source rather than hand-writing them. Two
# conditions before trusting it: the output must name only catalogue POIs, and
# every scheduling claim must be checkable against a cited source -- this is
# exactly the kind of confident, plausible, wrong sentence that cost four
# rounds of correction in the derived text above.
EDITORIAL = {
    "PAR": "Note the weekly closures when sequencing indoor stops: the Louvre "
           "Museum shuts on Tuesdays and the Musée d'Orsay on Mondays, and much "
           "of the independent food scene closes for August.",
    "BER": "Shops across Germany stay shut on Sundays, the Museum Island houses "
           "close on Mondays, and clubs here do not fill until well after "
           "midnight, so an evening planned around 21:00 will run early.",
}


def phrase_list(items):
    items = list(items)
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def zone_summary(df, city, n_zones=N_ZONES, require_open=False):
    """Cluster the catalogue and describe each cluster by a POI inside it.

    The anchor is the best-known POI *near the cluster's centre*, not the
    best-known one anywhere in it. Those differ badly: picking the most famous
    outright made the Louvre the anchor of a cluster whose centroid is 3 km
    away, and the guide then described the middle of Paris as "further out".

    Radius is the 80th percentile distance from the centroid, not the maximum,
    which a single outlying stop would otherwise inflate to half the city --
    that is what produced "the area around the Eiffel Tower, within 10.6 km".
    """
    coords = df[["lat", "lon"]].to_numpy()
    n = max(1, min(n_zones, len(df)))
    labels = KMeans(n_clusters=n, n_init=10, random_state=0).fit_predict(coords)
    df = df.assign(zone=labels)

    zones = []
    for _, g in df.groupby("zone"):
        clat, clon = g.lat.mean(), g.lon.mean()
        d_centroid = g.apply(lambda r: haversine_km(clat, clon, r.lat, r.lon), axis=1)
        radius = float(d_centroid.quantile(0.8))

        inner = g[d_centroid <= max(radius, 0.4)]
        pool = inner if len(inner) else g
        if require_open:
            # Sitelink fame favours venues that are famous in history rather
            # than open tonight: it picked the Cafe de la Regence, shut since
            # the 19th century, and the Grossgaststatte Ahornblatt, demolished
            # in 2000. Live opening hours in OSM are the cheapest available
            # evidence that a place is still trading.
            live = pool[pool.hours_source == "osm"]
            if len(live):
                pool = live
        anchor = pool.nlargest(1, "sitelink_count").iloc[0]

        zones.append({
            "anchor": anchor["name"],
            "clat": float(clat), "clon": float(clon),
            "n": len(g),
            "radius": radius,
            "anchor_km": haversine_km(city["lat"], city["lon"],
                                      anchor["lat"], anchor["lon"]),
            "fame": int(anchor["sitelink_count"]),
        })
    return sorted(zones, key=lambda z: z["anchor_km"])


def observed_closing_hour(df):
    """Median closing hour across stops whose hours actually came from OSM.

    Rows sitting on `category_default` carry the value this pipeline put there,
    so averaging them in would report our own fallback back to us as a finding
    about the city. An earlier draft did exactly that and confidently announced
    that museums in both Paris and Berlin close at 18:00 -- which is simply
    CATEGORY_DEFAULTS["museum"].
    """
    obs = df[df.hours_source == "osm"]
    if len(obs) < 10:
        return None, 0
    return int(obs.close_hour.median()), len(obs)


def build_guide(code, df):
    city = CITIES[code]
    zones = zone_summary(df, city)
    parts = []

    core = zones[0]
    parts.append(
        f"In {city['city']}, {core['n']} of this catalogue's {len(df)} stops sit "
        f"in the central cluster around {core['anchor']}, most of them within "
        f"{core['radius']:.1f} km of its centre"
        + (" and comfortably walkable end to end."
           if core["radius"] <= WALKABLE_KM
           else " -- close enough to chain on foot in stretches, but expect a "
                "transit hop to cross the whole cluster.")
    )

    outer = [z for z in zones[1:] if z["anchor_km"] > core["anchor_km"] + 1.0]
    if outer:
        parts.append(
            f"{phrase_list(z['anchor'] for z in outer)} "
            f"{'anchor' if len(outer) > 1 else 'anchors'} clusters "
            f"{min(z['anchor_km'] for z in outer):.0f}-"
            f"{max(z['anchor_km'] for z in outer):.0f} km out, each better "
            f"treated as its own half-day than as a detour."
        )

    night = df[df.category.isin(["nightlife", "food"])]
    if len(night) >= 5:
        # Name a landmark to locate the area *by*, not a venue to eat at.
        # Sitelink fame cannot pick a good restaurant: it ranks whatever has a
        # Wikipedia article, which means historical venues that have long since
        # closed (the Cafe de la Regence) or, once those are filtered out by
        # live opening hours, national chains (BLOCK HOUSE). Wikidata knows
        # chains and monuments; it does not know the good place round the
        # corner. A landmark is something it does know, and it answers the
        # question the guide is actually for -- which part of town to head to.
        #
        # Only claim a concentration if there is one. The mean position of a
        # dispersed set lands near the middle by construction, so "eating and
        # nightlife cluster near the centre" was true of the arithmetic rather
        # than of the city. Require a genuine hot spot: a KMeans cluster
        # holding at least a third of these stops inside a walkable radius.
        nz = zone_summary(night, city, n_zones=3)
        hot = [z for z in nz
               if z["n"] >= 0.33 * len(night) and z["radius"] <= WALKABLE_KM]
        if hot:
            h = max(hot, key=lambda z: z["n"])
            # Density proves the hot spot; a landmark names it. Naming it by
            # its own best-known *venue* fails both ways -- sitelink fame picks
            # long-closed places (the Cafe de la Regence), and filtering those
            # out by live opening hours picks national chains (BLOCK HOUSE).
            # Wikidata knows landmarks; it does not know where to have dinner.
            d = df.apply(
                lambda r: haversine_km(h["clat"], h["clon"], r.lat, r.lon), axis=1)
            near = df[d <= 1.0]
            ref = (near if len(near) else df).nlargest(1, "sitelink_count").iloc[0]
            if ref["name"] == core["anchor"]:
                # The hot spot sits inside the central cluster we already named,
                # so pointing at the same landmark twice adds nothing. Report
                # the share instead, which is the part that is actually news.
                parts.append(
                    f"{h['n']} of the {len(night)} food and nightlife stops fall "
                    f"inside that same central cluster, so evenings need no "
                    f"separate trip."
                )
            else:
                parts.append(
                    f"Eating and nightlife are densest in the streets around "
                    f"{ref['name']}, {h['anchor_km']:.1f} km from the centre, which "
                    f"holds {h['n']} of the {len(night)} food and nightlife stops here."
                )

    free = (df.price_level == 0).mean()
    top_cats = df.category.value_counts().head(3).index.tolist()
    parts.append(
        f"The catalogue leans {phrase_list(top_cats)}, and about {free:.0%} of it "
        f"is free to enter."
    )

    close, n_obs = observed_closing_hour(df)
    if close is not None:
        parts.append(
            f"Published closing times are known for {n_obs} of these stops and "
            f"run to about {close}:00, so the indoor half of a day is best taken "
            f"before then."
        )

    if code in EDITORIAL:
        parts.append(EDITORIAL[code])

    return " ".join(parts)


def check_grounding(text, df):
    """Every catalogue place the guide names. The guide is generated from these
    rows, so this should never be empty -- it is a regression check on the
    templates, not a filter."""
    return sorted(n for n in set(df.name) if n in text)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cities", nargs="*", default=list(CITIES))
    ap.add_argument("--write", action="store_true",
                    help="../data/city_guides/<CODE>.txt dosyalarina yaz")
    args = ap.parse_args()

    poi = pd.read_csv(DATA / "poi.csv")
    out_dir = DATA / "city_guides"
    if args.write:
        out_dir.mkdir(exist_ok=True)

    for code in [c.upper() for c in args.cities]:
        if code not in CITIES:
            sys.exit(f"bilinmeyen sehir: {code}")
        df = poi[poi.destination_id == code].copy()
        text = build_guide(code, df)
        named = check_grounding(text, df)
        print(f"\n===== {code} ({len(text.split())} kelime) =====")
        print(text)
        print(f"\n  katalogda dogrulanan yer adi: {len(named)} -> {named}")
        if args.write:
            (out_dir / f"{code}.txt").write_text(text + "\n")
            print(f"  -> {(out_dir / (code + '.txt'))}")


if __name__ == "__main__":
    main()
