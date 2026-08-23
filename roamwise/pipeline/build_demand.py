"""Build demand_timeseries.csv from Eurostat's regional monthly tourism series.

The repo's fetch_real_demand.py uses `tour_occ_nim`, which is national, and
says so honestly: each city stands in for its whole country because "Eurostat
only publishes this series at country level". That is no longer true.
`tour_occ_nin2m` -- "Nights spent at tourist accommodation establishments by
month and NUTS 2 region" -- carries the same measure per region from 2020.

That matters more than it sounds. Berlin *is* a NUTS 2 region (DE30), so its
series is the city itself rather than Germany. Paris maps to Ile-de-France
(FR10), which is wider than the city but far tighter than France, and the
seasonal shape is what the forecaster actually consumes: a city-break
destination and a country with alpine and coastal summer peaks do not share a
seasonal curve, so the national proxy was teaching the model the wrong shape.

Two things to know about the format:

  * The series is not indexed by month. `time` holds the year and a separate
    `month` dimension holds M01..M12 plus a TOTAL that has to be dropped, or
    every year gains a thirteenth month worth the sum of the other twelve.
  * Coverage is uneven. Ile-de-France runs to 2025-12; Berlin stops at
    2024-12, because Germany's regional returns lag. The 2025 periods exist in
    the dimension but carry no values, so a naive fetch looks complete.

    python build_demand.py --write
"""
import argparse
import urllib.parse

import pandas as pd

from common import DATA, http

EUROSTAT = ("https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/"
            "data/tour_occ_nin2m")

# destination_id -> NUTS 2 region. Berlin is a region in its own right; Paris
# is not, and Ile-de-France is the closest published unit.
DESTINATION_REGION = {
    "PAR": ("FR10", "Île-de-France"),
    "BER": ("DE30", "Berlin"),
}


def fetch_region(geo):
    """Monthly nights spent, all visitors, one NUTS 2 region."""
    params = {"format": "JSON", "lang": "EN", "geo": geo, "unit": "NR",
              "c_resid": "TOTAL", "nace_r2": "I551-I553"}
    d = http(f"{EUROSTAT}?{urllib.parse.urlencode(params)}",
             cache_key=f"eurostat-{geo}", timeout=90)

    dims = d["id"]
    sizes = [len(d["dimension"][x]["category"]["index"]) for x in dims]
    month_idx = {v: k for k, v in d["dimension"]["month"]["category"]["index"].items()}
    time_idx = {v: k for k, v in d["dimension"]["time"]["category"]["index"].items()}
    m_pos, t_pos = dims.index("month"), dims.index("time")

    out = {}
    for flat, value in d["value"].items():
        # JSON-stat flattens the dimension grid into one integer per cell.
        rem, coord = int(flat), []
        for size in reversed(sizes):
            coord.append(rem % size)
            rem //= size
        coord.reverse()
        month = month_idx[coord[m_pos]]
        if month == "TOTAL":          # an annual sum sitting in the month axis
            continue
        out[f"{time_idx[coord[t_pos]]}-{month[1:]}-01"] = int(value)
    return pd.Series(out).sort_index()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    rows = []
    for code, (geo, label) in DESTINATION_REGION.items():
        s = fetch_region(geo)
        print(f"{code} <- {geo} ({label}): {len(s)} ay, "
              f"{s.index.min()[:7]} .. {s.index.max()[:7]}")
        # forecast_city() trains on 2022-07 onward and needs two full seasons.
        usable = s[s.index >= "2022-07-01"]
        print(f"    forecast_city penceresi (2022-07+): {len(usable)} ay"
              f"{'  ⚠ 24 aydan az' if len(usable) < 24 else ''}")
        for date, visitors in s.items():
            rows.append({"destination_id": code, "date": date, "visitors": visitors})

    df = pd.DataFrame(rows).sort_values(["destination_id", "date"])
    print(f"\ntoplam {len(df)} satir")
    if args.write:
        out = DATA / "demand_timeseries.csv"
        df.to_csv(out, index=False)
        print(f"-> {out}")


if __name__ == "__main__":
    main()
