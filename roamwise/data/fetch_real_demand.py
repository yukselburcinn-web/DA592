"""
SUPERSEDED by `pipeline/build_demand.py`. Do not run to set the project up.

Two things are wrong with it now. It writes the old EIGHT-city series (IST,
PAR, ROM, BCN, AMS, PRG, VIE, LIS, with no BER at all), and it reverts the
measure to the **country-level** `tour_occ_nim`. The shipped series is
`tour_occ_nin2m` at NUTS 2 granularity, and that difference is not cosmetic:
the forecaster consumes seasonal *shape*, and France peaks in August at 2.35x
its annual mean while Ile-de-France sits at 1.06x -- the national proxy
changed the crowding label in 8 of 12 Paris months.

`demand_timeseries.csv` is committed, so setup needs none of this (see the
README's Quickstart). To rebuild it: `cd pipeline && python build_demand.py
--write`. Running this script instead also makes `forecast_city` fail on
"less than two full seasonal cycles" against the two-city destination list.

Kept for history. The docstring below describes the state of the world before
`tour_occ_nin2m` was found, and its central claim -- that Eurostat publishes
this only at country level -- is no longer true.

Original description follows.

Replace the procedurally-generated `demand_timeseries.csv` with real monthly
tourism-demand data from Eurostat's short-term tourism statistics
(`tour_occ_nim`: nights spent by non-resident tourists in accommodation
establishments, NACE I551-I553).

Run after generate_data.py / fetch_real_pois.py:

    python fetch_real_demand.py

Eurostat only publishes this series at country level (there is no free,
unauthenticated, monthly, per-city tourism-arrivals API covering all 8
destination cities -- a city-level series exists in Eurostat's Urban Audit
`urb_ctour` dataset, but only annually, which can't feed the Holt-Winters
monthly-seasonality forecasting model in `models/forecasting.py` unchanged).
So each destination_id uses its *country's* real monthly series as a proxy --
documented here and in the README as a known simplification, not a per-city
number. The series itself (values, trend, COVID-era dip, seasonality) is
100% real Eurostat data, not synthetic.
"""
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).parent
EUROSTAT_URL = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/tour_occ_nim"
HEADERS = {"User-Agent": "RoamWise-DataFetch/1.0 (student term project; contact via GitHub repo)"}

START_PERIOD = "2019-01"

# destination_id -> Eurostat country code. Same country covers every
# destination in this dataset today (one city per country), so this is
# also destination_id -> proxy country.
DESTINATION_COUNTRY = {
    "IST": "TR",
    "PAR": "FR",
    "ROM": "IT",
    "BCN": "ES",
    "AMS": "NL",
    "PRG": "CZ",
    "VIE": "AT",
    "LIS": "PT",
}


def fetch_country_series(geo: str) -> pd.Series:
    """Monthly nights spent by non-resident tourists, geo country, from
    START_PERIOD to the latest available month."""
    params = {
        "format": "JSON",
        "lang": "EN",
        "geo": geo,
        "unit": "NR",
        "c_resid": "FOR",
        "nace_r2": "I551-I553",
        "sinceTimePeriod": START_PERIOD,
    }
    resp = requests.get(EUROSTAT_URL, params=params, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    time_index = data["dimension"]["time"]["category"]["index"]
    idx_to_period = {v: k for k, v in time_index.items()}
    values = {int(k): v for k, v in data["value"].items()}

    series = pd.Series(
        {idx_to_period[idx]: val for idx, val in values.items()}
    ).sort_index()
    series.index = pd.PeriodIndex(series.index, freq="M").to_timestamp()
    return series


def main():
    rows = []
    for dest_id, geo in DESTINATION_COUNTRY.items():
        print(f"[{dest_id}] fetching Eurostat tour_occ_nim for {geo}...")
        series = fetch_country_series(geo)
        for date, visitors in series.items():
            rows.append({"destination_id": dest_id, "date": date.strftime("%Y-%m-%d"), "visitors": int(visitors)})
        print(f"[{dest_id}] {len(series)} months, {series.index.min().date()} .. {series.index.max().date()}")

    demand_df = pd.DataFrame(rows).sort_values(["destination_id", "date"])
    demand_df.to_csv(HERE / "demand_timeseries.csv", index=False)
    print(f"\nWrote demand_timeseries.csv ({len(demand_df)} rows)")


if __name__ == "__main__":
    main()
