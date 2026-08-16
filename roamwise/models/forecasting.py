"""
Demand forecasting ("The Tools" -> Demand Forecasting in the proposal).

The proposal suggests Prophet or LSTM. We use statsmodels' Holt-Winters
(triple exponential smoothing) instead: it needs no compiled Stan backend,
trains in milliseconds on 90 monthly points per city, and captures trend +
seasonality just as well at this data volume -- a deliberate engineering
trade-off documented in the final report rather than an oversight.

Output of forecast_city() is consumed by the ForecasterAgent to label future
months as low / medium / high crowding relative to that city's own history,
which is what the Router/itinerary layer actually needs (not raw visitor
counts).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

HERE = Path(__file__).parent
DATA_DIR = HERE.parent / "data"


def load_demand() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "demand_timeseries.csv", parse_dates=["date"])
    return df


def forecast_city(destination_id: str, horizon_months: int = 12, df: pd.DataFrame = None) -> pd.DataFrame:
    """Fit Holt-Winters on a city's post-pandemic-recovery history (from 2022-07
    onward, so the COVID shock doesn't distort the learned seasonality) and
    forecast `horizon_months` ahead."""
    if df is None:
        df = load_demand()
    series = df[df.destination_id == destination_id].sort_values("date")
    series = series[series.date >= "2022-07-01"].set_index("date")["visitors"]
    series = series.asfreq("MS")

    model = ExponentialSmoothing(
        series, trend="add", seasonal="add", seasonal_periods=12, damped_trend=True
    ).fit(optimized=True)
    forecast = model.forecast(horizon_months)

    hist_mean, hist_std = series.mean(), series.std()
    out = pd.DataFrame({"date": forecast.index, "destination_id": destination_id, "forecast_visitors": forecast.values})
    out["crowding_z"] = (out.forecast_visitors - hist_mean) / hist_std
    out["crowding_level"] = pd.cut(
        out.crowding_z, bins=[-np.inf, -0.4, 0.4, np.inf], labels=["low", "medium", "high"]
    )
    return out.reset_index(drop=True)


def forecast_all_cities(horizon_months: int = 12) -> pd.DataFrame:
    df = load_demand()
    dest_ids = df.destination_id.unique()
    frames = [forecast_city(d, horizon_months, df) for d in dest_ids]
    return pd.concat(frames, ignore_index=True)


def best_months_to_visit(destination_id: str, top_k: int = 3, horizon_months: int = 12) -> list[dict]:
    """Rank the forecast horizon by lowest predicted crowding -- directly answers
    the proposal's 'recommend destinations that avoid peak crowding' goal."""
    fc = forecast_city(destination_id, horizon_months)
    fc = fc.sort_values("crowding_z")
    return [
        {"month": row.date.strftime("%Y-%m"), "crowding_level": str(row.crowding_level),
         "forecast_visitors": int(row.forecast_visitors)}
        for row in fc.head(top_k).itertuples()
    ]


if __name__ == "__main__":
    df = load_demand()
    for dest_id in df.destination_id.unique():
        fc = forecast_city(dest_id)
        print(f"{dest_id}: next-12mo forecast mean={int(fc.forecast_visitors.mean()):,}  "
              f"low-crowd months={best_months_to_visit(dest_id)}")
