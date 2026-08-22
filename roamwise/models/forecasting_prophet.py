"""
Prophet-based alternative to `forecasting.py`'s Holt-Winters model (issue #3
-- the proposal names Prophet explicitly; `forecasting.py`'s docstring
explains why Holt-Winters was chosen as the default instead).

Optional dependency: not part of `requirements.txt` (pulls in `cmdstanpy`,
which compiles/runs a Stan model per fit -- seconds per city rather than
Holt-Winters' milliseconds). Install with:

    pip install -r requirements-prophet.txt

Exposes the same `forecast_city()` output shape as `forecasting.py` (date,
destination_id, forecast_visitors, crowding_z, crowding_level) so it's a
drop-in alternative, plus `fit_prophet()`, a lower-level train/forecast
helper reused by `evaluation/forecasting_comparison.py` for the MAE/RMSE
backtest against Holt-Winters.
"""
import logging

import numpy as np
import pandas as pd

from roamwise.models.forecasting import load_demand

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)


def fit_prophet(series: pd.Series, horizon_months: int) -> pd.Series:
    """series: a monthly-frequency pd.Series indexed by date. Returns the
    next horizon_months of forecast values, indexed by date."""
    from prophet import Prophet

    train_df = pd.DataFrame({"ds": series.index, "y": series.values})
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(train_df)

    future = model.make_future_dataframe(periods=horizon_months, freq="MS")
    forecast = model.predict(future)
    tail = forecast.set_index("ds")["yhat"].iloc[-horizon_months:]
    tail.index.name = "date"
    return tail


def forecast_city(destination_id: str, horizon_months: int = 12, df: pd.DataFrame = None) -> pd.DataFrame:
    """Same post-pandemic-recovery cutoff and output shape as
    forecasting.forecast_city(), so the two are directly comparable."""
    if df is None:
        df = load_demand()
    series = df[df.destination_id == destination_id].sort_values("date")
    series = series[series.date >= "2022-07-01"].set_index("date")["visitors"]
    series = series.asfreq("MS")

    forecast = fit_prophet(series, horizon_months)

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


if __name__ == "__main__":
    df = load_demand()
    for dest_id in df.destination_id.unique():
        fc = forecast_city(dest_id, df=df)
        print(f"{dest_id}: next-12mo forecast mean={int(fc.forecast_visitors.mean()):,}")
