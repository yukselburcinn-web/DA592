"""
MAE/RMSE backtest of forecasting.py's default Holt-Winters model against the
optional Prophet alternative in models/forecasting_prophet.py (issue #3).

For each city: hold out the last HOLDOUT_MONTHS of its real Eurostat demand
series, fit both models on everything before that, forecast HOLDOUT_MONTHS
ahead, and score both forecasts against the real held-out values. This
requires `prophet` (see requirements-prophet.txt); it is not part of the
default requirements.txt for the same reason forecasting_prophet.py isn't
used by default -- see that module's docstring.

Run with:
    pip install -r requirements-prophet.txt
    python evaluation/forecasting_comparison.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from roamwise.models.forecasting import load_demand
from roamwise.models.forecasting_prophet import fit_prophet

HERE = Path(__file__).parent
HOLDOUT_MONTHS = 6
TRAIN_START = "2022-07-01"  # same post-pandemic-recovery cutoff forecasting.py uses


def _city_series(df: pd.DataFrame, destination_id: str) -> pd.Series:
    series = df[df.destination_id == destination_id].sort_values("date")
    series = series[series.date >= TRAIN_START].set_index("date")["visitors"]
    return series.asfreq("MS")


def _mae_rmse(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    error = actual - predicted
    return float(np.mean(np.abs(error))), float(np.sqrt(np.mean(error ** 2)))


def _forecast_holt_winters(train: pd.Series, horizon: int) -> pd.Series:
    model = ExponentialSmoothing(
        train, trend="add", seasonal="add", seasonal_periods=12, damped_trend=True
    ).fit(optimized=True)
    return model.forecast(horizon)


def run_comparison(holdout_months: int = HOLDOUT_MONTHS) -> pd.DataFrame:
    df = load_demand()
    rows = []
    for destination_id in df.destination_id.unique():
        series = _city_series(df, destination_id)
        train, test = series.iloc[:-holdout_months], series.iloc[-holdout_months:]

        hw_forecast = _forecast_holt_winters(train, holdout_months)
        hw_mae, hw_rmse = _mae_rmse(test.values, hw_forecast.values)
        rows.append({"destination_id": destination_id, "model": "holt_winters", "mae": round(hw_mae, 1), "rmse": round(hw_rmse, 1)})

        prophet_forecast = fit_prophet(train, holdout_months)
        p_mae, p_rmse = _mae_rmse(test.values, prophet_forecast.values)
        rows.append({"destination_id": destination_id, "model": "prophet", "mae": round(p_mae, 1), "rmse": round(p_rmse, 1)})

        print(f"[{destination_id}] Holt-Winters MAE={hw_mae:,.0f} RMSE={hw_rmse:,.0f}  |  Prophet MAE={p_mae:,.0f} RMSE={p_rmse:,.0f}")

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("model")[["mae", "rmse"]].mean().round(1)


if __name__ == "__main__":
    results = run_comparison()
    results.to_csv(HERE / "forecasting_comparison_results.csv", index=False)
    summary = summarize(results)
    summary.to_csv(HERE / "forecasting_comparison_summary.csv")
    print("\n" + summary.to_string())
