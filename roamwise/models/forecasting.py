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

Two anchors live in this module and they are not the same point.
forecast_city() counts its horizon from the end of the city's *history*, which
is what a forecast model does. Everything a traveler is shown has to be
counted from *today* instead, and the gap between the two is a per-city
number: the sources lag by different amounts, so Berlin's series stops in
2024-12 while Paris's runs to 2025-12. forecast_window() is the conversion,
and #161 is what happens without it -- twelve bars titled "next 12 months",
all twelve of them already in the past.
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


# How far past the end of a city's history anything here will extrapolate.
#
# Holt-Winters projects the learned trend and seasonality forward with no new
# information, so a point three years out is a seasonal shape rather than a
# measurement, and saying so with a ceiling is more honest than quietly
# producing one. Lived in forecaster_agent.py until #161 needed the same
# ceiling for the chart; the agent imports it from here now, so the UI and the
# agent cannot drift onto two different limits.
MAX_HORIZON_MONTHS = 36


def current_month(today=None) -> pd.Period:
    """The month we are actually in.

    `today` is a parameter rather than a read of the clock inside each caller
    so that a test can pin it. #161 was a window anchored to the data instead
    of to today, and a test for it that reads the real clock only holds on the
    day it was written -- the bug is invisible on any date close enough to the
    end of the series.
    """
    return pd.Period(pd.Timestamp.today() if today is None else pd.Timestamp(today), freq="M")


def history_end(destination_id: str, df: pd.DataFrame = None) -> pd.Period:
    """Last month this city has observed data for, or None if it has none."""
    if df is None:
        df = load_demand()
    dates = df.loc[df.destination_id == destination_id, "date"]
    return pd.Period(dates.max(), freq="M") if len(dates) else None


def horizon_to_cover(destination_id: str, last_month, df: pd.DataFrame = None) -> int:
    """How many months forecast_city() must run to reach `last_month`.

    The conversion between the two anchors. Every caller that means "months
    from now" goes through here instead of passing 12 to forecast_city() and
    hoping the city's history ends near today (#161).

    Returns 0 for a city with no history at all -- there is nothing to fit, so
    there is no horizon to ask for.
    """
    end = history_end(destination_id, df)
    if end is None:
        return 0
    wanted = (pd.Period(last_month, freq="M") - end).n
    return max(1, min(wanted, MAX_HORIZON_MONTHS))


def forecast_window(destination_id: str, months: int = 12, df: pd.DataFrame = None,
                    today=None, include_month: str = None) -> pd.DataFrame:
    """`months` of forecast starting at the current month (#161).

    forecast_city(horizon_months=12) is twelve months from the end of the
    series, which for Berlin was 2025-01 .. 2025-12 -- rendered under the
    title "next 12 months" on 2026-08-30. The months between the end of the
    history and today still have to be forecast (Holt-Winters only projects
    forward from where it was fitted), they are just not shown.

    `include_month` extends the far end so a month the traveler has already
    chosen is inside the window even when it falls past `months`. The chart
    and the ForecasterAgent otherwise answer about two different months, which
    is the second half of #161: the narrative said "September is high demand"
    from a row the chart did not contain.
    """
    if df is None:
        df = load_demand()
    start = current_month(today)
    last = start + (months - 1)
    if include_month is not None:
        last = max(last, pd.Period(include_month, freq="M"))

    fc = forecast_city(destination_id, horizon_months=horizon_to_cover(destination_id, last, df), df=df)
    window = fc[(fc.date >= start.to_timestamp()) & (fc.date <= last.to_timestamp())]
    # MAX_HORIZON_MONTHS can stop the forecast short of the current month for a
    # city whose history ends more than three years back. Show the furthest the
    # model will go rather than an empty chart -- the caption built from
    # history_end() is what tells the reader how far behind that is.
    return (window if len(window) else fc.tail(months)).reset_index(drop=True)


def forecast_all_cities(horizon_months: int = 12) -> pd.DataFrame:
    df = load_demand()
    dest_ids = df.destination_id.unique()
    frames = [forecast_city(d, horizon_months, df) for d in dest_ids]
    return pd.concat(frames, ignore_index=True)


def best_months_to_visit(destination_id: str, top_k: int = 3, horizon_months: int = 12) -> list[dict]:
    """Rank the next `horizon_months` months by lowest predicted crowding --
    directly answers the proposal's 'recommend destinations that avoid peak
    crowding' goal.

    Counted from today, not from the end of the series: a month that has
    already been and gone is not a recommendation a traveler can act on, and
    for Berlin every month this returned was one (#161).
    """
    fc = forecast_window(destination_id, months=horizon_months)
    fc = fc.sort_values("crowding_z")
    return [
        {"month": row.date.strftime("%Y-%m"), "crowding_level": str(row.crowding_level),
         "forecast_visitors": int(row.forecast_visitors)}
        for row in fc.head(top_k).itertuples()
    ]


if __name__ == "__main__":
    df = load_demand()
    for dest_id in df.destination_id.unique():
        fc = forecast_window(dest_id)
        print(f"{dest_id}: {fc.date.min():%Y-%m}..{fc.date.max():%Y-%m} "
              f"forecast mean={int(fc.forecast_visitors.mean()):,}  "
              f"low-crowd months={best_months_to_visit(dest_id)}")
