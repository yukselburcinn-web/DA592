"""The two "tools" the proposal names, on their own: the Holt-Winters demand
model that flags crowding, and the KMeans segmenter that turns six sliders
into an archetype.

Split out of `test_retrieval.py` in #164. They were folded in there by #150 and
they never belonged: they test `models/`, and leaving them beside the retrieval
tests put two people in one file.
"""

import pandas as pd
import pytest

from roamwise.agents.forecaster_agent import ForecasterAgent
from roamwise.models.forecasting import (
    best_months_to_visit, current_month, forecast_city, forecast_window, history_end,
)
from roamwise.models.segmentation import TravelerSegmenter
from roamwise.tests.helpers import FULL_CITIES, needs_full_city


@needs_full_city
def test_forecasting_produces_crowding_labels():
    fc = forecast_city(FULL_CITIES[0])
    assert len(fc) == 12
    assert set(fc.crowding_level.astype(str)) <= {"low", "medium", "high"}
    months = best_months_to_visit(FULL_CITIES[0], top_k=3)
    assert len(months) == 3


# --- issue #161: the window a traveler is shown starts today, not where the
# data happens to stop ---
#
# forecast_city() counts its horizon from the end of the city's history, which
# is correct for a forecast model and wrong for everything a traveler reads.
# Berlin's series ends 2024-12, so the demand chart's "next 12 months" was
# 2025-01 .. 2025-12 -- twelve months that had already happened, on a page
# being used to plan 2026. Paris's series ends a year later, so the same code
# gave the two cities two different "futures".
#
# None of the tests below name a date. Each one either derives its clock from
# the city's own history or asserts that two parts of the app agree with each
# other, so they keep holding as the calendar moves and as the CSVs are
# refreshed. A test pinned to 2026 would go quiet exactly when the series is
# updated and the bug becomes invisible again.


@needs_full_city
@pytest.mark.parametrize("months_behind", [1, 8, 20, 25])
def test_the_forecast_window_starts_at_the_current_month(months_behind):
    """However far behind the series ends, the window opens on today's month."""
    city = FULL_CITIES[0]
    today = (history_end(city) + months_behind).to_timestamp()

    window = forecast_window(city, months=12, today=today)

    assert len(window) == 12
    assert pd.Period(window.date.min(), freq="M") == pd.Period(today, freq="M")
    assert set(window.crowding_level.astype(str)) <= {"low", "medium", "high"}


@needs_full_city
def test_every_month_shown_is_still_ahead_of_the_traveler():
    """The failure as a traveler met it: bars for months already gone."""
    city = FULL_CITIES[0]
    window = forecast_window(city, months=12)
    assert pd.Period(window.date.min(), freq="M") >= current_month()


@pytest.mark.skipif(len(FULL_CITIES) < 2, reason="tek sehirde iki pencere karsilastirilamaz")
def test_two_cities_whose_data_ends_at_different_times_show_the_same_window():
    """The window is a property of the calendar, not of the source's lag.

    This is the comparison that surfaced #161 -- the two cities' charts were
    both labelled "next 12 months" and covered different years, because their
    sources publish with different delays.
    """
    a, b = FULL_CITIES[0], FULL_CITIES[1]
    assert history_end(a) != history_end(b), "iki sehrin serisi ayni ayda bitiyor"

    span = lambda c: (forecast_window(c, months=12).date.min(),
                      forecast_window(c, months=12).date.max())
    assert span(a) == span(b)


@needs_full_city
def test_the_window_holds_the_month_the_forecaster_read():
    """Chart and narrative answer about the same month.

    The crowding level under the chart comes from ForecasterAgent, which
    extends its own horizon to reach the month asked about. Before #161 the
    chart did not, so the two sat side by side describing different months --
    and only the narrative's was the one the traveler had picked.
    """
    city = FULL_CITIES[0]
    # A month past the default 12-wide window, so the chart has to stretch for
    # it rather than happening to contain it.
    travel_month = (current_month() + 15).strftime("%Y-%m")

    forecast = ForecasterAgent().run(city, travel_month=travel_month, narrate=False)
    window = forecast_window(city, months=12, include_month=forecast["target_month"])

    row = window[window.date.dt.strftime("%Y-%m") == forecast["target_month"]]
    assert len(row) == 1, "forecaster'in okudugu ay grafikte yok"
    assert str(row.iloc[0].crowding_level) == forecast["crowding_level"]


@needs_full_city
def test_the_alternatives_offered_are_months_the_chart_shows():
    """"If your dates are flexible, try X" has to point inside the window.

    The agent's horizon used to stop at the month it was asked about, so for a
    city whose history lags it had nothing but the gap between today and the
    trip to choose from: Berlin offered "August 2026 (high), September 2026
    (high)" for a September 2026 trip -- two months, both of them the trip,
    neither of them low-crowding (#161).
    """
    city = FULL_CITIES[0]
    travel_month = (current_month() + 1).strftime("%Y-%m")

    forecast = ForecasterAgent().run(city, travel_month=travel_month, narrate=False)
    shown = set(forecast_window(city, months=12).date.dt.strftime("%Y-%m"))

    assert len(forecast["low_crowd_alternatives"]) == 3
    assert {a["month"] for a in forecast["low_crowd_alternatives"]} <= shown


@needs_full_city
def test_best_months_to_visit_only_offers_bookable_months():
    """A recommendation the traveler cannot act on is not a recommendation."""
    months = best_months_to_visit(FULL_CITIES[0], top_k=3)
    assert months
    assert all(pd.Period(m["month"], freq="M") >= current_month() for m in months)


def test_traveler_segmentation_classifies():
    seg = TravelerSegmenter()
    result = seg.classify({"budget": 0.1, "culture": 0.2, "nature": 0.9, "nightlife": 0.1, "relax": 0.3, "adventure": 0.9})
    assert result["archetype"] == "Nature & Adventure"


def test_preference_levels_produce_distinct_archetypes():
    """Issue #23: the sidebar exposes Low/Medium/High instead of raw 0-1
    sliders, so the three tiers must actually separate archetype centroids --
    not just satisfy TravelerSegmenter.classify()'s float contract."""
    from roamwise.views.itinerary import PREFERENCE_LEVELS
    from roamwise.models.segmentation import FEATURES

    seg = TravelerSegmenter()
    archetypes = {
        level: seg.classify({f: value for f in FEATURES})["archetype"]
        for level, value in PREFERENCE_LEVELS.items()
    }
    assert len(set(archetypes.values())) == len(archetypes)
