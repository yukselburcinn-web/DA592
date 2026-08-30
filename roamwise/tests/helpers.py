"""Shared fixtures for the test suite: the catalogue-derived constants every
test module reads, and the helpers that keep assertions off hardcoded city
codes.

These were module-level in `test_pipeline.py` before #150 split it. They live
here rather than in `conftest.py` because they are plain constants and
functions, not pytest fixtures -- the tests read them directly.
"""

import datetime
from pathlib import Path

import pandas as pd
import pytest


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _dataset():
    """Read the city list and counts out of the committed CSVs.

    These were literals -- "ROM", "VIE", City == 8, POI == 1200 -- which pinned
    the suite to one particular catalogue. Changing the destination list then
    turned tests red for reasons that had nothing to do with the code under
    test. Reading the dataset keeps every assertion just as strong while
    letting the catalogue change underneath it.

    `FULL_CITIES` is the narrower set: a city needs rows in demand_timeseries
    and transport as well, and those files do not necessarily cover every
    destination. Tests that need them skip with a clear message rather than
    failing somewhere confusing.
    """
    dests = pd.read_csv(DATA_DIR / "destinations.csv")
    pois = pd.read_csv(DATA_DIR / "poi.csv")
    demand = pd.read_csv(DATA_DIR / "demand_timeseries.csv")
    transport = pd.read_csv(DATA_DIR / "transport.csv")

    counts = pois.destination_id.value_counts()
    # Deepest catalogue first: several tests slice [:40] or [:24], and a shallow
    # city would make those assertions vacuous rather than wrong.
    with_pois = sorted((c for c in dests.destination_id if counts.get(c, 0)),
                       key=lambda c: -counts[c])
    full = [c for c in with_pois
            if c in set(demand.destination_id) and c in set(transport.destination_id)]
    return with_pois, full, pois


CITY_CODES, FULL_CITIES, _POIS = _dataset()
POI_COUNT = len(_POIS)
MAIN_CITY = CITY_CODES[0]

# Applied to the tests that index FULL_CITIES[0]: without it an incomplete
# dataset fails with an IndexError at collection rather than saying why.
needs_full_city = pytest.mark.skipif(
    not FULL_CITIES,
    reason="hicbir sehirde hem demand_timeseries hem transport satiri yok")


def city_with_category(category, cities=None):
    """The deepest city holding POIs in this category, or None.

    The catalogue is not obliged to carry every category in every city -- the
    two-city set has no `beach` at all -- so a test that needs one asks for it
    instead of naming a city that happened to have it.
    """
    for code in (cities or CITY_CODES):
        if len(_POIS[(_POIS.destination_id == code) & (_POIS.category == category)]):
            return code
    return None


@needs_full_city

def _flat(zones: dict):
    """Zones were how the old router was *told* which POI belonged to which
    day. TOPTW decides that itself, jointly with selection and ordering, so
    these tests hand it the same POIs as one pool and say how many days they
    have. What each test asserts about the result is unchanged."""
    return [poi for zone in zones.values() for poi in zone], len(zones)


# The weekday the suite plans against when a test needs the calendar to be
# fixed. It lives here rather than beside the opening-hours tests that first
# needed it (#70) because three files assert against it now: opening hours,
# the orchestrators' start_date wiring (#76), and the LLM budget sweep (#125).
MONDAY = datetime.date(2026, 9, 7)
