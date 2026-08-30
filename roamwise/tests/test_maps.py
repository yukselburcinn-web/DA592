"""Map framing (#21): every stop on the canvas, and none of the canvas wasted.
"""

import math

import pytest

from roamwise.knowledge_graph.build_graph import GraphIndex
from roamwise.tests.helpers import CITY_CODES


# --- issue #21: map framing ---

def _visible_spans(zoom: float, center_lat: float, width_px: int, height_px: int):
    """Degrees of latitude and longitude visible at a given Web Mercator zoom.

    Mirrors what MapLibre renders: the world is 512 * 2**zoom pixels wide, and
    latitude is compressed by cos(lat) relative to longitude.
    """
    world = 512 * 2 ** zoom
    return (height_px * 360.0 / world * math.cos(math.radians(center_lat)),
            width_px * 360.0 / world)


@pytest.mark.parametrize("city", CITY_CODES)
def test_map_view_frames_every_stop_without_wasting_the_canvas(city):
    """Both halves of the framing criterion at once: no stop may fall off the
    map, and the itinerary may not sit in a small island of empty canvas.

    The heuristic this replaced compared raw latitude and longitude degrees as
    if they were the same unit, so it mis-framed north-south days by the
    Mercator factor 1/cos(lat) and filled roughly a third of the viewport.
    """
    from roamwise.views.itinerary import _fit_view, _MAP_ASSUMED_WIDTH_PX, _MAP_HEIGHT_PX

    idx = GraphIndex()
    pois = idx.city_pois(city)[:12]
    lats = [p["lat"] for p in pois]
    lons = [p["lon"] for p in pois]

    zoom, center_lat, center_lon = _fit_view(lats, lons)
    # checked at the width the fit assumes, which is deliberately the narrowest
    # this column renders at -- a wider canvas only adds margin, never clips
    lat_span, lon_span = _visible_spans(zoom, center_lat, _MAP_ASSUMED_WIDTH_PX, _MAP_HEIGHT_PX)

    assert min(lats) >= center_lat - lat_span / 2, f"{city}: a stop falls off the south edge"
    assert max(lats) <= center_lat + lat_span / 2, f"{city}: a stop falls off the north edge"
    assert min(lons) >= center_lon - lon_span / 2, f"{city}: a stop falls off the west edge"
    assert max(lons) <= center_lon + lon_span / 2, f"{city}: a stop falls off the east edge"

    fill = max((max(lats) - min(lats)) / lat_span, (max(lons) - min(lons)) / lon_span)
    assert fill > 0.5, f"{city}: itinerary fills only {fill:.0%} of the map -- zoomed too far out"


def test_map_view_centres_on_the_bounding_box_not_the_mean():
    """One outlying stop used to drag the centre toward the cluster it was
    furthest from, because the old code averaged the coordinates."""
    from roamwise.views.itinerary import _fit_view

    clustered = [41.900, 41.901, 41.902, 41.903]
    outlier = [41.960]
    lats = clustered + outlier
    lons = [12.50] * len(lats)

    _, center_lat, _ = _fit_view(lats, lons)

    assert center_lat == pytest.approx((min(lats) + max(lats)) / 2)
    assert center_lat > sum(lats) / len(lats), "mean-centring would sit inside the cluster"


def test_map_view_survives_a_single_stop():
    """A one-stop day has no extent to fit against; it must clamp rather than
    divide by zero or zoom to infinity."""
    from roamwise.views.itinerary import _fit_view, _MAP_MAX_ZOOM

    zoom, center_lat, center_lon = _fit_view([41.9], [12.5])

    assert zoom == pytest.approx(_MAP_MAX_ZOOM)
    assert (center_lat, center_lon) == (41.9, 12.5)
