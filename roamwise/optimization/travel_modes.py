"""
Travel-mode profiles (issue #19).

Before this, every route was priced as walking at a flat 4.5km/h, so a
traveler who intends to drive got an itinerary sized for someone on foot --
far fewer stops per day than they could actually manage, and daily zones
drawn tighter than they needed to be.

A mode changes three things about how a day is costed (`transit` is the
exception that proves the shape -- see TRANSIT):
  - `speed_kmh`: how fast a leg is covered. This is the mode's whole speed
    model: with real routing off it prices the haversine estimate, and with
    it on it prices the real street distance (street_network.py). It is a
    measured door-to-door figure, not a posted limit -- see DRIVING.
  - `network_profile`: which street network a leg is measured on -- "foot"
    (footpaths) or "car" (the road network). It selects one of the committed
    per-city networks in `data/street_network/`, and was named
    `osrm_profile` while those distances came from a public OSRM server
    (issue #32 replaced it).
  - `stop_overhead_min`: fixed per-leg cost that isn't travel time itself --
    parking and walking in from the car for `driving`. Walking has none.

`hybrid` is deliberately not a single speed: it is the way people actually
move around a city -- walk anything close, take a car for the long hops
between neighbourhoods. It is modeled per leg (see `leg_minutes`), so a
2.5km hop across town is driven while the 300m between two adjacent museums
is walked, rather than averaging both into a speed that is wrong for each.
"""
from dataclasses import dataclass

# Below this, hybrid walks; above it, hybrid drives. 1.2km is roughly the
# distance past which a typical traveler stops walking by choice (~16 min on
# foot), and it keeps stops inside one neighbourhood on foot.
HYBRID_WALK_THRESHOLD_KM = 1.2


@dataclass(frozen=True)
class TravelMode:
    key: str
    label: str
    speed_kmh: float
    network_profile: str
    stop_overhead_min: float

    def leg_minutes(self, distance_km: float) -> float:
        return (distance_km / self.speed_kmh) * 60 + self.stop_overhead_min


@dataclass(frozen=True)
class HybridTravelMode(TravelMode):
    """Walks short legs, drives long ones -- each leg costed on its own."""
    walk: TravelMode = None
    drive: TravelMode = None
    threshold_km: float = HYBRID_WALK_THRESHOLD_KM

    def leg_minutes(self, distance_km: float) -> float:
        sub = self.walk if distance_km <= self.threshold_km else self.drive
        return sub.leg_minutes(distance_km)


WALKING = TravelMode(
    key="walking", label="Walking", speed_kmh=4.5, network_profile="foot", stop_overhead_min=0.0,
)
# 25km/h, not a highway speed: this is door-to-door urban driving in the
# historic centres these itineraries cover, where traffic and one-way systems
# dominate. The 8-minute overhead is parking plus the walk in from it.
DRIVING = TravelMode(
    key="driving", label="Driving", speed_kmh=25.0, network_profile="car", stop_overhead_min=8.0,
)
# Public transport (issue #32 stage 2). Unlike the others, this mode's times
# are not a distance divided by a speed: they are journey times solved by
# RAPTOR over the city's GTFS timetable, access walk, waiting and changes
# included (`optimization/raptor.py`), and they already take the walk wherever
# the timetable cannot beat it. So `stop_overhead_min` is zero -- the waiting
# is inside the number, not on top of it.
#
# `speed_kmh` is only the fallback for a city with no committed timetable, and
# 8km/h is what the Paris matrix actually works out to door-to-door in a
# straight line (median over 143,262 pairs; p10 4.3, p90 11.7). It is
# deliberately not a vehicle speed: a metro does 30km/h between stations and
# perhaps 8km/h once you count walking to it, waiting for it and changing.
# Cities without a timetable do not offer this mode at all -- see
# `street_network.available_cities("transit")`.
TRANSIT = TravelMode(
    key="transit", label="Public transport", speed_kmh=8.0,
    network_profile="transit", stop_overhead_min=0.0,
)
HYBRID = HybridTravelMode(
    key="hybrid", label="Walking + driving", speed_kmh=WALKING.speed_kmh,
    network_profile=WALKING.network_profile, stop_overhead_min=0.0,
    walk=WALKING, drive=DRIVING,
)

TRAVEL_MODES = {m.key: m for m in (WALKING, DRIVING, TRANSIT, HYBRID)}
DEFAULT_MODE = WALKING.key
# street_network.py is handed a profile, not a mode -- it has to turn "foot"
# back into the speed that profile's legs are priced at. HYBRID is not in
# here: it has no single profile, and is resolved into its walk/drive halves
# before a matrix is ever requested (see routing._build_hybrid_matrix_functions).
_MODE_BY_PROFILE = {m.network_profile: m for m in (WALKING, DRIVING, TRANSIT)}


def mode_for_network_profile(profile: str) -> TravelMode:
    """The mode whose legs are measured on `profile`'s street network."""
    return _MODE_BY_PROFILE.get(profile, WALKING)


def get_travel_mode(mode) -> TravelMode:
    """Accepts a mode key or a TravelMode; unknown keys fall back to walking
    so a bad value from the UI can never break planning."""
    if isinstance(mode, TravelMode):
        return mode
    return TRAVEL_MODES.get(mode or DEFAULT_MODE, WALKING)
