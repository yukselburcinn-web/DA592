"""
Travel-mode profiles (issue #19).

Before this, every route was priced as walking at a flat 4.5km/h, so a
traveler who intends to drive got an itinerary sized for someone on foot --
far fewer stops per day than they could actually manage, and daily zones
drawn tighter than they needed to be.

A mode changes three things about how a day is costed:
  - `speed_kmh`: how fast a leg is covered when the haversine estimate is
    used (no OSRM).
  - `osrm_profile`: which public OSRM profile supplies real street distances
    when `use_real_routing=True`. routing.openstreetmap.de exposes
    routed-foot / routed-bike / routed-car; we use foot and car.
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
    osrm_profile: str
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
    key="walking", label="Walking", speed_kmh=4.5, osrm_profile="foot", stop_overhead_min=0.0,
)
# 25km/h, not a highway speed: this is door-to-door urban driving in the
# historic centres these itineraries cover, where traffic and one-way systems
# dominate. The 8-minute overhead is parking plus the walk in from it.
DRIVING = TravelMode(
    key="driving", label="Driving", speed_kmh=25.0, osrm_profile="car", stop_overhead_min=8.0,
)
HYBRID = HybridTravelMode(
    key="hybrid", label="Walking + driving", speed_kmh=WALKING.speed_kmh,
    osrm_profile=WALKING.osrm_profile, stop_overhead_min=0.0,
    walk=WALKING, drive=DRIVING,
)

TRAVEL_MODES = {m.key: m for m in (WALKING, DRIVING, HYBRID)}
DEFAULT_MODE = WALKING.key


def get_travel_mode(mode) -> TravelMode:
    """Accepts a mode key or a TravelMode; unknown keys fall back to walking
    so a bad value from the UI can never break planning."""
    if isinstance(mode, TravelMode):
        return mode
    return TRAVEL_MODES.get(mode or DEFAULT_MODE, WALKING)
