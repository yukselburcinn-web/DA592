"""The router's optimizer: a Team Orienteering Problem with Time Windows.

The router used to solve a TSP -- order every candidate, minimise distance,
then drop whatever did not fit. That is the wrong question. A traveler's day
is time-limited and the catalogue is larger than the day, so the decision is
not *what order* but *which stops*: pick the subset worth visiting, assign it
to days, and sequence it, all against real opening hours. TOPTW is the
canonical formulation of exactly that, and the six special-case passes the
old router grew -- day filling, day rebalancing, meal insertion, evening
insertion, a nightlife hour floor and a nightlife-last reorder -- are all
constraints of this one model instead (issue #72).

Those passes were not merely redundant, they interacted. Each ran over a
route the previous one had settled, against the same budget, with no view of
the others: measured on the same candidate set, raising `min_food_per_day`
from 0 to 2 *removed* a nightlife stop, and the two-meal guarantee itself
held on only 28 of 72 measured days -- falling to 4 of 24 on eighteen-hour
days, because the longer the day the more passes competed over it. Under this
model it holds on all of them.

The model, as OR-Tools sees it:

  vehicles      one per day of the trip
  timeline      one absolute clock across the whole trip; day d owns the
                minutes [d*1440 + start, d*1440 + start + budget]
  nodes         one copy of each POI *per day*, pinned to that day's vehicle
                and carrying that day's opening window; a measured POI gets a
                second copy per day pinned to its quietest hours (#33)
  disjunction   a POI's copies, max cardinality 1 -- visit it on one of the
                days, or pay `drop_penalty_m` to skip it
  arc cost      distance in metres
  dimensions    Time (with waiting slack), Meals (exactly n per day),
                and one per category (the diversity cap)

The per-day copies are load-bearing, not tidiness. Modelled the other way --
one node per POI carrying the union of every day's windows -- nothing binds a
window to a vehicle until the routing is half-built, propagation collapses,
and a 71-POI pool does not find a non-empty route in twenty seconds. Pinned
copies bind each window to one vehicle up front, and the same pool solves in
under a second.

Scale: comfortable to roughly 120 POIs (about 360 nodes, ~2s). A full city
catalogue (371 POIs, 1113 nodes) did not solve in ten minutes, which is why
callers select candidates before handing them here -- see
`optimization.scoring.select_by_score`.
"""
import datetime

from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from ortools.util.optional_boolean_pb2 import BOOL_TRUE as _BOOL_TRUE

from roamwise.optimization.routing import (
    DEFAULT_DAY_START_HOUR, FOOD_CATEGORY, NIGHTLIFE_EARLIEST_HOUR,
    _build_distance_functions, _is_food, _is_nightlife, _opening_intervals)
from roamwise.optimization.scoring import busyness_over, expected_busyness, quality
from roamwise.optimization.travel_modes import DEFAULT_MODE

# What a stop is worth, in metres of extra walking, when nothing says
# otherwise. Swept over 72 days x two pool sizes: below 2000 the solver buys
# distance by dropping stops the old router kept, and from 8000 up the
# frontier is flat because the candidate pool rather than the penalty is the
# binding constraint. 8000 is the cheapest point on the flat part -- it takes
# every stop the pool can offer without paying for stops nobody asked for.
DEFAULT_DROP_PENALTY_M = 8000
# What a landmark is worth keeping, on top of that (issue #122).
#
# One scalar penalty means the solver values every stop the same, so a stop
# that sits away from the centre is always the cheapest thing to drop --
# measured, the Eiffel Tower reached the shortlist in 3 of 7 archetypes and the
# plan in 0 of 7, and so did the Berlin Wall. Nothing upstream can fix that:
# widening the pool to the whole catalogue leaves it dropped, and makes iconic
# coverage *worse*, because more candidates means more competition under a
# score that does not favour fame (#122's own measurement).
#
# This is not the idea `evaluation/toptw_scoring_ablation.py` rejected at #72.
# That one reweighted *every* node with the selection score, and cost stops and
# distance for a 4% move in its own objective. Here a handful of POIs -- the
# ones whose within-pool prominence clears the threshold -- cost more to skip,
# and everything else keeps the flat price. The two knobs are swept together in
# `evaluation/iconic_penalty_sweep.py`, over 2 cities x 7 archetypes x 3 days:
#
#   thresh  mult   iconic kept   stops/day   km/stop
#     -      1.0      33/168      8.071       0.624     <- before
#    0.99    1.5      65/168      7.976       0.763
#    0.97    1.5      80/168      8.095       0.755     <- shipped
#    0.97    2.0      84/168      8.048       0.815
#    0.95    3.0      84/168      7.929       0.983
#
# 0.97 x 1.5 because it dominates: more landmarks kept than 0.99 (80 against
# 65) at *less* distance per stop (+21.0% against +22.3%) and no cost in stops
# at all (+0.3%). The pairs that keep another four (0.97 x 2.0, 0.95 x 2.0) buy
# them at +30.6% and +54.5% km/stop, so this follows #63 and takes the setting
# that gives back least of #72's -55.3% rather than the biggest headline count.
#
# **This pair was chosen twice, and the first choice is recorded rather than
# quietly replaced.** Swept before `router_agent.ICONIC_GUARANTEED` existed --
# when a landmark reached the pool for 3 of 7 archetypes rather than all 7 --
# the answer was 0.99 x 1.5, and 0.97 looked like it cost half again as much
# distance for its extra coverage. Widening which POIs are candidates changed
# which threshold is efficient, because 0.99 admits about four POIs of a
# working set and 0.97 about eight: with landmarks guaranteed, the second rank
# is what 0.99 leaves on the table. A swept number is only worth the conditions
# it was swept under -- the same lesson #126's chain weight recorded.
#
# The multiplier saturates: at 0.97 both 2.0 and 3.0 return 84/168, because a
# landmark already costs more to skip than any detour to reach it is worth.
#
# What it spends is real and is not a rounding error: a landmark sits away from
# the centre cluster, so keeping it means walking to it. See REPORT §5.
ICONIC_QUALITY_THRESHOLD = 0.97
ICONIC_DROP_MULTIPLIER = 1.5
# At most this many stops of one category in a day, the diversity factor from
# the issue's score formula. It cannot be a node weight -- "the third museum
# today" is a property of a partial route, not of a POI -- so it lives here,
# as the category quota a constraint model can express and a chain of passes
# cannot. Food is exempt: it carries its own exact-per-day contract.
#
# 4, swept over 72 days at two pool sizes against the pre-#72 router:
#
#   pool          cap 3            cap 4            cap 5           uncapped
#   24    6.08 / 0.862     6.69 / 0.906     7.15 / 1.006     7.36 / 1.084
#   72    8.85 / 0.536     9.06 / 0.471     9.28 / 0.481     9.32 / 0.474
#                          (stops per day / km per stop)
#
# The two pools disagree, and that is the whole reason this value is not
# obvious. On the retrieved pool the app asks for today (24 POIs, and for a
# Culture Enthusiast 19 of them museums) the cap is a hard constraint: 3 costs
# stops outright against the old router, and loosening it buys stops by
# spending distance. On a 72-POI pool there is enough variety that the cap
# stops binding past 4 -- km per stop is flat from there (0.471, 0.481, 0.474)
# and the extra stops are close to free.
#
# 4 is the value that is defensible on both. It is the km-per-stop optimum on
# the larger pool, and on the smaller one it still beats the old router on
# stops (+7.8%) where 3 does not (-2.0%). Going further trades away the
# diversity the cap exists for, and buys little the larger pool does not
# already give.
DEFAULT_MAX_SAME_CATEGORY = 4
# A fixed iteration budget rather than a wall-clock one, so the same input
# gives the same itinerary on any machine and under any load. Verified: three
# consecutive solves of the same case return byte-identical routes.
SOLUTION_LIMIT = 150
# Where each meal aims, as a fraction of the day's own window rather than a
# fixed clock time. Fixed times (13:00, 19:00) are wrong for a day running
# 09:00-15:00 -- dinner falls outside it and the meal is simply dropped.
# Fractions degrade sensibly: on a 12-hour day from 09:00 they land at 13:48
# and 19:12, real lunch and dinner; on a 6-hour day they become an early and a
# late lunch, the honest answer for someone out only until 15:00.
MEAL_DAY_FRACTIONS = (0.40, 0.85)
# How far from its target a meal may sit. Wide enough to reach a restaurant
# the traveler is already passing, narrow enough that two meals cannot both
# land at noon -- which a bare "two food stops a day" count would allow, and
# which is why that count alone is not what the issue asked for.
MEAL_WINDOW_HOURS = 2.0
# The hours people actually eat, and the least a meal may be from the next.
# Fractions of the day window alone are not enough: a Nightlife Seeker's day
# starts at 15:00, so on an 18-hour day the two fractions landed at 22:12 and
# *06:18* -- an hour with no restaurant open anywhere in the catalogue, which
# is why ten of the twelve Nightlife Seeker trips came back a meal short. The
# band is what keeps a meal at a mealtime; the gap is what stops two of them
# collapsing onto each other once the band has clamped them.
EATING_BAND = (11.5, 22.5)
MIN_MEAL_GAP_HOURS = 3.0
# What a missed sitting costs, as a multiple of one dropped stop. It has to
# outweigh the stops a day gives up to fit a meal in, and nothing more: set
# far above the rest of the objective (10,000,000 was the first attempt) the
# guided local search degenerates and returns an empty trip -- every stop
# dropped -- because one term dwarfs every gradient it could follow. Measured:
# at 100,000 the solver returns nothing, at 3x the drop penalty it fills the
# days and honours both sittings.
MEAL_SHORTFALL_MULTIPLE = 3
# A day the traveler asked for should not come back empty while there is still
# something to put in it. TOPTW on its own has no reason to spread: packing
# every stop into one day collects the same score for less distance, so a
# two-day trip offering four museums returned all four on day one and nothing
# on day two. This is the constraint form of what `_rebalance_days` used to do
# by hand, and it binds only when the pool is small -- on a real retrieved pool
# the time budget fills the days without it.
MIN_STOPS_PER_DAY = 1
# Roughly what a day holds, used only to tell a pool that cannot fill the trip
# from one that can. See the local-search operators below.
TYPICAL_STOPS_PER_DAY = 9
_DEFAULT_VISIT_MINUTES = 60

# --- The hour half of the crowding factor (issue #33) -----------------------
# `scoring.crowding_discount` knows how busy a POI is at a given hour and
# nothing asked it: a node weight is static, and which hour a stop is visited
# was the solver's decision rather than an input to it. So the router placed
# people at 56.9% busy against the 47.9% their own typical level predicts --
# systematically worse than chance, because the hours a day naturally fills
# are the hours everyone else fills too.
#
# The fix is the same trick the meal sittings already use: a POI gets one node
# per slot of the day, each carrying that slot's window, and the model picks a
# slot by picking a node. Busyness then attaches to a *node*, which a static
# weight can express.
# How wide the quiet window is. Three hours is about two stops, and it has to
# be wide enough that the rest of the day can still be built around it: at one
# hour almost nothing fits inside it once travel and a 60-90 minute visit are
# paid for, and at six it stops being a quiet window at all -- the catalogue's
# mean hour runs 30% busy at 09:00 and 61% at 15:00, so a six-hour window
# starting in the morning has the afternoon peak inside it.
CROWD_QUIET_WINDOW_HOURS = 3.0
# The same idea inside a meal sitting, and narrower because the sitting is
# already narrow: a band is `2 * MEAL_WINDOW_HOURS` = 4 hours wide, so a
# three-hour window inside it would barely be a choice. Two hours still holds
# a 60-minute sitting plus the walk to it.
CROWD_MEAL_WINDOW_HOURS = 2.0
# What a busy hour costs, in the metres this model denominates everything else
# in -- a stop is already "worth taking if it costs less than `drop_penalty_m`
# of walking", so an hour can be priced the same way.
#
# Swept over 2 cities x 4 archetypes x 3 day lengths
# (`evaluation/crowding_hour_measurement.py`). The gap columns are exposure
# minus the same stops' own typical level: how much busier than its own
# average a stop is at the hour the router picked for it, which is the only
# thing here that is about the hour rather than about which POI was chosen.
# Read `sight gap` for this knob -- meals are priced by CROWD_MEAL_COST_M and
# were held at 1500 throughout.
#
#   cost      stops/day   km/stop   exposure   sight gap
#   off (pre-#33)  6.71     1.025      58.6%      +10.7
#   0              6.71     1.040      51.5%       +8.0
#   1000           6.68     1.054      47.5%       +1.6
#   2000           6.64     1.001      49.1%       +4.3
#   4000           6.62     1.028      44.3%       -2.1
#   8000           6.56     1.004      44.3%       -2.4
#   16000          4.71     0.803      36.4%       -7.3
#
# 4000 is the cheapest price that takes the sight gap negative -- stops land
# slightly quieter than their own average rather than busier -- and it costs
# 1.3% of the stops and nothing measurable in distance. The middle of the
# sweep is not monotone (1000 beats 2000), which is the search finding a
# different local optimum rather than a real preference, so no value between
# them is worth reading closely.
#
# Past 8000 the sweep stops buying quiet hours and starts buying quiet by not
# going: at 16000 a stop at a 50%-busy hour costs more than the 8000 m it is
# worth to skip, so the solver skips it -- 30% of the trip's stops, which is
# not a scheduling improvement. The 0 arm is worth reading too: a quarter of
# the gain is the extra node alone, before anything is charged for taking it.
CROWD_COST_M = 4000
# Absolute busyness, not busyness relative to the POI's own quietest hour. The
# relative form is the tidier idea -- every POI keeps a way to be visited for
# nothing, so the price can only ever be about *when*, never about which POI
# is worth visiting. Measured over the same 24 trips at the same 4000 it is
# beaten on both counts: hour gap +4.4 against +2.0, km/stop 1.110 against
# 1.055. Being blind to which POI it is looking at costs it -- the solver
# takes a busier POI whenever geometry favours it and pays nothing for the
# swap, and exposure is an absolute quantity. Set True to reproduce.
CROWD_RELATIVE_TO_QUIETEST = False
# Meals get their own, lower price (#109). A sitting is compulsory, so the
# solver cannot answer a high crowd price by skipping the meal -- it answers by
# rebuilding the day around whichever restaurant has a cheap quiet hour, and
# the sightseeing stops pay for it. At the sightseeing price of 4000 the meal
# gap overshot to -9.4 points while the sightseeing gap went the wrong way,
# -0.4 -> +2.1, and the trip lost 2.5% of its stops.
#
# Swept over the same 24 trips, sightseeing held at CROWD_COST_M:
#
#   meal cost          sight gap   meal gap   stops/day   km/stop
#   0 (offered, free)      -1.0       +7.1       6.67      1.033
#   1000                   +0.7       -2.1       6.62      1.011
#   1500                   -2.1       -5.1       6.62      1.028
#   2000                   -1.1       -9.0       6.58      1.051
#   2500                   -0.6      -10.5       6.60      1.058
#
# 1500 takes the meal gap 12.2 points, from +7.1 to -5.1, for 0.7% of the
# stops and nothing in distance. Above it the gap keeps falling and the stops
# start paying: 2500 buys 5.4 more points for 2.4% on distance per stop, which
# is spending walking on moving a dinner half an hour. The meal contract is
# unmoved at every price in the sweep -- 2.0 meals a day, both sittings filled
# on 100% of the 72 days -- which is the first thing to check here, because a
# sitting cannot be declined and a change that improved this gap by quietly
# dropping one would look like a win in every other column (#20, #29).
CROWD_MEAL_COST_M = 1500
# How much quieter the quiet window has to be before a POI earns a second
# node. Nodes are not free: the first design cut every measured POI's day into
# a fixed grid of slots, which tripled the node count, and at the same
# iteration budget the search got worse at everything -- on Paris / Culture
# Enthusiast / 12h it cost 1.7km with the crowd price set to zero, before the
# factor could buy anything at all.
CROWD_MIN_SPREAD = 10.0
_WEEKDAY_CODES = ("Mo", "Tu", "We", "Th", "Fr", "Sa", "Su")


def _weekday_code(start_date, day: int):
    """`data/crowding.csv`'s day code for trip day `day`, or None with no date.

    None is not a degraded answer here -- `busyness_over` reads it as "any
    weekday", which averages the readings. A trip with no calendar date
    genuinely does not know whether its day 2 is a Tuesday or a Saturday, and
    a POI's weekday means differ by a median 8.7 points, so averaging is the
    honest stand-in rather than picking one."""
    if start_date is None:
        return None
    return _WEEKDAY_CODES[(start_date + datetime.timedelta(days=day)).weekday()]


def _quiet_window(poi: dict, weekday, band_start: float, band_end: float,
                  width: float = CROWD_QUIET_WINDOW_HOURS):
    """The `width` hours inside `[band_start, band_end]` when this POI is emptiest.

    Returns `(low, high, level)`, or None when the POI was never measured or
    the band holds no room for a window of that width."""
    best = None
    # From the band's own start, not from the hour boundary below it. A meal
    # band starts at 17.2 as readily as at 17.0, and flooring it let a dinner
    # window open twelve minutes before dinner -- small, but the band is the
    # only thing keeping a sitting at a mealtime (#20, #29).
    low = float(band_start)
    while low + width <= band_end + 1e-9:
        level = busyness_over(poi, weekday, low, low + width)
        if level is not None and (best is None or level < best[2]):
            best = (low, low + width, level)
        low += 1.0
    return best


def _crowd_slots(poi: dict, weekday, day_start_hour: float, budget_minutes: int,
                 hour_aware: bool) -> list:
    """This POI's options for the day, each with the busyness it carries.

    Always at least `(None, typical)`: one node spanning the whole day, priced
    at the level the POI runs at over its open hours. A measured POI with a
    genuinely quieter stretch gets a second node pinned to it, and the choice
    between the two is the choice of hour -- which is the thing a static node
    weight could not express and the thing issue #33 is about.

    Adding a node rather than replacing the day-wide one matters. It means
    turning this on cannot take an itinerary away: every route that was
    reachable before is still reachable, at the same price, and the quiet
    window is an extra offer rather than a constraint. The first design cut the
    day into slots instead, and a stop that fitted no slot simply lost its
    place in the day.

    The whole-day node is priced even for a POI nobody measured -- via
    `expected_busyness`'s category fallback -- because a stop priced at nothing
    is the cheapest stop in the pool, and 59% of the catalogue is unmeasured.
    Being unmeasured must not be an advantage; it is simply not a *choice* of
    hour."""
    if not hour_aware:
        # Not "priced at zero" -- priced at nothing at all. Charging the
        # day-wide level here too would make `hour_aware=False` a *different*
        # router rather than today's one, and the arm this is measured against
        # has to be the thing that shipped.
        return [(None, None)]
    anytime = [(None, expected_busyness(poi))]
    window = _quiet_window(poi, weekday, day_start_hour,
                           day_start_hour + budget_minutes / 60)
    if window is None or anytime[0][1] - window[2] < CROWD_MIN_SPREAD:
        return anytime
    low, high, level = window
    return [(("q", low, high), level)] + anytime


def _meal_slots(poi: dict, weekday, meals: list, hour_aware: bool) -> list:
    """A meal POI's options, one sitting at a time.

    Issue #33 left the sittings out on the grounds that *when* a meal happens
    is not a choice the model has -- the meal dimension makes both sittings
    compulsory. Half of that is right and the half that is wrong turned out to
    carry the whole residual: a sitting is pinned to a four-hour band around
    its target, not to a minute, and inside that band a restaurant's busiest
    hour is nothing like its quietest. Measured over the 26 restaurants the
    series covers, the dinner band runs 36% busy at its quietest hour and 100%
    at its busiest, a median 25 points between the quiet hour and the band's
    own average. Sightseeing stops came out of #33 at -0.4 points against
    their own typical level while meal stops stayed at +8.2, and this is where
    that came from (#109).

    The band-wide copy always stays, for the same reason it does for a sight:
    the quiet stretch is an extra offer, never a constraint, so a day that can
    only fit dinner at its busiest hour still fits dinner."""
    out = []
    for sitting, low, high in meals:
        band = expected_busyness(poi, weekday, low, high) if hour_aware else None
        if hour_aware:
            window = _quiet_window(poi, weekday, low, high, CROWD_MEAL_WINDOW_HOURS)
            if window is not None and band - window[2] >= CROWD_MIN_SPREAD:
                out.append((("m", sitting, window[0], window[1]), window[2]))
        out.append((sitting, band))
    return out


def _sitting_of(slot):
    """Which sitting a copy fills, or None if it is not a meal copy at all.

    A sitting has two kinds of copy now -- the band and the quiet stretch
    inside it -- and the meal dimension has to count them as the same sitting,
    or "one lunch per day" becomes "one lunch per day per kind of copy"."""
    if isinstance(slot, tuple):
        return slot[1] if slot[0] == "m" else None
    return slot


def _visit_minutes(poi: dict) -> int:
    return int(round(poi.get("avg_visit_minutes", _DEFAULT_VISIT_MINUTES)))


def meal_target_hours(day_start_hour: float, daily_minutes_budget: int,
                       n_meals: int) -> list[float]:
    """Clock time each meal aims at: spread over the day, then pulled back to
    hours a kitchen is open.

    The fractions come first because they are what makes a short day work --
    a day running 09:00-15:00 has no dinner in it, and two sittings inside
    what it does have is the honest answer. The band comes second because the
    fractions alone do not know what a mealtime is: a day starting at 15:00
    put its second sitting after midnight. When clamping pushes two sittings
    within MIN_MEAL_GAP_HOURS of each other, they are respread evenly across
    whatever of the band the day covers, which is the case the band was
    needed for in the first place."""
    if n_meals < 1:
        return []
    span = daily_minutes_budget / 60
    fractions = (list(MEAL_DAY_FRACTIONS) if n_meals <= 2
                 else [(i + 1) / (n_meals + 1) for i in range(n_meals)])
    low = max(day_start_hour, EATING_BAND[0])
    high = min(day_start_hour + span, EATING_BAND[1])
    if low >= high:  # the day does not overlap any mealtime at all
        return [day_start_hour + f * span for f in fractions[:n_meals]]

    targets = [min(max(day_start_hour + f * span, low), high)
               for f in fractions[:n_meals]]
    too_close = any(b - a < MIN_MEAL_GAP_HOURS
                    for a, b in zip(targets, targets[1:]))
    if too_close:
        targets = [low + (i + 1) / (n_meals + 1) * (high - low)
                   for i in range(n_meals)]
    return targets


def _earliest_hour_for(poi: dict):
    """A bar is not worth visiting the moment its doors unlock. Several carry
    early OSM hours, so the category has an earliest *sensible* hour of its
    own (issue #59) -- here it is simply the lower bound of the time window
    rather than a separate reordering pass."""
    return NIGHTLIFE_EARLIEST_HOUR if _is_nightlife(poi) else None


def _day_windows(poi: dict, day: int, day_start_hour: float, budget_minutes: int,
                  start_date, respect_opening_hours: bool,
                  slot_hours: tuple[float, float] = None) -> list[tuple[int, int]]:
    """When this POI may be *arrived at* on this day, in absolute minutes.

    Two rules, the same two `optimize_day_route` applied one stop at a time:
    walk in before it closes, and finish inside the day's budget. A visit that
    runs past closing time is allowed, exactly as it was before -- the model is
    not held to a stricter standard than the router it replaces."""
    base = day * 1440 + int(round(day_start_hour * 60))
    day_end = base + budget_minutes
    if slot_hours is not None:  # a meal slot narrows the day to lunch or dinner
        base = max(base, day * 1440 + int(round(slot_hours[0] * 60)))
        day_end = min(day_end, day * 1440 + int(round(slot_hours[1] * 60)))
    visit = _visit_minutes(poi)
    if not respect_opening_hours:
        return [(base, day_end - visit)] if base <= day_end - visit else []

    day_date = None if start_date is None else start_date + datetime.timedelta(days=day)
    windows = []
    for open_hour, close_hour in _opening_intervals(poi, _earliest_hour_for(poi), day_date):
        low = max(base, day * 1440 + int(round(open_hour * 60)))
        high = min(day_end - visit, day * 1440 + int(round(close_hour * 60)) - 1)
        if low <= high:
            windows.append((low, high))
    return windows


def solve(pois: list[dict], n_days: int, start_hub: dict = None,
          arrival_hub: dict = None,
          daily_minutes_budget: int = 480, day_start_hour: float = 9.0,
          respect_opening_hours: bool = True, start_date=None,
          distance_fn=None, duration_fn=None, used_real_routing: bool = False,
          leg_mode_fn=None,
          min_food_per_day: int = 0, drop_penalty_m: int = DEFAULT_DROP_PENALTY_M,
          iconic_quality_threshold: float = None, iconic_drop_multiplier: float = None,
          max_same_category: int = DEFAULT_MAX_SAME_CATEGORY,
          hour_aware: bool = True) -> list[dict]:
    """One model, every day. Returns one dict per day in the shape the rest of
    the app already reads -- route, distance_km, total/active/idle minutes,
    schedule -- so views and narration need no change.

    `pois` is the whole working set, meals included; which of them get visited,
    on which day, in which order and at what hour is what this decides.

    `start_hub` is where every day begins -- the city centre, standing in for a
    centrally booked hotel. `arrival_hub` overrides that for day 1 only, and is
    the gateway the traveler actually lands at; leave it None and the trip
    begins in the city, which is the right answer for someone already there.
    """
    days_out = [_empty_day(v, start_date) for v in range(n_days)]
    if not pois or n_days < 1:
        return days_out

    meals = [(k, low, high) for k, target in enumerate(
        meal_target_hours(day_start_hour, daily_minutes_budget, min_food_per_day))
        for low, high in [(target - MEAL_WINDOW_HOURS, target + MEAL_WINDOW_HOURS)]]

    # One node per (POI, day, slot). A meal POI gets one slot per sitting, each
    # carrying that sitting's window -- which is how "lunch and dinner" becomes
    # a constraint rather than a count. A bare count of two food stops a day is
    # satisfied by two lunches at 11:00 and 11:45, and that is what the old
    # `_ensure_daily_meals` pass existed to prevent.
    #
    # A sightseeing POI used to have a single slot spanning the whole day. It
    # now gets one per hour slot when its busyness varies across them (#33),
    # which is what lets the model choose the hour rather than inherit it. The
    # sittings are deliberately left out of that: the meal dimension makes
    # every sitting compulsory, so *when* a meal happens is not a choice the
    # model has -- only which restaurant fills it, and which is what the static
    # discount in `scoring` already answers.
    slot_hours = {k: (low, high) for k, low, high in meals}
    copies = []       # (poi_index, day, slot or None)
    crowd_level = {}  # index into `copies` -> busyness this copy's hours carry
    for poi_i, poi in enumerate(pois):
        is_meal = min_food_per_day > 0 and _is_food(poi)
        for day in range(n_days):
            if is_meal:
                slots = _meal_slots(poi, _weekday_code(start_date, day), meals,
                                    hour_aware)
            else:
                slots = _crowd_slots(poi, _weekday_code(start_date, day),
                                     day_start_hour, daily_minutes_budget, hour_aware)
            for slot, level in slots:
                if isinstance(slot, tuple):
                    slot_hours[slot] = tuple(slot[-2:])
                if level is not None:
                    crowd_level[len(copies)] = level
                copies.append((poi_i, day, slot))

    depot = 0
    end_node = 1 + len(copies)
    # A trip that begins at an airport or a station begins there exactly once.
    # `arrival_hub` is a node of its own that only day 1 starts from, which is
    # what `router_agent.py` has promised in a comment since #19 without an
    # implementation behind it: anchoring *every* day at the gateway would walk
    # the traveler back out to the edge of town each morning -- Charles de
    # Gaulle is 23km from the centre -- and that is what the city-centre depot
    # is for.
    arrival_node = end_node + 1 if arrival_hub else None
    n_nodes = end_node + 1 + (1 if arrival_hub else 0)
    hub = start_hub or pois[0]
    coords = ([hub] + [pois[c[0]] for c in copies] + [hub]
              + ([arrival_hub] if arrival_hub else []))
    terminals = {depot, end_node, arrival_node}

    def entry_of(node: int):
        """(poi, day, slot) for a copy, or None for a depot or the arrival hub."""
        return None if node in terminals else (
            pois[copies[node - 1][0]], copies[node - 1][1], copies[node - 1][2])

    # Precomputed: the callbacks are hit hundreds of thousands of times during
    # local search, and recomputing haversine inside each one dominated the solve.
    dist_m = [[0] * n_nodes for _ in range(n_nodes)]
    time_m = [[0] * n_nodes for _ in range(n_nodes)]
    for a in range(n_nodes):
        for b in range(n_nodes):
            # Nothing leaves the end depot and nothing travels *to* the
            # arrival hub -- it is where day 1 starts, not a place to reach.
            if a == b or a == end_node or b == end_node or b == arrival_node:
                continue
            dist_m[a][b] = int(round(distance_fn(coords[a], coords[b]) * 1000))
            time_m[a][b] = int(round(duration_fn(coords[a], coords[b])))

    # What a busy hour costs, in the metres this model denominates everything
    # else in -- a stop is already "worth taking if it costs less than
    # `drop_penalty_m` of walking", so an hour can be priced the same way.
    #
    # Every stop carries one, measured or not -- see `_crowd_slots`. A POI
    # split into hour slots has a different one per slot, which is the whole
    # mechanism: choosing the node chooses the hour, and the hour has a price.
    crowd_m = [0] * n_nodes
    if crowd_level:
        quietest: dict[int, float] = {}
        if CROWD_RELATIVE_TO_QUIETEST:
            for c, level in crowd_level.items():
                poi_i = copies[c][0]
                quietest[poi_i] = min(quietest.get(poi_i, level), level)
        for c, level in crowd_level.items():
            floor = quietest.get(copies[c][0], 0.0)
            cost = (CROWD_MEAL_COST_M if _sitting_of(copies[c][2]) is not None
                    else CROWD_COST_M)
            crowd_m[c + 1] = int(round(cost * (level - floor) / 100.0))

    starts = [arrival_node if arrival_node is not None and v == 0 else depot
              for v in range(n_days)]
    manager = pywrapcp.RoutingIndexManager(n_nodes, n_days, starts,
                                            [end_node] * n_days)
    routing = pywrapcp.RoutingModel(manager)

    def distance_cb(i, j):
        # The crowd surcharge rides on the arc *into* a node, which is what
        # makes it sum to the trip's total crowd exposure over whatever route
        # the solver settles on. Reported distance is not taken from here --
        # `_finish_day` re-measures it with `distance_fn` -- so the itinerary
        # still shows the kilometres actually walked.
        b = manager.IndexToNode(j)
        return dist_m[manager.IndexToNode(i)][b] + crowd_m[b]

    def time_cb(i, j):
        a, b = manager.IndexToNode(i), manager.IndexToNode(j)
        entry = entry_of(a)
        return (_visit_minutes(entry[0]) if entry else 0) + time_m[a][b]

    routing.SetArcCostEvaluatorOfAllVehicles(routing.RegisterTransitCallback(distance_cb))
    time_idx = routing.RegisterTransitCallback(time_cb)

    # An empty vehicle is "unused" by default, and an unused vehicle's soft
    # cumul penalties are left out of the objective entirely -- so every
    # per-day floor below (a meal at each sitting, at least one stop) silently
    # stopped applying to exactly the days that needed it: the empty ones. It
    # is why raising those penalties changed nothing.
    for v in range(n_days):
        routing.SetVehicleUsedWhenEmpty(True, v)

    routing.AddDimension(time_idx, daily_minutes_budget, (n_days + 1) * 1440, False, "Time")
    time_dim = routing.GetDimensionOrDie("Time")
    for v in range(n_days):
        base = v * 1440 + int(round(day_start_hour * 60))
        time_dim.CumulVar(routing.Start(v)).SetRange(base, base)
        time_dim.CumulVar(routing.End(v)).SetRange(base, base + daily_minutes_budget)

    solver = routing.solver()
    by_poi: dict[int, list[int]] = {}
    for node, (poi_i, day, slot) in enumerate(copies, start=1):
        index = manager.NodeToIndex(node)
        by_poi.setdefault(poi_i, []).append(index)
        routing.VehicleVar(index).SetValues([-1, day])  # this copy is that day's
        windows = _day_windows(pois[poi_i], day, day_start_hour, daily_minutes_budget,
                               start_date, respect_opening_hours,
                               slot_hours.get(slot) if slot is not None else None)
        if not windows:
            solver.Add(routing.ActiveVar(index) == 0)  # shut, or shut at this sitting
            continue
        cumul = time_dim.CumulVar(index)
        cumul.SetRange(min(w[0] for w in windows), max(w[1] for w in windows))
        if len(windows) > 1:  # e.g. a lunchtime closure splits the day
            solver.Add(solver.Max([(cumul >= low) * (cumul <= high)
                                   for low, high in windows]) == 1)
        else:
            cumul.SetRange(*windows[0])

    # Visit each POI once across the whole trip, or pay to skip it. Spanning
    # every copy -- all days, all sittings -- is also what stops the same
    # restaurant being booked twice.
    # What it costs to skip each POI. Scalar until #122: every stop was worth
    # the same 8 km of walking, so the ones furthest from the cluster were
    # always the cheapest to lose however well known they were. Prominence is
    # `scoring.quality` -- the same min-max of `popularity_score` retrieval
    # ranks with, normalised within this working set, so "the best known here"
    # means the best known among the candidates this trip actually has.
    #
    # None rather than the constants as default arguments: a default is bound
    # once at import, and the sweep sets the module constants.
    threshold = (ICONIC_QUALITY_THRESHOLD if iconic_quality_threshold is None
                 else iconic_quality_threshold)
    multiplier = (ICONIC_DROP_MULTIPLIER if iconic_drop_multiplier is None
                  else iconic_drop_multiplier)
    fame = quality(pois) if multiplier != 1.0 else None
    for poi_i in sorted(by_poi):
        penalty = drop_penalty_m
        if fame is not None and fame[poi_i] >= threshold:
            penalty = drop_penalty_m * multiplier
        routing.AddDisjunction(by_poi[poi_i], max(int(penalty), 1), 1)

    for k, _, _ in meals:
        # One sitting filled per day, as a constraint of the model rather than
        # a pass that runs afterwards and evicts what an earlier pass placed
        # (#20, #29). The upper bound matters as much as the lower one: without
        # it, under a pool where restaurants outnumber sights and cluster
        # tightly, the cheapest way to collect stops is a food crawl -- measured,
        # a day came back holding nine restaurants and nothing else.
        def meal_cb(i, _j, k=k):
            entry = entry_of(manager.IndexToNode(i))
            return 1 if entry and _sitting_of(entry[2]) == k else 0

        name = f"Meal_{k}"
        routing.AddDimension(routing.RegisterTransitCallback(meal_cb), 0, len(pois),
                             True, name)
        dim = routing.GetDimensionOrDie(name)
        for v in range(n_days):
            dim.CumulVar(routing.End(v)).SetMax(1)
            # Soft below, so a day with no restaurant open at that sitting stays
            # solvable rather than making the whole trip infeasible.
            dim.SetCumulVarSoftLowerBound(
                routing.End(v), 1, max(int(drop_penalty_m) * MEAL_SHORTFALL_MULTIPLE, 1))

    # Every day gets something, if anything can go in it.
    def stop_cb(i, _j):
        return 1 if entry_of(manager.IndexToNode(i)) else 0

    routing.AddDimension(routing.RegisterTransitCallback(stop_cb), 0, len(copies),
                         True, "Stops")
    stops_dim = routing.GetDimensionOrDie("Stops")
    for v in range(n_days):
        stops_dim.SetCumulVarSoftLowerBound(
            routing.End(v), MIN_STOPS_PER_DAY,
            max(int(drop_penalty_m) * MEAL_SHORTFALL_MULTIPLE, 1))

    if max_same_category:
        for category in sorted({p.get("category") for p in pois} - {FOOD_CATEGORY, None}):
            def category_cb(i, _j, category=category):
                entry = entry_of(manager.IndexToNode(i))
                return 1 if entry and entry[0].get("category") == category else 0

            routing.AddDimension(routing.RegisterTransitCallback(category_cb), 0,
                                 max_same_category, True, f"Cat_{category}")

    params = pywrapcp.DefaultRoutingSearchParameters()
    # PARALLEL_CHEAPEST_INSERTION, not the usual PATH_CHEAPEST_ARC. Every day
    # here carries per-vehicle quotas -- one stop per meal sitting, a cap per
    # category -- and PATH_CHEAPEST_ARC builds routes one vehicle at a time, so
    # it spends the reachable meal stops on the first day and dead-ends on the
    # rest. Measured on a 24-candidate Paris pool it returned [0, 0, 7] stops
    # across the three days; building every day at once returns [6, 7, 6] on
    # the same input. Guided local search could not repair it afterwards --
    # raising the iteration limit sevenfold changed nothing.
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION)
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    # Moving a POI from one day to another means deactivating that day's copy
    # and activating another day's -- two nodes, one decision. No default
    # local-search operator makes that move, so an unbalanced first solution
    # stayed unbalanced: four museums over two days came back [4, 0], the
    # empty day's penalty paid rather than fixed.
    params.local_search_operators.use_swap_active = _BOOL_TRUE
    if len(pois) < n_days * TYPICAL_STOPS_PER_DAY:
        # ...but only a pool too small to fill the trip needs that repair, and
        # the repair is expensive. `use_extended_swap_active` is what actually
        # fixes [4, 0] (plain `use_swap_active` does not), and on a pool that
        # small it costs nothing. On a full one it is 4-5x the entire solve for
        # no gain at all: measured on a Paris trip, three days went 3.0s -> 16.2s
        # and *lost* a stop ([9,9,9] at 9.00 stops/day became [9,9,8] at 8.67),
        # four days 4.2s -> 18.2s for a byte-identical itinerary. An oversupplied
        # pool does not need the repair -- the time budget and the drop penalty
        # spread the stops on their own -- so the operator is spent only where
        # it earns its cost.
        params.local_search_operators.use_extended_swap_active = _BOOL_TRUE
    params.solution_limit = SOLUTION_LIMIT
    solution = routing.SolveWithParameters(params)
    if solution is None:
        return days_out

    for v in range(n_days):
        route, schedule = [], []
        origin = arrival_hub if (arrival_hub and v == 0) else start_hub
        km, active, previous = 0.0, 0.0, origin
        index = routing.Start(v)
        while not routing.IsEnd(index):
            entry = entry_of(manager.IndexToNode(index))
            if entry:
                poi = entry[0]
                arrival = solution.Value(time_dim.CumulVar(index)) / 60 - v * 24
                visit = _visit_minutes(poi)
                # Per leg, not just summed into the day's totals: the itinerary
                # tells the traveler how each stop is reached (#159), and the
                # only honest source for that is the leg the solver was
                # actually charged for. `leg_mode` matters on hybrid, where it
                # differs from stop to stop; the other modes report themselves.
                leg_km = 0.0 if previous is None else distance_fn(previous, poi)
                leg_min = 0.0 if previous is None else duration_fn(previous, poi)
                km += leg_km
                active += leg_min + visit
                route.append(poi)
                schedule.append({
                    "arrival": arrival, "finish": arrival + visit / 60,
                    "leg_km": leg_km, "leg_minutes": leg_min,
                    # None when `solve` was called directly without one --
                    # the evaluation scripts do that, and they never read it.
                    "leg_mode": (None if previous is None or leg_mode_fn is None
                                 else leg_mode_fn(previous, poi)),
                })
                previous = poi
            index = solution.Value(routing.NextVar(index))
        days_out[v] = _finish_day(v, start_date, route, schedule, km, active,
                                  day_start_hour, used_real_routing,
                                  starts_from=(arrival_hub if origin is arrival_hub else None),
                                  origin=origin)
    return days_out


def _empty_day(day: int, start_date) -> dict:
    return {"day": day + 1,
            "date": None if start_date is None else start_date + datetime.timedelta(days=day),
            "route": [], "distance_km": 0.0, "total_minutes": 0, "active_minutes": 0,
            "idle_minutes": 0, "schedule": [], "used_real_routing": False,
            "starts_from": None, "origin": None}


def _finish_day(day: int, start_date, route, schedule, km, active,
                day_start_hour, used_real_routing, starts_from=None,
                origin=None) -> dict:
    """`starts_from` is the arrival hub, on the one day that begins at one. It
    is carried rather than inferred because the day's first leg is measured
    from a place that is not in `route`, and a reader looking at 23km on day 1
    otherwise has no way to see where it came from.

    `origin` is that same place for every day -- the city centre on an ordinary
    day, the arrival hub on the one that begins at one -- and it carries
    coordinates rather than a name because the map has to *draw* the leg. It
    used to be dropped here, so the map showed a day's stops joined to each
    other while `distance_km` also counted the journey out to the first one:
    for a Paris walking day that is 1.2-1.7km of the total, drawn nowhere
    (#94)."""
    if not schedule:
        return _empty_day(day, start_date)
    # Derived from each other rather than rounded independently, so the three
    # always reconcile and no caller can show a breakdown that doesn't add up.
    span = int(round((max(s["finish"] for s in schedule) - day_start_hour) * 60))
    active_minutes = min(int(round(active)), span)
    return {
        "day": day + 1,
        "date": None if start_date is None else start_date + datetime.timedelta(days=day),
        "route": route,
        "distance_km": round(km, 2),
        "total_minutes": span,
        "active_minutes": active_minutes,
        "idle_minutes": span - active_minutes,
        "schedule": schedule,
        "used_real_routing": used_real_routing,
        "starts_from": None if starts_from is None else starts_from.get("name"),
        "origin": None if origin is None else {
            "name": origin.get("name"), "lat": origin["lat"], "lon": origin["lon"]},
    }


def build_multi_day_itinerary(pois: list[dict], n_days: int, start_hub: dict = None,
                               arrival_hub: dict = None,
                               daily_minutes_budget: int = 480,
                               day_start_hour: float = DEFAULT_DAY_START_HOUR,
                               respect_opening_hours: bool = True,
                               use_real_routing: bool = False, travel_mode=DEFAULT_MODE,
                               food_pois: list[dict] = None, min_food_per_day: int = 0,
                               start_date=None,
                               drop_penalty_m: int = DEFAULT_DROP_PENALTY_M,
                               iconic_quality_threshold: float = None,
                               iconic_drop_multiplier: float = None,
                               max_same_category: int = DEFAULT_MAX_SAME_CATEGORY,
                               hour_aware: bool = True) -> list[dict]:
    """The router's entry point: candidates in, one routed day per trip day out.

    Note what this signature no longer takes. It used to be handed
    `pois_by_zone` -- KMeans had already decided which POIs belonged to which
    day, and everything after that could only shuffle stops between days that
    geography had already fixed. Day assignment is a decision the model makes
    now, jointly with selection and ordering, so the caller passes a flat pool
    and says how many days it has.

    `food_pois` are candidates for the day's meals. They are kept apart from
    the sightseeing pool for the same reason as before -- retrieval is
    preference-driven and a Culture Enthusiast's query surfaces no restaurants
    (issue #20) -- but they are now solved *with* everything else rather than
    inserted into a finished route.

    `start_date` is the trip's first day. Day N falls on `start_date + N-1`,
    and that date is what lets opening hours be read as the grammar they are
    rather than as a single open/close pair (issue #70).

    `arrival_hub` is the airport, station or terminal the traveler arrives at.
    Day 1 starts there instead of at `start_hub`; every later day is unchanged
    (issue #32).
    """
    meal_pool = list(food_pois or []) if min_food_per_day > 0 else []
    working_set = list(pois) + meal_pool
    if not working_set:
        return [_empty_day(v, start_date) for v in range(max(n_days, 0))]

    # One distance matrix for the whole trip rather than one per day. This
    # was originally about not being rate-limited by a public routing server;
    # since #32 the matrix is local, but solving a trip's geometry once is
    # still the right shape -- the days share a single set of points.
    points = ([start_hub] if start_hub else []) + \
        ([arrival_hub] if arrival_hub else []) + working_set
    distance_fn, duration_fn, used_real_routing, leg_mode_fn = _build_distance_functions(
        points, use_real_routing, travel_mode)

    return solve(working_set, n_days, start_hub=start_hub, arrival_hub=arrival_hub,
                 daily_minutes_budget=daily_minutes_budget,
                 day_start_hour=day_start_hour,
                 respect_opening_hours=respect_opening_hours, start_date=start_date,
                 distance_fn=distance_fn, duration_fn=duration_fn,
                 used_real_routing=used_real_routing, leg_mode_fn=leg_mode_fn,
                 min_food_per_day=min_food_per_day, drop_penalty_m=drop_penalty_m,
                 iconic_quality_threshold=iconic_quality_threshold,
                 iconic_drop_multiplier=iconic_drop_multiplier,
                 max_same_category=max_same_category, hour_aware=hour_aware)
