"""
The traveler-facing prototype (proposal deliverable #1: "An Interactive
Prototype... allowing a user to input preferences and receive a data-backed,
graph-grounded, optimized itinerary").

A page, not an entry point: `roamwise/app.py` is what Streamlit runs, and it
has already fixed sys.path and called st.set_page_config by the time this
executes. Run the app with: streamlit run roamwise/app.py
"""
import datetime
import math
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from roamwise.agents.llm_client import LLMRequestFailed, describe_client, fallback_reason
from roamwise.agents.orchestrator import RoamWiseOrchestrator
from roamwise.models.forecasting import forecast_window, history_end
from roamwise.optimization.routing import route_geometry
from roamwise.optimization.travel_modes import TRAVEL_MODES

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


# What a leg of each mode is called in the itinerary. Hybrid never appears
# here: it names the trip, not a leg -- every hybrid leg is priced as one of
# walking or driving and reports that (#159).
_LEG_VERBS = {"walking": "walk", "driving": "drive", "transit": "by transit"}


def _clock(hour: float) -> str:
    """Fractional hour (13.5) -> wall clock ("13:30"), so the itinerary reads
    as a plan for a day rather than as an ordered list."""
    hours, minutes = divmod(int(round(hour * 60)), 60)
    return f"{hours % 24:02d}:{minutes:02d}"


# The map is drawn at a fixed height in a half-width column. Zoom in Web
# Mercator is defined against pixels, so framing an itinerary means asking how
# many pixels its bounding box needs and picking the zoom where that fits.
_MAP_HEIGHT_PX = 460
# Width is fluid (width='stretch'), so Python cannot measure it, and the
# fit has to assume a value. The two failure modes are not symmetric: assuming
# too much width pushes stops off the edge and the user loses information,
# while assuming too little only leaves unused margin. So this is the narrowest
# the half-width column realistically renders at, not the typical one.
#
# 420 was tried first and measured 5 of 8 cities clipping on a 291px canvas --
# which is what this column collapses to in a narrow window. At 300 nothing
# clips down to that width. The cost is real and worth stating: on a 482px
# canvas the itinerary fills ~54% instead of ~67%. Removing that slack needs
# the actual width, which means measuring it client-side rather than guessing.
_MAP_ASSUMED_WIDTH_PX = 300
_MAP_PADDING = 0.82  # keep markers and their labels clear of the edges
_MAP_MAX_ZOOM = 16.0  # a one-stop day would otherwise zoom to street level

# The legend strip above the map. Plotly wraps a horizontal legend against the
# width it actually gets, which Python cannot measure -- these only decide how
# much vertical room to reserve for it. Reserving too little would let the
# legend sit on top of the map, so the estimate is deliberately the pessimistic
# one: three entries per row is what fits at the narrowest this column renders
# at (_MAP_ASSUMED_WIDTH_PX). On a wide screen a five-day legend fits on one
# row and the second reserved row is simply unused white space.
_MAP_LEGEND_PER_ROW = 3
_MAP_LEGEND_ROW_PX = 30  # a rendered row measures 29px; round up, never down


# A leg with no real geometry is drawn as a dashed straight line. Plotly's map
# traces carry no `dash` property -- `Scattermap.line` is width and colour and
# nothing else -- so the gaps are made in the data, by splitting the segment
# and dropping every other piece. A fixed number of dashes rather than a fixed
# dash length, because the map is pan-and-zoomable and a metre-based dash
# would collapse to a solid line at one zoom and a dotted haze at another,
# while a fixed count reads the same at every scale.
_DASHES_PER_LEG = 13


def _dashed(start, end, pieces: int = _DASHES_PER_LEG):
    """(lats, lons) for a dashed straight segment, gapped with None."""
    lats, lons = [], []
    for i in range(0, pieces, 2):
        a, b = i / pieces, min((i + 1) / pieces, 1.0)
        lats += [start[0] + (end[0] - start[0]) * a,
                 start[0] + (end[0] - start[0]) * b, None]
        lons += [start[1] + (end[1] - start[1]) * a,
                 start[1] + (end[1] - start[1]) * b, None]
    return lats, lons


def _fit_view(lats: list[float], lons: list[float]) -> tuple[float, float, float]:
    """Bounding box of the stops -> (zoom, center_lat, center_lon).

    This replaces a heuristic that compared raw latitude and longitude degrees
    against each other. In Web Mercator they are not comparable: latitude is
    stretched by 1/cos(latitude), so at Amsterdam one degree of latitude covers
    1.6x the pixels of one degree of longitude. A north-south day was therefore
    framed as though it were 1.6x smaller than it is. The old version also
    centred on the *mean* of the stops rather than the middle of their bounding
    box, which let one outlying stop drag the frame away from everything else,
    and clamped zoom to [11, 15] for no geometric reason -- a clamp that bound
    on two of the eight cities. Measured over 3-day itineraries in all eight,
    the itinerary filled about a third of the map; fitting the box properly
    fills it.
    """
    south, north = min(lats), max(lats)
    west, east = min(lons), max(lons)
    center_lat, center_lon = (south + north) / 2, (west + east) / 2

    # a degenerate box (one stop, or several at the same address) has no scale
    # to fit against, so fall through to the max-zoom clamp below
    lat_span = max(north - south, 1e-6)
    lon_span = max(east - west, 1e-6)

    # Web Mercator, 512px tiles: the whole world is 512 * 2**zoom pixels wide.
    zoom_x = math.log2(_MAP_ASSUMED_WIDTH_PX * 360.0 / (512.0 * lon_span))
    zoom_y = math.log2(_MAP_HEIGHT_PX * 360.0 * math.cos(math.radians(center_lat))
                       / (512.0 * lat_span))
    zoom = min(zoom_x, zoom_y) + math.log2(_MAP_PADDING)
    return min(_MAP_MAX_ZOOM, zoom), center_lat, center_lon


# The sidebar used to let travelers pick fusion/hybrid/standard themselves.
# That was a comparative-analysis knob from the proposal, not a choice anyone
# planning a trip can make meaningfully -- and only one answer is ever right,
# since Fusion RAG wins on every metric the evaluation measures. It is pinned
# here, and the comparison it came from now lives on the System logs screen
# where it reads as evidence instead of as a question (issue #42).
RETRIEVAL_CONFIG = "fusion"


@st.cache_resource(show_spinner="Loading RoamWise engine...")
def get_orchestrator(config: str):
    # Without an explicit show_spinner message, Streamlit's default caching
    # spinner shows the raw call signature ("Running get_orchestrator(...)."),
    # which is an implementation detail, not something a user needs to see.
    return RoamWiseOrchestrator(retrieval_config=config)


_MODE_LABELS = {"walking": "On foot", "driving": "By car",
                "transit": "Public transport",
                "hybrid": "Walk nearby, drive between areas"}


def _travel_mode_options(destination_id) -> dict:
    """Travel modes this city can honestly offer, in TRAVEL_MODES' order.

    Walking, driving and hybrid are arithmetic and work anywhere. Public
    transport is a timetable, and a city whose timetable we do not ship has no
    transit answer at all -- offering the mode there would dress a straight
    line at an average speed up as a journey, on services that may not exist.
    Paris is the pilot (issue #32 stage 2); every other city keeps exactly the
    three modes it had.
    """
    from roamwise.optimization.street_network import available_cities

    has_timetable = destination_id in available_cities("transit")
    return {key: _MODE_LABELS[key] for key in TRAVEL_MODES
            if key != "transit" or has_timetable}


# Below this, the transfer in from the gateway is just a leg. Above it, it is
# the shape of day 1: an hour of a twelve-hour day is already most of a
# morning, and the Charles de Gaulle walk is 5.8 of them.
_LONG_TRANSFER_MINUTES = 60
# And below this, the alternative is not worth interrupting anyone for.
# Driving in from Charles de Gaulle takes 65 minutes against transit's 50 --
# real, but not the difference between a plan and a write-off.
_WORTH_SWITCHING = 1.5


def _arrival_transfer_hint(destination_id, arrival_hub_id, travel_mode):
    """"That gateway is a long way out and there is a faster way in", or None.

    The itinerary already tells the truth about this -- day 1 comes back with
    23.84 km and two fewer stops -- but only after planning, and only to
    someone who thinks to compare it against day 2. A traveller who picks an
    airport and "on foot" has asked for a 5.8-hour walk without knowing it,
    and the honest moment to say so is before the trip is planned.

    Deliberately not a block or an auto-switch: walking in from Orly is a
    strange choice, not an invalid one, and the mode stays the traveller's.
    """
    if not arrival_hub_id or travel_mode == "transit":
        return None
    hubs = pd.read_csv(DATA_DIR / "transport.csv")
    hub = hubs[hubs.transport_id == arrival_hub_id]
    destinations = pd.read_csv(DATA_DIR / "destinations.csv")
    centre = destinations[destinations.destination_id == destination_id]
    if hub.empty or centre.empty:
        return None
    points = [{"lat": float(hub.iloc[0].lat), "lon": float(hub.iloc[0].lon)},
              {"lat": float(centre.iloc[0].lat), "lon": float(centre.iloc[0].lon)}]

    # Costed exactly the way the router will cost it, rather than by a rule of
    # thumb that could disagree with the itinerary it is warning about.
    from roamwise.optimization.routing import _build_distance_functions

    distance_fn, duration_fn, real, _ = _build_distance_functions(
        points, use_real_routing=True, travel_mode=travel_mode)
    chosen = duration_fn(points[0], points[1])
    if not real or chosen < _LONG_TRANSFER_MINUTES:
        return None

    _, transit_duration, by_timetable, _ = _build_distance_functions(
        points, use_real_routing=True, travel_mode="transit")
    if not by_timetable:
        return None
    transit = transit_duration(points[0], points[1])
    if transit * _WORTH_SWITCHING > chosen:
        return None

    return (f"**{hub.iloc[0]['name']}** is {distance_fn(*points):.0f} km from the centre. "
            f"Getting in {_MODE_LABELS[travel_mode].lower()} takes about {_clock_span(chosen)}, "
            f"which comes out of day 1. Public transport does it in "
            f"{_clock_span(transit)}.")


def _clock_span(minutes: float) -> str:
    hours, rest = divmod(int(round(minutes)), 60)
    if not hours:
        return f"{rest} min"
    return f"{hours}h {rest:02d}m" if rest else f"{hours}h"


@st.cache_data
def _destination_options() -> dict:
    """City label -> destination_id, read from the catalogue.

    This was a hard-coded literal of eight cities, which meant the picker
    outlived the dataset: after the catalogue moved to Paris and Berlin it
    still offered Istanbul and Rome, and choosing one returned an itinerary
    with no stops. The dropdown has to be whatever `destinations.csv` holds.
    """
    dests = pd.read_csv(DATA_DIR / "destinations.csv")
    return {"Let RoamWise choose": None,
            **{row.city: row.destination_id for row in dests.itertuples()}}


# What `transport.csv`'s `type` column reads as to a traveller.
_HUB_LABELS = {"airport": "airport", "train_station": "train station",
               "bus_station": "bus terminal"}


def _arrival_options(destination_id) -> dict:
    """Gateway label -> transport_id, for the pinned city.

    Default first and default meaning "no gateway": the trip then begins in the
    city, which is both the previous behaviour and the right answer for someone
    who is already there. Silently assuming an airport would put Charles de
    Gaulle -- 23km out -- at the start of every Paris trip, including one that
    arrives by train.

    An unpinned destination offers nothing to choose from: the city is not
    known until planning has run, and the gateways are a property of the city.
    """
    options = {"Already in the city": None}
    if destination_id is None:
        return options
    hubs = pd.read_csv(DATA_DIR / "transport.csv")
    hubs = hubs[hubs["destination_id"] == destination_id]
    for tid, name, ttype in zip(hubs["transport_id"], hubs["name"], hubs["type"]):
        options[f"{name} ({_HUB_LABELS.get(ttype, ttype)})"] = tid
    return options



def _leg_caption(slot: dict, trip_mode: str) -> str:
    """"12 min walk &middot; 0.9 km" -- how the traveler got to this stop.

    Reads `leg_minutes` / `leg_km` / `leg_mode` off the schedule entry, which
    the router writes from the very leg it priced the day on (#159). Recomputing
    them here would be a second opinion, and #94 is what a second opinion costs.

    Distance is dropped for transit: the timetable solves a journey time, and
    the kilometres alongside it are the straight-line distance between the two
    stops rather than the route a service takes -- showing them would invite
    the reader to divide one by the other.
    """
    if not slot or slot.get("leg_minutes") is None:
        return ""
    minutes = slot["leg_minutes"]
    if minutes <= 0:
        return ""
    mode_key = slot.get("leg_mode") or trip_mode
    verb = _LEG_VERBS.get(mode_key, "travel")
    shown = f"{round(minutes)} min {verb}"
    if mode_key != "transit" and slot.get("leg_km"):
        shown += f" &middot; {slot['leg_km']:.1f} km"
    return f"&darr;&nbsp; {shown}"


def _humanize_category(category: str) -> str:
    return category.replace("_", " ").title() if category else ""


st.title("\U0001f9ed RoamWise")
st.caption("Agentic AI framework for personalized tourism forecasting and itinerary optimization -- DA592 prototype")

# The archetype match is a KMeans over six 0-1 survey features (see
# TravelerSegmenter), which needs real numbers -- but a 0.35 vs. 0.40 choice
# on a raw slider is not a meaningful distinction to a user and doesn't
# correspond to anything they can feel. Three named levels are plenty to
# separate archetype centroids (see roamwise/tests/test_pipeline.py) and are
# actually legible; the float mapping below is presentation-only and never
# changes the segmentation contract or user_survey.csv.
PREFERENCE_LEVELS = {"Low": 0.2, "Medium": 0.5, "High": 0.8}
_LEVEL_NAMES = list(PREFERENCE_LEVELS)
# What the sixth survey feature is held at now that the traveler is not asked
# for it -- the survey's own mean, so the segmentation sees a typical traveler
# on that axis rather than an invented one. See the sidebar below for why.
_ADVENTURE_FIXED = 0.42


def _level_slider(label: str, default: str, **kwargs) -> float:
    choice = st.select_slider(label, options=_LEVEL_NAMES, value=default, **kwargs)
    return PREFERENCE_LEVELS[choice]


with st.sidebar:
    st.header("Traveler preferences")
    budget = _level_slider(
        "Everyday or upmarket", "Medium",
        help="Shifts the plan between everyday stops and upmarket ones -- it reaches "
             "the score through shopping and landmarks. It does not filter by what "
             "things cost: the catalogue records free-or-paid rather than a price, so "
             "cost is shown per stop instead of being used to choose stops.")
    culture = _level_slider("Culture / history interest", "High")
    nature = _level_slider("Nature / outdoors interest", "Medium")
    nightlife = _level_slider("Nightlife interest", "Low")
    relax = _level_slider(
        "Relaxed pace", "Low",
        help="Slower days built around cafes, restaurants and easy culture stops.")
    # No "Adventure interest" slider, and its absence is the honest answer
    # rather than an oversight (#79). The catalogue has no adventure-ish
    # category -- `beach` was the nearest one the affinity table offered and it
    # holds zero POIs in these two inland cities -- so the fitted preference
    # matrix gives `adventure` a row of zeros and the score cannot act on it.
    # Measured: with every other slider held at 0.5, moving adventure from 0.1
    # to 0.9 changed none of the 24 selected POIs. What it *did* still do was
    # move the KMeans archetype, which is worse than doing nothing: a traveler
    # who asked for maximum adventure came back classified a Budget Backpacker
    # and got a different plan for a reason that has nothing to do with
    # adventure.
    #
    # The survey mean keeps the segmentation contract intact -- KMeans still
    # fits six features, `user_survey.csv` is untouched, and all seven
    # archetypes stay reachable from the five remaining sliders (Nature &
    # Adventure falls from 5.8% of profiles to 2.5%, reached through the nature
    # slider). Give the catalogue a real adventure category and this becomes a
    # slider again.
    adventure = _ADVENTURE_FIXED

    st.header("Trip constraints")
    dest_options = _destination_options()
    dest_label = st.selectbox("Destination", list(dest_options.keys()))
    destination_id = dest_options[dest_label]
    n_days = st.slider("Trip length (days)", 1, 5, 3)
    arrival_options = _arrival_options(destination_id)
    arrival_hub_id = arrival_options[st.selectbox(
        "Arriving at", list(arrival_options),
        help="Day 1 starts from the gateway you land at instead of from the city centre, "
             "so the transfer in is part of the plan rather than invisible. Later days "
             "start in the city. Pick a destination above to see its gateways.",
    )]
    # A date, not a month. The forecaster only ever needed the month and still
    # reads it from here, but opening hours are a rule over days of the week --
    # without a date the router cannot tell a Monday from a Tuesday and will
    # schedule a Monday-closed museum on a Monday (issue #70). One control still,
    # answering both questions.
    start_date = st.date_input(
        "Trip starts on",
        value=datetime.date.today() + datetime.timedelta(days=30),
        min_value=datetime.date.today(),
        help="Day 1 falls on this date. It sets the crowding forecast's month, and "
             "lets each stop be checked against the hours it actually keeps that "
             "day -- plenty of museums close on Mondays.",
    )
    # The cap used to be 12:00, which put an evening-shaped day out of reach:
    # nightlife is never scheduled before 18:00, so a day opening at 09:00
    # spends its first nine hours with nothing schedulable in them and comes
    # back holding one bar. Measured over Paris and Berlin, a 3-day 12-hour
    # trip returns [1, 1, 1] stops from 09:00 and [4, 4, 3] from 14:00 (#61).
    # Auto is the default because the archetype is only known after planning
    # runs -- the checkbox is how a traveler overrides it up front.
    auto_start = st.checkbox(
        "Start the day to suit my traveler type", value=True,
        help="A Nightlife Seeker's day opens at 15:00, a Culture Enthusiast's at 09:00. "
             "Uncheck to pick the hour yourself.")
    day_start_hour = None
    if not auto_start:
        day_start_hour = st.slider(
            "Day starts at", 7, 18, 9,
            format="%d:00",
            help="What time each day begins. A later start shifts the whole day, so evening "
                 "venues stay reachable within the same number of hours.",
        )
    daily_hours = st.slider(
        "Time out per day (hours)", 12, 18, 12, step=3,
        help="The whole active day, not just museums -- travel, visits, meals and any "
             "evening stop all come out of it. Nightlife venues open at 18:00, so a day "
             "has to be long enough to still be out then for one to be scheduled at all.",
    )
    if day_start_hour is not None:
        st.caption(f"Your day runs {day_start_hour:02d}:00 – "
                   f"{(day_start_hour + daily_hours) % 24:02d}:00.")
    else:
        st.caption(f"Each day will run {daily_hours} hours from whatever start your "
                   "traveler type calls for.")
    modes = _travel_mode_options(destination_id)
    travel_mode = st.radio(
        "How are you getting around?", list(modes), format_func=modes.get,
        help="Sets how travel between stops is costed, so a driving day can cover more ground "
             "than a walking one within the same time budget. Public transport appears for "
             "cities whose timetable ships with the app, and is read from that timetable "
             "rather than estimated.",
    )
    hint = _arrival_transfer_hint(destination_id, arrival_hub_id, travel_mode)
    if hint:
        st.warning(hint)
    # Not a checkbox any more (#160). It had been on by default since #94, and
    # the box asked the traveler a question they cannot answer: whether this
    # city has a committed street network. `fetch_distance_duration_matrix`
    # already answers it -- it returns None for a city we hold no network for
    # and every caller falls back to the haversine estimate rather than
    # raising, which is the "real road if we have one, straight line if not"
    # behaviour the box appeared to be offering. So the box never switched a
    # capability on; it switched measured distances *off*, and a day priced on
    # straight lines fills its budget wrongly and brings back #94's
    # contradiction between the route drawn and the distance reported.
    #
    # The parameter stays. `evaluation/toptw_measurement.py`,
    # `toptw_scoring_ablation.py` and `comparative_analysis.py` pass
    # `use_real_routing=False` explicitly to hold the condition REPORT
    # documents, and #93/#94 needed to ask what the table looks like with it
    # off. What is removed is the sidebar control, not the ability to answer
    # that question.
    use_real_routing = True

    run = st.button("Plan my trip", type="primary", width='stretch')

preferences = {"budget": budget, "culture": culture, "nature": nature,
               "nightlife": nightlife, "relax": relax, "adventure": adventure}

# What the plan on screen was built from. Kept beside the plan because the
# sidebar keeps moving after it: rendering "3-day route" from a slider the
# traveler has since dragged to 5 would describe the itinerary wrongly.
# `use_real_routing` was here until #160 removed its control: a setting that
# cannot change can never differ from the plan on screen, so watching it only
# invited the list to drift out of step with the sidebar.
_SHOWN_SETTINGS = ("n_days", "daily_hours", "auto_start", "travel_mode")

if run:
    orch = get_orchestrator(RETRIEVAL_CONFIG)
    with st.spinner("Agents at work: segmenting traveler, forecasting demand, retrieving grounded context, routing..."):
        try:
            st.session_state["plan_settings"] = {
                "n_days": n_days, "daily_hours": daily_hours, "auto_start": auto_start,
                "travel_mode": travel_mode,
            }
            st.session_state["plan"] = orch.plan_trip(
                preferences, destination_id=destination_id, n_days=n_days,
                start_date=start_date,
                daily_minutes_budget=daily_hours * 60,
                day_start_hour=(None if day_start_hour is None
                                else float(day_start_hour)),
                use_real_routing=use_real_routing, travel_mode=travel_mode,
                arrival_hub_id=arrival_hub_id)
        except LLMRequestFailed as exc:
            # A hosted model that would not answer (issue #7). The plan itself
            # is computed before the narration and does not depend on it, but
            # plan_trip narrates last, so there is nothing to keep -- and a
            # traceback is the wrong thing to show a traveler. Drop any stale
            # plan so the screen cannot pair a new error with an old itinerary.
            st.session_state.pop("plan", None)
            st.session_state["plan_error"] = str(exc)
        else:
            st.session_state.pop("plan_error", None)

if st.session_state.get("plan_error"):
    st.error(
        f"**The narration engine did not answer.** {st.session_state['plan_error']}\n\n"
        f"Nothing was planned. Check the System logs screen for the retries, or unset "
        f"`ROAMWISE_LLM` to fall back to the offline template."
    )

# Rendered from session state, not from the `run` branch, so the plan survives
# the reruns Streamlit performs on every widget interaction (issue #7). It used
# to live only inside `if run:` -- so nudging any sidebar slider silently threw
# the itinerary away and made the traveler press the button again, which on a
# hosted model is two more API calls and on the local one another eight minutes.
if st.session_state.get("plan"):
    orch = get_orchestrator(RETRIEVAL_CONFIG)
    result = st.session_state["plan"]

    # Everything below describes the plan, so it reads the settings the plan
    # was made with, not whatever the sidebar says now.
    _shown = st.session_state["plan_settings"]
    _live = {"n_days": n_days, "daily_hours": daily_hours, "auto_start": auto_start,
             "travel_mode": travel_mode}
    n_days, daily_hours, auto_start, travel_mode = (
        _shown[name] for name in _SHOWN_SETTINGS)

    if _live != _shown:
        st.info("Your settings have changed since this plan was made. "
                "Press **Plan my trip** to rebuild it.")

    city_name = orch.destinations.set_index("destination_id").loc[result["destination_id"], "city"]
    st.success(f"Plan ready: **{city_name}** for a **{result['archetype']}**")

    tab1, tab2, tab3 = st.tabs(["\U0001f4cd Itinerary", "\U0001f4c8 Demand forecast", "\U0001f50e Retrieved context"])

    with tab1:
        st.subheader(f"{n_days}-day route in {city_name}")
        chosen_start = result["routing"]["day_start_hour"]
        st.caption(
            f"Days run {_clock(chosen_start)}–{_clock(chosen_start + daily_hours)}"
            + (f", set from your **{result['archetype']}** profile." if auto_start
               else ", as you set them."))
        # Real routing is always asked for since #160, so the only question
        # left is whether this city had a network to answer with -- which is
        # what `used_real_routing` records, per day, from the router itself.
        # A traveler should be able to tell measured distances from estimated
        # ones without knowing which flag implied which.
        any_real = any(d.get("used_real_routing") for d in result["routing"]["itinerary"])
        if any_real and travel_mode == "transit":
            st.caption("Times below are journeys on the published timetable, including the "
                       "walk to the stop, the wait and any changes.")
        elif any_real:
            st.caption("Distances/times below follow the real street network.")
        else:
            st.caption("No street network covers this city, so distances below are "
                       "straight-line estimates at a flat walking speed.")
        col1, col2 = st.columns([1, 1])
        with col1:
            budget_minutes = daily_hours * 60
            for day in result["routing"]["itinerary"]:
                with st.container(border=True):
                    used = day["total_minutes"]
                    # The weekday is worth showing, not just carrying: it is the
                    # reason a stop is or isn't in this day at all (issue #70).
                    when = f" &middot; {day['date']:%a %d %b}" if day.get("date") else ""
                    st.markdown(f"**Day {day['day']}**{when} &mdash; {used // 60}h "
                                f"{used % 60:02d}m of your {daily_hours}h "
                                f"&middot; {day['distance_km']} km")
                    st.progress(min(1.0, used / budget_minutes) if budget_minutes else 0.0)
                    # The day's first leg is measured from a place that is not
                    # in `route`. Without this line a 26km day 1 next to a 4km
                    # day 2 looks like a bug rather than the airport transfer.
                    if day.get("starts_from"):
                        st.caption(f"Starts from {day['starts_from']}.")
                    if day["route"]:
                        schedule = day.get("schedule", [])
                        for i, poi in enumerate(day["route"], 1):
                            slot = schedule[i - 1] if i <= len(schedule) else None
                            when = f"`{_clock(slot['arrival'])}`&nbsp; " if slot else ""
                            # Meals are called "meal stop" rather than flagged
                            # with a cutlery emoji: the icon depends on the
                            # viewer having that glyph in their emoji font,
                            # and a word can't fail to render.
                            label = ("meal stop" if poi.get("category") == "food"
                                     else _humanize_category(poi.get("category", "")))
                            # Free entry is the only cost fact the catalogue
                            # actually holds -- OSM's `fee` tag. It used to be
                            # read by a filter that could never fire and shown
                            # to nobody (#67).
                            free = " &middot; _free_" if poi.get("price_level", 0) == 0 else ""
                            # How this stop is reached, before naming it. The
                            # sidebar lets the traveler pick a mode and the
                            # panel used to show only the day's total km, so
                            # "on foot" never said how far, and hybrid never
                            # said which legs were driven (#159). The numbers
                            # are the ones the router was charged for, read off
                            # the schedule -- not recomputed here, which is how
                            # the map and the panel came to disagree in #94.
                            leg = _leg_caption(slot, travel_mode)
                            if leg:
                                st.markdown(f"<div style='margin:-6px 0 2px 22px;opacity:.6;"
                                            f"font-size:.85em'>{leg}</div>",
                                            unsafe_allow_html=True)
                            st.markdown(f"{i}. {when}**{poi['name']}** _{label}_{free}")
                    else:
                        st.markdown("_No stops fit the time budget for this day._")
        with col2:
            fig = go.Figure()
            colors = px.colors.qualitative.Bold
            drawn_real = []
            for day in result["routing"]["itinerary"]:
                if not day["route"]:
                    continue
                color = colors[(day["day"] - 1) % len(colors)]
                stops = day["route"]
                schedule = day.get("schedule", [])
                # The marker carries the stop's position in the day so the map
                # reads in the same order as the list beside it. Names moved to
                # hover: printed on the map they overlapped into an unreadable
                # smear as soon as a day had more than three or four stops in a
                # dense city centre, which is every day in this dataset.
                hover = []
                for i, poi in enumerate(stops, 1):
                    slot = schedule[i - 1] if i <= len(schedule) else None
                    when = f"{_clock(slot['arrival'])} &middot; " if slot else ""
                    kind = "meal stop" if poi.get("category") == "food" else _humanize_category(poi.get("category", ""))
                    hover.append(f"<b>{poi['name']}</b><br>{when}Day {day['day']}, stop {i}<br><i>{kind}</i>")
                # The line used to be drawn straight from stop to stop whatever
                # the mode said, so a day the panel reported as 6.85 km was
                # drawn as 3.17 (#94). Where a street network priced the day,
                # the same network now supplies the path; where nothing can --
                # transit keeps minutes and no geometry, and a straight line is
                # the model itself when real routing is off -- the leg is
                # dashed rather than dressed up as a route.
                # The day's journey starts at its origin -- the city centre, or
                # the arrival hub on day 1 -- and `distance_km` has always
                # counted that first leg. It was drawn nowhere, so even a map
                # tracing every street would still come up short of the panel
                # by 1.2-1.7km on a Paris walking day.
                origin = day.get("origin")
                drawn_stops = ([origin] if origin else []) + stops
                geometry = route_geometry(drawn_stops, bool(day.get("used_real_routing")),
                                          travel_mode)
                line_lat, line_lon = [], []
                for vertices, is_real_route in geometry:
                    if is_real_route:
                        line_lat += [v[0] for v in vertices] + [None]
                        line_lon += [v[1] for v in vertices] + [None]
                    else:
                        dashed_lat, dashed_lon = _dashed(vertices[0], vertices[-1])
                        line_lat += dashed_lat
                        line_lon += dashed_lon
                    drawn_real.append(is_real_route)
                # Two traces, not one: a line whose vertices are the route and
                # markers that stay on the stops. `markers+lines` would put a
                # numbered circle on every vertex of the path.
                if line_lat:
                    fig.add_trace(go.Scattermap(
                        lat=line_lat, lon=line_lon, mode="lines",
                        line=dict(width=3, color=color),
                        hoverinfo="skip", showlegend=False,
                    ))
                fig.add_trace(go.Scattermap(
                    lat=[p["lat"] for p in stops], lon=[p["lon"] for p in stops],
                    mode="markers+text",
                    text=[str(i) for i in range(1, len(stops) + 1)],
                    textposition="middle center",
                    textfont=dict(size=11, color="white", family="Arial Black"),
                    hovertext=hover, hoverinfo="text",
                    name=f"Day {day['day']}",
                    marker=dict(size=20, color=color, opacity=0.95),
                ))
            if any(day["route"] for day in result["routing"]["itinerary"]):
                # Origins are in the fit because they are now drawn: leave them
                # out and day 1's line to an arrival hub 23km away runs off the
                # edge of the map with no way to tell it is there.
                drawn_points = [p for day in result["routing"]["itinerary"]
                                for p in (([day["origin"]] if day.get("origin") else [])
                                          + day["route"])]
                all_lats = [p["lat"] for p in drawn_points]
                all_lons = [p["lon"] for p in drawn_points]
                zoom, center_lat, center_lon = _fit_view(all_lats, all_lons)
                # The legend sits in the margin *below* the map. Every other
                # strip is already occupied: inside the map at the bottom is
                # the OpenStreetMap attribution, and the top belongs to
                # Plotly's zoom/pan modebar, which appears on hover. Anchored
                # bottom-left inside the plot it collided with the attribution
                # and ran out of width -- this column is half-width and
                # collapses to ~291px (see _MAP_ASSUMED_WIDTH_PX) -- so from
                # three days on the later entries were cut off mid-word and the
                # user had no key for those colours at all (#83). Below the map
                # it gets the full column width, wraps instead of clipping, and
                # covers neither the route nor a control.
                n_days_drawn = sum(1 for day in result["routing"]["itinerary"] if day["route"])
                legend_rows = math.ceil(n_days_drawn / _MAP_LEGEND_PER_ROW)
                legend_px = legend_rows * _MAP_LEGEND_ROW_PX
                fig.update_layout(
                    map=dict(style="open-street-map", zoom=zoom,
                             center=dict(lat=center_lat, lon=center_lon)),
                    margin=dict(l=0, r=0, t=0, b=legend_px),
                    height=_MAP_HEIGHT_PX + legend_px,
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="top", y=0,
                                xanchor="left", x=0, bgcolor="rgba(0,0,0,0)",
                                borderwidth=0),
                    hoverlabel=dict(bgcolor="white", font_size=12),
                )
                st.plotly_chart(fig, width='stretch')
                # What the line between two stops is, said plainly. A solid
                # line is a route; a dashed one is a straight connector and
                # nothing more, which is all transit and all straight-line
                # costing can honestly offer (#94).
                if all(drawn_real):
                    route_note = ("Route lines follow the real street network, so their "
                                  "length matches the distance shown per day. ")
                elif not any(drawn_real):
                    reason = ("the timetable stores journey times, not the path a service takes"
                              if travel_mode == "transit" else
                              "distances are straight-line estimates, which is what the dashed "
                              "lines show")
                    route_note = (f"Dashed lines join stops directly rather than tracing a route -- "
                                  f"{reason}. ")
                else:
                    route_note = ("Solid lines follow the real street network; dashed ones join "
                                  "stops directly and are not routes. ")
                st.caption(route_note
                           + "Numbers are the visiting order. Hover a stop for its name and arrival time. "
                           "Map tiles load from OpenStreetMap over the network -- give it a moment on first load.")

        free_share = result.get("free_entry_share")
        if free_share is not None:
            st.caption(
                f"{free_share:.0%} of these stops are free to enter. "
                "Restaurants, bars and shops carry a price tier where the catalogue has "
                "one; for sights RoamWise knows whether a place charges admission, but "
                "not how much."
            )

        st.markdown("##### Agent narrative")
        # Which engine wrote this, said where the writing is rather than on the
        # System logs screen. A run that fell back to the template produces
        # text that reads like an answer but is the prompt echoed back, and
        # nothing on this page used to distinguish the two (issue #133).
        llm = get_orchestrator(RETRIEVAL_CONFIG).llm
        missing = fallback_reason(llm)
        if missing:
            st.error(
                f"**No model is running.** {missing}, but the client behind it could not "
                f"be started, so this text is the prompt echoed back rather than generated "
                f"prose. See the System logs screen for the reason, and treat nothing below "
                f"as model output."
            )
        else:
            st.caption(f"Written by: {describe_client(llm)}")
        st.info(result["final_plan"])
        # A generation that ran out of room stops mid-sentence and otherwise
        # reads like a finished answer, so the reader is told rather than left
        # to notice that the last day is missing (issue #125).
        if result.get("final_plan_truncated"):
            st.warning(
                "This narrative was cut short: the model reached its output limit before "
                "it finished describing the trip, so the last day or days may be missing. "
                "The itinerary above is complete and unaffected."
            )

    with tab2:
        st.subheader(f"Tourism demand forecast for {city_name}")
        # Twelve months from *now*, and stretched if necessary to hold the
        # month this trip actually starts in. Counting the horizon from the end
        # of the city's series instead put all twelve of Berlin's "next 12
        # months" in the past -- its source lags about 20 months, so the chart
        # was showing 2025 to someone planning 2026 (#161). The forecaster read
        # the right month all along; only the chart beside it did not.
        target_month = result["forecast"]["target_month"]
        fc = forecast_window(result["destination_id"], months=12, include_month=target_month)
        first, last = fc.date.min(), fc.date.max()
        fig = px.bar(fc, x="date", y="forecast_visitors", color="crowding_level",
                     color_discrete_map={"low": "#4CAF50", "medium": "#FFC107", "high": "#F44336"},
                     # Built from the window rather than asserted over it, so the
                     # title cannot outlive the thing it describes (#161).
                     title=f"Forecasted monthly visitors ({first:%b %Y} – {last:%b %Y})")
        st.plotly_chart(fig, width='stretch')

        # How far past its last real observation this city is being
        # extrapolated. It differs per city and the reader cannot see it in the
        # bars, which look equally confident either way.
        end = history_end(result["destination_id"])
        if end is not None:
            lead = (pd.Period(first, freq="M") - end).n
            st.caption(
                f"Measured demand for this city ends {end}. Every month above is a "
                f"Holt-Winters extrapolation {lead}–{lead + len(fc) - 1} months past that "
                f"last observation, so read the level as a seasonal shape rather than a "
                f"visitor count. Your trip starts in "
                f"{pd.Period(target_month, freq='M').strftime('%B %Y')}, which is the "
                f"month the Forecaster Agent's crowding call below is taken from."
            )
        st.markdown(f"**Forecaster Agent:** {result['forecast']['narrative']}")

    with tab3:
        st.subheader("What the plan was grounded in")
        st.caption("Every stop above was drawn from these retrieved records rather than generated from scratch.")
        if result["fusion_rag"]["results"]:
            for r in result["fusion_rag"]["results"]:
                via = ", ".join(r.get("retrieved_by", []))
                st.markdown(f"**{r.get('name', r['doc_id'])}** &nbsp; `via: {via or 'n/a'}`  \n{r['text']}")
                st.divider()
        else:
            # Unreachable while retrieval is healthy, but a city with no
            # matching records must say so rather than render an empty tab.
            st.warning("Nothing was retrieved for this city, so the itinerary above falls back to an "
                       "unfiltered list of its top-rated places.")

else:
    st.markdown(
        """
        Set your travel preferences in the sidebar and click **Plan my trip**.

        Under the hood, RoamWise runs four agents in sequence:
        1. **Traveler segmentation** (KMeans) classifies your preferences into an archetype.
        2. **Forecaster Agent** interprets a demand time-series model to flag crowding and recommend timing.
        3. **Fusion RAG Agent** retrieves grounded context by fusing Semantic, Graph, and Keyword search (RRF).
        4. **Router Agent** scores those POIs against your own sliders, then solves which to visit,
           on which day and at what hour, against opening hours and your time budget (TOPTW).
        """
    )
