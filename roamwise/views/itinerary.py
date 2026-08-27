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

from roamwise.agents.orchestrator import RoamWiseOrchestrator
from roamwise.models.forecasting import forecast_city
from roamwise.optimization.travel_modes import TRAVEL_MODES

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _clock(hour: float) -> str:
    """Fractional hour (13.5) -> wall clock ("13:30"), so the itinerary reads
    as a plan for a day rather than as an ordered list."""
    hours, minutes = divmod(int(round(hour * 60)), 60)
    return f"{hours % 24:02d}:{minutes:02d}"


# The map is drawn at a fixed height in a half-width column. Zoom in Web
# Mercator is defined against pixels, so framing an itinerary means asking how
# many pixels its bounding box needs and picking the zoom where that fits.
_MAP_HEIGHT_PX = 460
# Width is fluid (use_container_width), so Python cannot measure it, and the
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


def _level_slider(label: str, default: str, **kwargs) -> float:
    choice = st.select_slider(label, options=_LEVEL_NAMES, value=default, **kwargs)
    return PREFERENCE_LEVELS[choice]


with st.sidebar:
    st.header("Traveler preferences")
    budget = _level_slider("Budget", "Medium",
                            help="Economical / Medium / Luxury")
    culture = _level_slider("Culture / history interest", "High")
    nature = _level_slider("Nature / outdoors interest", "Medium")
    nightlife = _level_slider("Nightlife interest", "Low")
    relax = _level_slider("Relaxation / beach interest", "Low")
    adventure = _level_slider("Adventure interest", "Low")

    st.header("Trip constraints")
    dest_options = _destination_options()
    dest_label = st.selectbox("Destination", list(dest_options.keys()))
    destination_id = dest_options[dest_label]
    n_days = st.slider("Trip length (days)", 1, 5, 3)
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
    travel_mode = st.radio(
        "How are you getting around?",
        list(TRAVEL_MODES),
        format_func=lambda m: {"walking": "On foot", "driving": "By car",
                               "hybrid": "Walk nearby, drive between areas"}[m],
        help="Sets how travel between stops is costed, so a driving day can cover more ground "
             "than a walking one within the same time budget.",
    )
    use_real_routing = st.checkbox(
        "Use real street routing", value=False,
        help="Measures each leg along the real street network (walking or driving, matching "
             "the mode above) instead of as a straight line. Runs offline from OpenStreetMap "
             "data shipped with the app -- no server, no wait.",
    )

    run = st.button("Plan my trip", type="primary", use_container_width=True)

preferences = {"budget": budget, "culture": culture, "nature": nature,
               "nightlife": nightlife, "relax": relax, "adventure": adventure}

if run:
    orch = get_orchestrator(RETRIEVAL_CONFIG)
    with st.spinner("Agents at work: segmenting traveler, forecasting demand, retrieving grounded context, routing..."):
        result = orch.plan_trip(preferences, destination_id=destination_id, n_days=n_days,
                                 start_date=start_date,
                                 daily_minutes_budget=daily_hours * 60,
                                 day_start_hour=(None if day_start_hour is None
                                                 else float(day_start_hour)),
                                 use_real_routing=use_real_routing, travel_mode=travel_mode)

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
        if use_real_routing:
            any_real = any(d.get("used_real_routing") for d in result["routing"]["itinerary"])
            if any_real:
                st.caption("Distances/times below follow the real street network.")
            else:
                st.caption("Real routing was requested but no street network covers this city -- "
                           "showing the straight-line + flat-walking-speed estimate instead.")
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
                            st.markdown(f"{i}. {when}**{poi['name']}** _{label}_{free}")
                    else:
                        st.markdown("_No stops fit the time budget for this day._")
        with col2:
            fig = go.Figure()
            colors = px.colors.qualitative.Bold
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
                fig.add_trace(go.Scattermap(
                    lat=[p["lat"] for p in stops], lon=[p["lon"] for p in stops],
                    mode="markers+lines+text",
                    text=[str(i) for i in range(1, len(stops) + 1)],
                    textposition="middle center",
                    textfont=dict(size=11, color="white", family="Arial Black"),
                    hovertext=hover, hoverinfo="text",
                    name=f"Day {day['day']}",
                    marker=dict(size=20, color=color, opacity=0.95),
                    line=dict(width=3, color=color),
                ))
            if any(day["route"] for day in result["routing"]["itinerary"]):
                all_lats = [p["lat"] for day in result["routing"]["itinerary"] for p in day["route"]]
                all_lons = [p["lon"] for day in result["routing"]["itinerary"] for p in day["route"]]
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
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Numbers are the visiting order. Hover a stop for its name and arrival time. "
                           "Map tiles load from OpenStreetMap over the network -- give it a moment on first load.")

        free_share = result.get("free_entry_share")
        if free_share is not None:
            st.caption(
                f"{free_share:.0%} of these stops are free to enter. "
                "RoamWise knows whether a place charges admission, but not how much -- "
                "so restaurants and nightlife are never separated by price."
            )

        st.markdown("##### Agent narrative")
        st.info(result["final_plan"])

    with tab2:
        st.subheader(f"Tourism demand forecast for {city_name}")
        fc = forecast_city(result["destination_id"], horizon_months=12)
        fig = px.bar(fc, x="date", y="forecast_visitors", color="crowding_level",
                     color_discrete_map={"low": "#4CAF50", "medium": "#FFC107", "high": "#F44336"},
                     title="Forecasted monthly visitors (next 12 months)")
        st.plotly_chart(fig, use_container_width=True)
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
