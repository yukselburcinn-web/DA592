"""
RoamWise interactive prototype (proposal deliverable #1: "An Interactive
Prototype... allowing a user to input preferences and receive a data-backed,
graph-grounded, optimized itinerary").

Run with: streamlit run app.py
"""
import math
import sys
from pathlib import Path

# Streamlit puts the *script's* directory (roamwise/) on sys.path, not the repo
# root, so the `roamwise.*` package the rest of the codebase imports would not
# resolve. Put the repo root first so there is exactly one import path for every
# module -- importing the same file under two names loads it twice (see #26).
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from roamwise.agents.orchestrator import RoamWiseOrchestrator
from roamwise.models.forecasting import forecast_city
from roamwise.optimization.travel_modes import TRAVEL_MODES

st.set_page_config(page_title="RoamWise", page_icon="\U0001f9ed", layout="wide", initial_sidebar_state="expanded")


def _clock(hour: float) -> str:
    """Fractional hour (13.5) -> wall clock ("13:30"), so the itinerary reads
    as a plan for a day rather than as an ordered list."""
    hours, minutes = divmod(int(round(hour * 60)), 60)
    return f"{hours % 24:02d}:{minutes:02d}"


# The map is drawn at a fixed height in a half-width column. Zoom in Web
# Mercator is defined against pixels, so framing an itinerary means asking how
# many pixels its bounding box needs and picking the zoom where that fits.
_MAP_HEIGHT_PX = 460
# Width is fluid (use_container_width), so Python cannot measure it. This is a
# deliberate under-estimate of the half-width column in Streamlit's wide
# layout: guessing low costs a little unused margin, guessing high pushes stops
# off the edge, and only one of those is recoverable by the user panning.
_MAP_ASSUMED_WIDTH_PX = 420
_MAP_PADDING = 0.82  # keep markers and their labels clear of the edges
_MAP_MAX_ZOOM = 16.0  # a one-stop day would otherwise zoom to street level


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


@st.cache_resource
def get_orchestrator(config: str):
    return RoamWiseOrchestrator(retrieval_config=config)


st.title("\U0001f9ed RoamWise")
st.caption("Agentic AI framework for personalized tourism forecasting and itinerary optimization -- DA592 prototype")

with st.sidebar:
    st.header("Traveler preferences")
    budget = st.slider("Budget (0 = shoestring, 1 = luxury)", 0.0, 1.0, 0.4, 0.05)
    culture = st.slider("Culture / history interest", 0.0, 1.0, 0.6, 0.05)
    nature = st.slider("Nature / outdoors interest", 0.0, 1.0, 0.4, 0.05)
    nightlife = st.slider("Nightlife interest", 0.0, 1.0, 0.3, 0.05)
    relax = st.slider("Relaxation / beach interest", 0.0, 1.0, 0.3, 0.05)
    adventure = st.slider("Adventure interest", 0.0, 1.0, 0.3, 0.05)

    st.header("Trip constraints")
    dest_options = {"Let RoamWise choose": None, "Istanbul": "IST", "Paris": "PAR", "Rome": "ROM",
                     "Barcelona": "BCN", "Amsterdam": "AMS", "Prague": "PRG", "Vienna": "VIE", "Lisbon": "LIS"}
    dest_label = st.selectbox("Destination", list(dest_options.keys()))
    destination_id = dest_options[dest_label]
    n_days = st.slider("Trip length (days)", 1, 5, 3)
    travel_month = st.selectbox(
        "Travel month (optional)",
        [None] + [f"2026-{m:02d}" for m in range(9, 13)] + [f"2027-{m:02d}" for m in range(1, 9)],
        format_func=lambda x: "Flexible / let RoamWise recommend" if x is None else x,
    )
    max_price_level = st.select_slider(
        "Max price level per stop", options=[1, 2, 3], value=3,
        format_func=lambda p: {1: "$ Budget only", 2: "$$ Budget + mid-range", 3: "$$$ No limit"}[p],
        help="Drops POIs pricier than this before routing (falls back to the unfiltered list if nothing survives).",
    )
    daily_hours = st.slider("Daily sightseeing time (hours)", 3, 12, 8,
                             help="How many hours per day to spend travelling/visiting -- feeds the router's time budget.")
    travel_mode = st.radio(
        "How are you getting around?",
        list(TRAVEL_MODES),
        format_func=lambda m: {"walking": "On foot", "driving": "By car",
                               "hybrid": "Walk nearby, drive between areas"}[m],
        help="Sets how travel between stops is costed, so a driving day can cover more ground "
             "than a walking one within the same time budget.",
    )
    use_real_routing = st.checkbox(
        "Use real street routing (OSRM)", value=False,
        help="Fetches real street distances/times from a public OSRM server (walking or driving "
             "network, matching the mode above) instead of the straight-line estimate. "
             "Needs internet; falls back automatically if unreachable.",
    )

    st.header("Retrieval architecture")
    config = st.radio(
        "Fusion RAG configuration",
        ["fusion", "hybrid", "standard"],
        format_func=lambda c: {
            "fusion": "Fusion RAG (Semantic + Graph + Keyword)",
            "hybrid": "Hybrid RAG (Semantic + Keyword only)",
            "standard": "Standard prompting (no retrieval)",
        }[c],
        help="Switch this to see how the comparative-analysis conditions from the proposal change the output.",
    )

    run = st.button("Plan my trip", type="primary", use_container_width=True)

preferences = {"budget": budget, "culture": culture, "nature": nature,
               "nightlife": nightlife, "relax": relax, "adventure": adventure}

if run:
    orch = get_orchestrator(config)
    with st.spinner("Agents at work: segmenting traveler, forecasting demand, retrieving grounded context, routing..."):
        result = orch.plan_trip(preferences, destination_id=destination_id, n_days=n_days, travel_month=travel_month,
                                 max_price_level=max_price_level, daily_minutes_budget=daily_hours * 60,
                                 use_real_routing=use_real_routing, travel_mode=travel_mode)

    city_name = orch.destinations.set_index("destination_id").loc[result["destination_id"], "city"]
    st.success(f"Plan ready: **{city_name}** for a **{result['archetype']}**")

    tab1, tab2, tab3, tab4 = st.tabs(["\U0001f4cd Itinerary", "\U0001f4c8 Demand forecast", "\U0001f50e Retrieved context", "\U0001f9e0 Agent trace"])

    with tab1:
        st.subheader(f"{n_days}-day route in {city_name}")
        if use_real_routing:
            any_real = any(d.get("used_real_routing") for d in result["routing"]["itinerary"])
            if any_real:
                st.caption("Distances/times below use real OSRM street routing.")
            else:
                st.caption("Real routing was requested but OSRM was unreachable -- showing the "
                           "straight-line + flat-walking-speed estimate instead.")
        col1, col2 = st.columns([1, 1])
        with col1:
            budget_minutes = daily_hours * 60
            for day in result["routing"]["itinerary"]:
                with st.container(border=True):
                    used = day["total_minutes"]
                    st.markdown(f"**Day {day['day']}** &mdash; {used // 60}h {used % 60:02d}m "
                                f"of your {daily_hours}h &middot; {day['distance_km']} km")
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
                                     else poi.get("category", ""))
                            st.markdown(f"{i}. {when}**{poi['name']}** _{label}_")
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
                    kind = "meal stop" if poi.get("category") == "food" else poi.get("category", "")
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
                fig.update_layout(
                    map=dict(style="open-street-map", zoom=zoom,
                             center=dict(lat=center_lat, lon=center_lon)),
                    margin=dict(l=0, r=0, t=0, b=0), height=_MAP_HEIGHT_PX,
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=0.01,
                                xanchor="left", x=0.01,
                                bgcolor="rgba(255,255,255,0.75)", borderwidth=0),
                    hoverlabel=dict(bgcolor="white", font_size=12),
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Numbers are the visiting order. Hover a stop for its name and arrival time. "
                           "Map tiles load from OpenStreetMap over the network -- give it a moment on first load.")

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
        st.subheader(f"Retrieved context ({config} configuration)")
        if result["fusion_rag"]["results"]:
            for r in result["fusion_rag"]["results"]:
                via = ", ".join(r.get("retrieved_by", []))
                st.markdown(f"**{r.get('name', r['doc_id'])}** &nbsp; `via: {via or 'n/a'}`  \n{r['text']}")
                st.divider()
        else:
            st.warning("Standard-prompting configuration retrieves nothing -- the itinerary below falls back to "
                       "an unfiltered, non-personalized list of the city's top-rated POIs.")

    with tab4:
        st.subheader("Multi-agent orchestration trace")
        st.json({
            "1_segmentation": result["segmentation"],
            "2_destination_selected": result["destination_id"],
            "3_forecast": {k: v for k, v in result["forecast"].items() if k != "narrative"},
            "4_retrieval_config": result["fusion_rag"]["config"],
            "5_n_pois_routed": sum(len(d["route"]) for d in result["routing"]["itinerary"]),
        })
else:
    st.markdown(
        """
        Set your travel preferences in the sidebar and click **Plan my trip**.

        Under the hood, RoamWise runs four agents in sequence:
        1. **Traveler segmentation** (KMeans) classifies your preferences into an archetype.
        2. **Forecaster Agent** interprets a demand time-series model to flag crowding and recommend timing.
        3. **Fusion RAG Agent** retrieves grounded context by fusing Semantic, Graph, and Keyword search (RRF).
        4. **Router Agent** geographically zones the retrieved POIs and solves each day's route (2-opt).
        """
    )
