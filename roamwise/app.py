"""
RoamWise interactive prototype (proposal deliverable #1: "An Interactive
Prototype... allowing a user to input preferences and receive a data-backed,
graph-grounded, optimized itinerary").

Run with: streamlit run app.py
"""
import math

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from agents.orchestrator import RoamWiseOrchestrator
from models.forecasting import forecast_city

st.set_page_config(page_title="RoamWise", page_icon="\U0001f9ed", layout="wide", initial_sidebar_state="expanded")


def _fit_zoom(lats: list[float], lons: list[float]) -> float:
    """Rough degree-span-to-mapbox-zoom heuristic so the map frames whatever
    the itinerary actually covers instead of a fixed zoom that's too tight
    for spread-out days and too loose for a single-block day."""
    span = max(max(lats) - min(lats), max(lons) - min(lons), 0.002)
    return min(15.0, max(11.0, 13.5 - math.log2(span * 100)))


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
                             help="How many hours per day to spend walking/visiting -- feeds the router's time budget.")
    use_real_routing = st.checkbox(
        "Use real street routing (OSRM)", value=False,
        help="Fetches real walking distances/times from a public OSRM server instead of "
             "straight-line + flat 4.5km/h. Needs internet; falls back automatically if unreachable.",
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
                                 use_real_routing=use_real_routing)

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
            for day in result["routing"]["itinerary"]:
                with st.container(border=True):
                    st.markdown(f"**Day {day['day']}** &mdash; {day['distance_km']} km, ~{day['total_minutes']} min")
                    if day["route"]:
                        for i, poi in enumerate(day["route"], 1):
                            st.markdown(f"{i}. **{poi['name']}** _{poi.get('category', '')}_")
                    else:
                        st.markdown("_No stops fit the time budget for this day._")
        with col2:
            fig = go.Figure()
            colors = px.colors.qualitative.Set2
            for day in result["routing"]["itinerary"]:
                if not day["route"]:
                    continue
                lats = [p["lat"] for p in day["route"]]
                lons = [p["lon"] for p in day["route"]]
                names = [p["name"] for p in day["route"]]
                fig.add_trace(go.Scattermapbox(
                    lat=lats, lon=lons, mode="markers+lines+text", text=names, textposition="top right",
                    name=f"Day {day['day']}", marker=dict(size=11, color=colors[(day['day'] - 1) % len(colors)]),
                ))
            if any(day["route"] for day in result["routing"]["itinerary"]):
                all_lats = [p["lat"] for day in result["routing"]["itinerary"] for p in day["route"]]
                all_lons = [p["lon"] for day in result["routing"]["itinerary"] for p in day["route"]]
                fig.update_layout(
                    mapbox=dict(style="open-street-map", zoom=_fit_zoom(all_lats, all_lons),
                                center=dict(lat=sum(all_lats) / len(all_lats), lon=sum(all_lons) / len(all_lons))),
                    margin=dict(l=0, r=0, t=0, b=0), height=450, showlegend=True,
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Map tiles load from OpenStreetMap over the network -- give it a moment on first load.")

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
