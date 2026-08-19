# RoamWise

[![Tests](https://github.com/yukselburcinn-web/DA592/actions/workflows/tests.yml/badge.svg)](https://github.com/yukselburcinn-web/DA592/actions/workflows/tests.yml)

An agentic AI framework for personalized tourism forecasting and intelligent itinerary optimization, built for the DA592 Summer 2026 term project.

RoamWise takes a traveler's preferences, forecasts destination crowding from a time-series model, retrieves grounded destination knowledge through a **Fusion RAG** pipeline (Semantic + Graph + Keyword search fused with reciprocal rank fusion), and hands the result to an optimization tool that builds a geographically coherent day-by-day itinerary — all orchestrated by a small multi-agent pipeline.

This repository is a working, end-to-end implementation of the architecture described in the term project proposal (`Berk Nacar_Burçin Yüksel_İsmet Tutar .pdf`), built in a single continuous pass rather than across the proposal's original 10-week schedule. See [`REPORT.md`](REPORT.md) for what was built, how it maps to the proposal, the evaluation results, and where this prototype deliberately diverges from the original plan (and why).

## Quickstart

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r roamwise/requirements.txt
cd roamwise
python data/generate_data.py      # generates destinations/transport/survey/city guides (one-time)
python data/fetch_real_pois.py    # replaces poi.csv with real OSM + Wikidata data (needs network access)
python data/fetch_real_demand.py  # replaces demand_timeseries.csv with real Eurostat data (needs network access)
python knowledge_graph/build_graph.py   # builds and caches the knowledge graph
streamlit run app.py
```

`poi.csv` and `demand_timeseries.csv` are already committed with real data, so the two `fetch_real_*` steps are only needed if you want to refresh them. If you have no network access, skip them — `generate_data.py` alone still produces a fully offline, structurally-equivalent synthetic fallback for both files.

Open the URL Streamlit prints (default `http://localhost:8501`). Set your preferences in the sidebar, optionally pin a destination, choose a Fusion RAG configuration, and click **Plan my trip**. The router respects each POI's opening hours by default, and can optionally fetch real OSRM street-network walking distances/times ("Use real street routing" checkbox) instead of the straight-line estimate -- see [`REPORT.md` §5](REPORT.md#5-known-limitations--honest-gaps-vs-the-proposal) for why that's opt-in rather than the default.

### Run with Docker

```bash
docker build -t roamwise .
docker run -p 8501:8501 roamwise
```

Open `http://localhost:8501`. The image ships the committed dataset (real POIs + real demand data, see [Data note](#data-note)) and pre-built knowledge graph, so no data-generation step is needed first.

### Run the test suite

```bash
cd roamwise
pytest tests/ -v
```

### Run the comparative analysis (Fusion RAG vs. Hybrid RAG vs. standard prompting)

```bash
cd roamwise
python evaluation/comparative_analysis.py
```

Writes `evaluation/comparative_analysis_results.csv` (per-query) and `evaluation/comparative_analysis_summary.csv` (aggregated).

### Optional: real LLM narration

By default every agent's "reasoning" step uses a deterministic template (see `agents/llm_client.py`) so the whole system runs offline with zero API cost. To use live Claude narration instead:

```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-...
```

No code changes needed — `get_default_llm_client()` detects the key automatically. This also unlocks a real generative-hallucination probe in `evaluation/comparative_analysis.py::run_llm_hallucination_probe`.

### Optional: LangGraph orchestrator

`agents/orchestrator.py` (the default) is a hand-rolled state machine. `agents/orchestrator_langgraph.py` is an alternative implementation of the exact same flow on `langgraph.graph.StateGraph`, with an identical `plan_trip()` interface. See [`REPORT.md` §3.4.1](REPORT.md#341-custom-orchestrator-vs-langgraph--a-direct-comparison) for a direct comparison. To use it:

```bash
pip install -r roamwise/requirements-langgraph.txt
```

```python
from agents.orchestrator_langgraph import RoamWiseLangGraphOrchestrator
orch = RoamWiseLangGraphOrchestrator()
```

## Project layout

```
roamwise/
  data/                  dataset + fetchers (destinations, POIs, transport, demand, city guides -- see Data note)
  knowledge_graph/       NetworkX-based Graph-RAG substrate (build_graph.py)
  models/                forecasting.py (Holt-Winters demand model), segmentation.py (KMeans)
  retrieval/             semantic_search.py, keyword_search.py, graph_search.py, fusion.py (RRF)
  optimization/          routing.py (nearest-neighbor + 2-opt daily route solver)
  agents/                ForecasterAgent, FusionRAGAgent, RouterAgent, orchestrator.py
  evaluation/            comparative_analysis.py + saved results
  tests/                 pytest smoke tests covering every module
  app.py                 Streamlit interactive prototype
```

## Architecture at a glance

```
User preferences
      │
      ▼
TravelerSegmenter (KMeans)  ──►  archetype (e.g. "Culture Enthusiast")
      │
      ▼
ForecasterAgent  ──►  Holt-Winters demand model  ──►  crowding-aware destination pick / timing
      │
      ▼
FusionRAGAgent  ──►  Semantic (TF-IDF+LSA) + Graph (NetworkX traversal) + Keyword (BM25)
      │              fused via Reciprocal Rank Fusion
      ▼
RouterAgent  ──►  POIZoner (KMeans day-zones) + 2-opt route solver
      │
      ▼
Synthesized day-by-day itinerary + narrative
```

## Data note

The proposal names Kaggle/TripAdvisor/OpenStreetMap/Wikidata as data sources. `data/generate_data.py` procedurally generates destinations, transport hubs, user-survey archetypes and city guides (8 cities, 16 transport hubs, hand-written city guides) — these stay curated/synthetic since they're either hand-picked real-world facts (city/transport coordinates) or illustrative training data (survey archetypes), not something an API replaces. `poi.csv` and `demand_timeseries.csv`, however, are now sourced from real, live APIs:

- **POIs** (`data/fetch_real_pois.py`): each city's 150 points of interest are pulled live from **OpenStreetMap** via the Overpass API (tourist attractions, museums, historic sites, places of worship, parks, beaches, markets and nightlife venues within ~7km of the city center), mapped onto RoamWise's existing category taxonomy. Where an OSM element carries a `wikidata` tag, **Wikidata** (SPARQL) supplies an English description and a sitelink count used as a real popularity proxy (`popularity_score`). `avg_visit_minutes` and `price_level` remain heuristic by category — OSM doesn't reliably carry either — falling back to the OSM `fee` tag when present. `poi.csv`'s columns are unchanged, so every downstream module (knowledge graph, retrieval, agents, tests) keeps working unmodified.
- **Demand** (`data/fetch_real_demand.py`): `demand_timeseries.csv` is Eurostat's `tour_occ_nim` series — monthly nights spent by non-resident tourists in accommodation establishments (NACE I551-I553) — for each destination's **country** (2019 to the latest available month). This is a real, unmodified, monthly time series (including the actual COVID-era collapse and recovery), not synthetic — but it's a country-level number used as each city's demand proxy, not a true city-level series. A free, unauthenticated, monthly, per-city tourism-demand API covering all 8 cities (including Istanbul) doesn't exist; Eurostat does publish a city-level series (`urb_ctour`, Urban Audit) covering all 8 cities, but only annually, which can't drive `models/forecasting.py`'s monthly-seasonality Holt-Winters model without redesigning that model — out of scope here (see `BACKLOG.md` issue #3, forecasting model). This simplification is deliberate and documented rather than glossed over, in the same spirit as `REPORT.md` §5.

See `REPORT.md` for the fuller rationale behind the original synthetic-data decision and other documented gaps.
