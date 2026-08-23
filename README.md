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

Open the URL Streamlit prints (default `http://localhost:8501`). Set your preferences in the sidebar, optionally pin a destination, choose how you're getting around, and click **Plan my trip**. Planning always runs the full Fusion RAG stack; the measurement behind that choice is on the System logs screen's **Results** tab. The router balances the trip so every day is filled close to your daily sightseeing budget rather than leaving some days empty, gives every day at least two meal stops timed for lunch and dinner, and costs travel between stops according to the selected mode -- on foot, by car, or walking short hops and driving between neighbourhoods. It respects each POI's opening hours by default, and can optionally fetch real OSRM street-network distances/times ("Use real street routing" checkbox) instead of the straight-line estimate -- see [`REPORT.md` §5](REPORT.md#5-known-limitations--honest-gaps-vs-the-proposal) for why that's opt-in rather than the default.

The sidebar's **System logs** page is the operator-facing view, in two tabs. **Logs** records every agent step the pipeline runs with its duration, filterable by level and text and downloadable as a `.log` file; it replaced the old "Agent trace" tab, which put a raw JSON dump of orchestrator internals in the middle of the traveler's itinerary. **Results** shows the comparative analysis across all three retrieval architectures — recall, precision, grounding, itinerary coherence and latency — and can re-run it live; it replaced the sidebar radio that used to ask travelers to pick a retrieval configuration themselves.

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

The results are committed and rendered in the app on **System logs → Results**, so you only need this if you want to recompute them from the command line (the tab can also re-run it live):

```bash
cd roamwise
python evaluation/comparative_analysis.py
```

Writes `evaluation/comparative_analysis_results.csv` (per-query), `evaluation/comparative_analysis_summary.csv` (aggregated) and `evaluation/comparative_analysis_significance.csv` (the pairwise tests below). It scores 51 queries across 8 cities and 7 archetypes on five metrics: multi-hop recall against a graph-computed gold set, archetype precision, grounded-entity rate, itinerary coherence (km per stop), and retrieval latency. Current summary:

| Config | Multi-hop recall | Archetype precision | Grounded | Km/stop | Latency |
|---|---|---|---|---|---|
| **Fusion RAG** | 0.278 | **0.917** | 1.000 | 1.32 | 22.9 ms |
| Hybrid RAG | 0.278 | 0.749 | 1.000 | 1.35 | 9.4 ms |
| Standard prompting | 0.000 | 0.475 | 0.000 | 1.35 | 0.00 ms |

Because a gap between two averages can be noise, each metric is also tested pairwise (Wilcoxon signed-rank across the same queries — see the **Is the lead real?** table in the app). Against Hybrid RAG, Fusion RAG is significantly better on **archetype precision** (p<0.0001), *tied by construction* on grounded entities, and **not distinguishable** on multi-hop recall (p=0.91) or km per stop (p=0.54), at roughly 13 ms more per retrieval. Archetype precision is why the app pins Fusion rather than asking the traveler to choose.

The queries come in two tiers, and the split is the point: 19 are hand-written the way a traveler would phrase a question, and 32 are generated to sweep city × category cells evenly. Archetype precision holds in **both** tiers (p=0.001 hand-written, p=0.0003 grid), so it is a property of the architecture. Multi-hop recall does not — it favours Fusion only on the hand-written tier (p=0.044) and slightly favours Hybrid on the grid (p=0.066). An earlier 18-query set reported recall as a clear Fusion win; broadening and balancing the queries removed it. See [`#50`](https://github.com/yukselburcinn-web/DA592/issues/50) for how the query set is built and [`#48`](https://github.com/yukselburcinn-web/DA592/issues/48) for why some queries grade the graph retriever against its own traversal.

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

### Optional: Prophet forecasting comparison

`models/forecasting.py` (the default) uses Holt-Winters. `models/forecasting_prophet.py` is a Prophet-based alternative with the same `forecast_city()` output shape, added to empirically test the Holt-Winters-by-default rationale rather than just assert it. See [`REPORT.md` §3.2](REPORT.md#32-predictive--segmentation-models-the-tools) for the MAE/RMSE backtest results. To reproduce it:

```bash
pip install -r roamwise/requirements-prophet.txt
python roamwise/evaluation/forecasting_comparison.py
```

## Project layout

```
roamwise/
  data/                  dataset + fetchers (destinations, POIs, transport, demand, city guides -- see Data note)
  knowledge_graph/       NetworkX-based Graph-RAG substrate (build_graph.py)
  models/                forecasting.py (Holt-Winters demand model), forecasting_prophet.py (optional), segmentation.py (KMeans)
  retrieval/             semantic_search.py, keyword_search.py, graph_search.py, fusion.py (RRF)
  optimization/          routing.py (nearest-neighbor + 2-opt daily route solver, day balancing), travel_modes.py (walking/driving/hybrid), osrm_client.py
  agents/                ForecasterAgent, FusionRAGAgent, RouterAgent, orchestrator.py
  evaluation/            comparative_analysis.py + saved results
  tests/                 pytest smoke tests covering every module
  app.py                 Streamlit entry point (declares the pages, routes between them)
  views/                 the pages themselves: itinerary.py (traveler-facing), system_logs.py
  app_logging.py         in-process log buffer the System logs page reads
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
FusionRAGAgent  ──►  Semantic (sentence-transformers + FAISS) + Graph (NetworkX traversal) + Keyword (BM25)
      │              fused via Reciprocal Rank Fusion
      ▼
RouterAgent  ──►  POIZoner (KMeans day-zones) + 2-opt route solver
      │
      ▼
Synthesized day-by-day itinerary + narrative
```

## Data note

The proposal names Kaggle/TripAdvisor/OpenStreetMap/Wikidata as data sources. `data/generate_data.py` procedurally generates destinations, transport hubs, user-survey archetypes and city guides (8 cities, 16 transport hubs, hand-written city guides) — these stay curated/synthetic since they're either hand-picked real-world facts (city/transport coordinates) or illustrative training data (survey archetypes), not something an API replaces. `poi.csv` and `demand_timeseries.csv`, however, are now sourced from real, live APIs:

- **POIs** (`data/fetch_real_pois.py`): each city's 150 points of interest are pulled live from **OpenStreetMap** via the Overpass API (tourist attractions, museums, historic sites, places of worship, parks, beaches, markets and nightlife venues within ~7km of the city center), mapped onto RoamWise's existing category taxonomy. Candidates are ranked within each category by **Wikidata** sitelink count before the cut, so a city's 150 are its best-known sights rather than an arbitrary sample of its tagged features — see `REPORT.md` §3.1 for what that changed. Entities whose English Wikipedia article carries no coordinates are rejected as concepts rather than places (OSM's `wikidata` tag sometimes points at a species, a saint or a unit of measurement instead of the building). Descriptions come from the **Wikipedia** article where one exists, the Wikidata one-liner otherwise; `popularity_score` blends sitelink count with 2025 Wikipedia pageviews. `avg_visit_minutes` and `price_level` remain heuristic by category — OSM doesn't reliably carry either — falling back to the OSM `fee` tag when present. Every row records where each field came from (`description_source`, `hours_source`, `price_source`, plus `wikidata_qid` / `wikipedia_title`), and the original columns are unchanged, so every downstream module keeps working unmodified.
- **Demand** (`data/fetch_real_demand.py`): `demand_timeseries.csv` is Eurostat's `tour_occ_nim` series — monthly nights spent by non-resident tourists in accommodation establishments (NACE I551-I553) — for each destination's **country** (2019 to the latest available month). This is a real, unmodified, monthly time series (including the actual COVID-era collapse and recovery), not synthetic — but it's a country-level number used as each city's demand proxy, not a true city-level series. A free, unauthenticated, monthly, per-city tourism-demand API covering all 8 cities (including Istanbul) doesn't exist; Eurostat does publish a city-level series (`urb_ctour`, Urban Audit) covering all 8 cities, but only annually, which can't drive `models/forecasting.py`'s monthly-seasonality Holt-Winters model without redesigning that model — out of scope here (see `BACKLOG.md` issue #3, forecasting model). This simplification is deliberate and documented rather than glossed over, in the same spirit as `REPORT.md` §5.

See `REPORT.md` for the fuller rationale behind the original synthetic-data decision and other documented gaps.
