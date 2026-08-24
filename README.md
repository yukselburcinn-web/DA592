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

Writes `evaluation/comparative_analysis_results.csv` (per-query), `evaluation/comparative_analysis_summary.csv` (aggregated) and `evaluation/comparative_analysis_significance.csv` (the pairwise tests below). It scores 55 queries across 2 cities and 7 archetypes on five metrics: multi-hop recall against a graph-computed gold set, archetype precision, grounded-entity rate, itinerary coherence (km per stop), and retrieval latency. Current summary:

| Config | Multi-hop recall | Archetype precision | Grounded | Km/stop | Latency |
|---|---|---|---|---|---|
| **Fusion RAG** | 0.162 | **0.837** | 1.000 | 0.88 | 11.3 ms |
| Hybrid RAG | 0.154 | 0.716 | 1.000 | 0.90 | 8.8 ms |
| Standard prompting | 0.000 | 0.500 | 0.000 | 0.69 | 0.0 ms |

Because a gap between two averages can be noise, each metric is also tested pairwise (Wilcoxon signed-rank across the same queries — see the **Is the lead real?** table in the app). Against Hybrid RAG, Fusion RAG is significantly better on **archetype precision** (p<0.0001), *tied by construction* on grounded entities, and **not distinguishable** on multi-hop recall (p=0.49) or km per stop (p=0.45), at roughly 2 ms more per retrieval. Archetype precision is why the app pins Fusion rather than asking the traveler to choose.

**Read multi-hop recall against its ceiling, not against 1.0.** Recall is measured at k=8 while the median query's gold set holds 32 POIs, so even a perfectly ordered retriever tops out at **0.317**. Fusion's 0.162 is 51% of what is reachable. The number is lower than the 0.278 this table carried for the previous eight-city catalogue for the same reason: that catalogue held 150 POIs per city, so its gold sets — and its ceiling — were far smaller. The two figures are not comparable, and neither is evidence about retrieval quality on its own.

The queries come in two tiers, and the split is the point: 19 are hand-written the way a traveler would phrase a question, and 36 are generated to sweep city × category cells evenly. Archetype precision holds in **both** tiers (p=0.0013 hand-written, p=0.0001 grid), so it is a property of the architecture. Multi-hop recall does not — it is indistinguishable in **both** tiers (p=0.13 hand-written, p=0.82 grid). An earlier 18-query set reported recall as a clear Fusion win; broadening and balancing the queries removed it. See [`#50`](https://github.com/yukselburcinn-web/DA592/issues/50) for how the query set is built and [`#48`](https://github.com/yukselburcinn-web/DA592/issues/48) for why some queries grade the graph retriever against its own traversal.

### Optional: real LLM narration

By default every agent's "reasoning" step uses a deterministic template (see `agents/llm_client.py`) so the whole system runs offline with zero API cost. To use live Claude narration instead:

```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-...
```

No code changes needed — `get_default_llm_client()` detects the key automatically. This also unlocks a real generative-hallucination probe in `evaluation/comparative_analysis.py::run_llm_hallucination_probe`.

### Optional: local LLM narration (no API key, Apple Silicon only)

A free alternative to the Anthropic path above ([`#54`](https://github.com/yukselburcinn-web/DA592/issues/54)): a small open-weight model (Qwen3-4B-Instruct, Apache-2.0) running entirely on-device via MLX, no key, no per-call cost.

```bash
pip install -r requirements-local-llm.txt
export ROAMWISE_LOCAL_LLM=1
```

The model (~2.1GB) downloads to your own Hugging Face cache on first use, not to this repo — see `requirements-local-llm.txt` for why. A 3-day plan takes roughly 20–50s of generation against milliseconds for the template default; see [`REPORT.md` §3.4.2](REPORT.md#342-a-live-non-template-llm-run-issues-54-56-57) for the measurements, and for the two bugs that only a real generative model could surface (issues #56 and #57). If `ANTHROPIC_API_KEY` is also set, Anthropic takes priority.

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

The proposal names Kaggle/TripAdvisor/OpenStreetMap/Wikidata as data sources. Every dataset the app ships is now derived from those sources rather than procedurally generated: `data/generate_data.py` produced the previous synthetic destinations, transport hubs and hand-written city guides, and is retained only for `user_survey.csv` (illustrative archetype training data, which no API replaces).

The catalogue covers **2 cities — Paris (400 POIs) and Berlin (300)**. That is a deliberate trade of breadth for depth: the previous eight-city set carried 150 POIs each, and per-city depth is what the retrieval comparison and the itinerary optimiser actually exercise. Coverage against an independent Wikivoyage reference list roughly doubled as a result (Paris 38% → 75%, measured on the same one-to-one matching).

- **POIs** (`pipeline/build_catalogue.py`): each city's points of interest are pulled from **Wikidata** (SPARQL, via the QLever endpoint) and cross-matched against **OpenStreetMap** (Overpass) within ~7km of the city centre, mapped onto RoamWise's ten-category taxonomy. Candidates are ranked by a fame score blending Wikidata sitelink count with Wikipedia pageviews (weights fitted against the reference list), so a city's catalogue is its best-known sights rather than an arbitrary sample of its tagged features. Entities whose English Wikipedia article carries no coordinates are rejected as concepts rather than places (OSM's `wikidata` tag sometimes points at a species, a saint or a unit of measurement instead of the building). Descriptions come from the **Wikipedia** article where one exists (639 of 700), the Wikidata one-liner otherwise (12), a category template only as a last resort (49). `avg_visit_minutes` and `price_level` remain heuristic by category — OSM doesn't reliably carry either — falling back to the OSM `fee` tag when present (78 rows). 228 of 700 rows carry real OSM opening hours. Every row records where each field came from (`description_source`, `hours_source`, `price_source`, plus `wikidata_qid` / `wikipedia_title`), and the column set is unchanged, so every downstream module keeps working unmodified.
- **Demand** (`pipeline/build_demand.py`): `demand_timeseries.csv` is Eurostat's **`tour_occ_nin2m`** series — monthly nights spent at tourist accommodation establishments (NACE I551-I553) by **NUTS 2 region**. This replaces the country-level `tour_occ_nim` proxy the project previously used, and the difference is not cosmetic: the forecaster consumes *seasonal shape*, and a country's shape is not its capital's. France peaks in August at 2.35× its annual mean while Île-de-France sits at 1.06× — Parisians leave, the coast fills — so the national series would have told a traveller to avoid Paris in August and to visit in October, when October is in fact the busier of the two. Swapping to regional data changes the crowding label in 8 of 12 months for Paris. Berlin is a NUTS 2 region in its own right (DE30), so its series is the city itself; Paris maps to Île-de-France (FR10), wider than the city but far tighter than France.

  Two honest limitations remain. The measure is **nights spent, not arrivals** — a three-night stay counts as three, and day-trippers are invisible; Eurostat publishes no monthly regional arrivals series (`tour_occ_arn2m` does not exist), so this is the only measure available at this granularity, and the column is still named `visitors`. And **regional coverage lags unevenly**: Île-de-France runs to 2025-12, Berlin only to 2024-12, because Germany's regional returns are slow. `agents/forecaster_agent.py` computes its horizon from today rather than from the end of the series and reports `data_through`, `horizon_months_used` and `requested_month_available` alongside every forecast, so the lag is visible rather than silently returning the wrong month.

### Rebuilding the datasets

Every shipped CSV is reproducible from `roamwise/pipeline/`, which reads its city list from `common.CITIES`. The scripts import each other by module name, so run them from that directory:

```bash
cd roamwise/pipeline && python build_catalogue.py PAR BER
```

That writes `../data/poi.csv`. `build_destinations.py`, `build_transport.py`, `build_demand.py` and `city_guide.py --write` produce the other four files, and `gold_list.py PAR BER --catalogue ../data/poi.csv` scores the catalogue against an independent Wikivoyage reference list. HTTP responses are cached under `roamwise/.cache/` (gitignored), so a re-run after a tuning change costs nothing.

**These five files have to move together** -- `destinations.csv`, `poi.csv`, `transport.csv`, `demand_timeseries.csv` and `city_guides/`. A half-migrated set does not start the app: two destinations against eight guide files raises `KeyError` in `retrieval/corpus.py`, and two destinations against eight demand series makes `forecast_city` fail on "less than two full seasonal cycles". Rebuild `knowledge_graph.gml` afterwards with `python -m roamwise.knowledge_graph.build_graph`.

The previous synthetic generators (`data/generate_data.py`) and the earlier single-source fetchers (`data/fetch_real_pois.py`, `data/fetch_real_demand.py`) are kept for history; they produce the old eight-city catalogue and are no longer what ships.

See `REPORT.md` for the fuller rationale behind the original synthetic-data decision and other documented gaps.
