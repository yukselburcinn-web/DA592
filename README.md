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
python data/generate_data.py      # generates the synthetic dataset (one-time)
python knowledge_graph/build_graph.py   # builds and caches the knowledge graph
streamlit run app.py
```

Open the URL Streamlit prints (default `http://localhost:8501`). Set your preferences in the sidebar, optionally pin a destination, choose a Fusion RAG configuration, and click **Plan my trip**.

### Run with Docker

```bash
docker build -t roamwise .
docker run -p 8501:8501 roamwise
```

Open `http://localhost:8501`. The image ships the committed synthetic dataset and pre-built knowledge graph, so no data-generation step is needed first.

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
  data/                  synthetic dataset + generator (destinations, POIs, transport, demand, city guides)
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

The proposal names Kaggle/TripAdvisor/OpenStreetMap/Wikidata as data sources. This sandbox has no credentialed access to those services, so `data/generate_data.py` procedurally generates a structurally equivalent synthetic dataset (8 cities, 80 POIs, 16 transport hubs, ~6 years of monthly demand series with trend/seasonality/COVID-shock, hand-written city guides) — see `REPORT.md` for the full rationale and how to swap in real data later.
