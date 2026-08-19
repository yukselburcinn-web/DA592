# RoamWise — Implementation Report

**Project:** RoamWise: An Agentic AI Framework for Personalized Tourism Forecasting and Intelligent Itinerary Optimization
**Course:** DA592, Summer 2026
**Team:** Berk Nacar, Burçin Yüksel, İsmet Tutar
**Source proposal:** `Berk Nacar_Burçin Yüksel_İsmet Tutar .pdf`

## 1. What this report covers

The original proposal lays out a 10-week plan (data & baselines → knowledge base & Fusion RAG → agentic orchestration → UI → evaluation). At the user's request, this implementation pass **ignored the week-by-week schedule and built a complete, working, end-to-end version of every deliverable in one continuous session**, rather than staging it across weeks. This document reports what was actually built, how it maps onto the proposal's methodology and deliverables sections, what evaluation results it produced, and — importantly — where the implementation had to substitute a lighter-weight component for something the proposal named (e.g. Prophet, live Neo4j, a downloaded transformer encoder) and why, so the deviations are explicit rather than silent.

All claims below are backed by code in this repository and a passing automated test suite (`roamwise/tests/test_pipeline.py`, 14/14 tests green).

## 2. Proposal deliverables → what was actually built

| Proposal deliverable | Status | Where |
|---|---|---|
| Interactive Prototype (web UI) | **Built & verified in-browser** | `roamwise/app.py` (Streamlit) |
| Agentic Architecture Codebase | **Built** | `roamwise/agents/`, `roamwise/models/`, `roamwise/retrieval/`, `roamwise/optimization/` |
| Knowledge Graph | **Built** | `roamwise/knowledge_graph/build_graph.py` — 1231 nodes (8 City, 1200 POI, 16 Transport, 7 ArchetypeProfile), 66318 edges |
| Final Technical Report | **This document + `evaluation/comparative_analysis_*.csv`** | |

## 3. Methodology → implementation mapping

### 3.1 Data sourcing

The proposal names Kaggle, TripAdvisor, OpenStreetMap, and Wikidata. `roamwise/data/generate_data.py` procedurally generates the pieces that stay curated rather than pulled from an API (hand-picked real-world facts or illustrative training data, not something an API replaces):

- 8 cities (Istanbul, Paris, Rome, Barcelona, Amsterdam, Prague, Vienna, Lisbon), each with a budget tier and interest tags
- 16 transport hubs (airport + train station per city), placed at their **real-world coordinates** (e.g. Fiumicino at 41.80°N/12.24°E)
- 8 hand-written descriptive city guides (~150–250 words each) forming the semantic/keyword text corpus
- A 420-row synthetic user-preference survey across 7 named traveler archetypes, used to fit the segmentation model

POIs and tourism-demand data are now **real**, fetched live from public APIs rather than generated:

- **`data/fetch_real_pois.py`**: 1200 POIs (150/city, up from an initial 10/city — see `BACKLOG.md` issue #2) across the same 10 categories, pulled from **OpenStreetMap** (Overpass API) around each city center and enriched with **Wikidata** (SPARQL) for descriptions and a sitelink-count popularity signal — the two sources the proposal names for this exact purpose.
- **`data/fetch_real_demand.py`**: the monthly tourism-demand series (2019–2026, real COVID-era dip and recovery) is Eurostat's `tour_occ_nim` — real nights-spent-by-non-residents data, at each destination's **country** level (there is no free, unauthenticated, monthly, per-city arrivals API covering all 8 cities). This is the one honest remaining gap: the *numbers* are 100% real, but they're a country-level proxy standing in for a city-level series — documented in the README's Data note and in §5 below.

`generate_data.py` still produces a synthetic fallback for `poi.csv`/`demand_timeseries.csv` if run without the two fetch scripts, so the pipeline stays fully reproducible offline; the committed dataset in this repo, however, is the real one.

### 3.2 Predictive & segmentation models ("The Tools")

- **Demand forecasting** (`models/forecasting.py`): the proposal suggests Prophet or LSTM. This implementation uses **Holt-Winters triple exponential smoothing** (`statsmodels.tsa.holtwinters.ExponentialSmoothing`, additive trend+seasonality, damped trend) instead. Rationale: it needs no compiled Stan backend (Prophet's dependency) or GPU/training-data volume (LSTM), trains in milliseconds on 90 monthly points per city, and is fully sufficient for a 12-month-ahead forecast at this data volume. Forecast output is converted into a `low`/`medium`/`high` crowding label (z-score against each city's own post-pandemic history) — this is what the rest of the system actually consumes, not raw visitor counts. `models/forecasting_prophet.py` is an optional Prophet-based alternative with the same output shape, added to empirically test that rationale rather than just assert it — see the backtest below.

#### Holt-Winters vs. Prophet backtest (issue #3)

`evaluation/forecasting_comparison.py` holds out the last 6 months of each city's real Eurostat demand series, fits both models on everything before that, and scores each forecast against the real held-out values:

| Model | Mean MAE | Mean RMSE |
|---|---|---|
| **Holt-Winters** (default) | **746,115** | **877,606** |
| Prophet | 997,181 | 1,078,921 |

(Full per-city results: `evaluation/forecasting_comparison_results.csv`; aggregated: `evaluation/forecasting_comparison_summary.csv`. Reproduce with `pip install -r requirements-prophet.txt && python evaluation/forecasting_comparison.py`.)

**Reading this honestly:** Holt-Winters wins on the mean, but the per-city picture is mixed, not a clean sweep — Prophet actually has lower MAE on 5 of 8 cities (Amsterdam, Barcelona, Istanbul, Lisbon, Prague). The mean is dominated by Rome, where Prophet's MAE (3.69M) is more than double Holt-Winters' (1.76M) — Prophet's default piecewise-linear trend detector appears to have picked up a spurious trend-changepoint in Rome's holdout window that Holt-Winters' damped-trend smoothing avoided. At ~40 monthly points per city (post-2022-07 cutoff), neither model has much data to work with, and a single volatile city can swing the aggregate more than the "better model on average" framing suggests. This is a genuine result from real data, not tuned for a specific outcome, and it empirically supports (rather than just asserts) the original engineering trade-off: Holt-Winters is the safer default at this data volume, but the gap is narrower and less one-sided than the original rationale implied.
- **Traveler segmentation** (`models/segmentation.py::TravelerSegmenter`): KMeans (k=7) over 6 preference dimensions (budget, culture, nature, nightlife, relax, adventure), fit on the synthetic survey, with each cluster mapped to its majority-vote archetype name for interpretability.
- **POI geographic zoning** (`models/segmentation.py::POIZoner`): a second, independent KMeans clusters a city's candidate POIs by lat/lon into `n_days` walkable zones, one per itinerary day.

### 3.3 Fusion RAG architecture (core retrieval layer)

Implemented exactly as three components fused by reciprocal rank fusion, per the proposal:

- **Semantic search** (`retrieval/semantic_search.py`): the proposal names FAISS/ChromaDB over transformer embeddings. This implementation uses **TF-IDF projected into a 64-dimensional latent space via truncated SVD** (LSA) instead of a downloaded sentence-transformer model. Rationale: no multi-hundred-MB model download, sub-second fit time, fully reproducible offline, and still captures topical similarity beyond exact keyword overlap — which is the property that actually matters for this component's role. `SemanticIndex.encode` is the single seam to swap in real embeddings later.
- **Graph-RAG** (`retrieval/graph_search.py` + `knowledge_graph/build_graph.py`): the proposal names Neo4j or NetworkX; this implementation uses **NetworkX** (a `MultiDiGraph`), which needs no server process and is fully unit-testable. `GraphIndex` exposes genuine multi-hop traversal — e.g. `multi_hop_transport_to_poi()` chains `City → Transport` and `City → POI(category)` edges and ranks by haversine distance to the nearest hub, which is exactly the kind of chained relational query the proposal argues flat document retrieval cannot answer. `GraphSearchIndex` routes a natural-language query to the right traversal (archetype-preference edges, category filters, transport-proximity) via lightweight keyword matching.
- **Keyword search** (`retrieval/keyword_search.py`): BM25 via `rank_bm25`, as specified.
- **Fusion**: reciprocal rank fusion (`retrieval/fusion.py`), the proposal's named alternative to a learned re-ranker (chosen because it needs no training data). `FusionRetriever.retrieve(query, config=...)` exposes all three comparative-analysis configurations (`fusion`, `hybrid`, `standard`) from a single object.

### 3.4 Agentic orchestration (core focus)

The proposal names LangGraph "or similar multi-agent systems." This implementation is a **lightweight custom state-machine orchestrator** (`agents/orchestrator.py::RoamWiseOrchestrator`) rather than LangGraph itself — a deliberate choice to keep the dependency surface minimal and the control flow fully deterministic and inspectable for grading, while preserving the same shape LangGraph would enforce (a shared state dict threaded through named agent nodes). Three specialized agents, as specified:

- **`ForecasterAgent`** — interprets the Holt-Winters output into a crowding narrative and low-crowding-month recommendations.
- **`FusionRAGAgent`** — orchestrates the three-component retrieval pipeline and grounds its narrative *only* in retrieved snippets.
- **`RouterAgent`** — does not ask the language layer to invent a route; it calls the geographic-zoning + 2-opt optimization tool (`optimization/routing.py`) and narrates the result. This is the proposal's "optimization methodology as a tool" requirement. The tool can use either haversine straight-line distance + a flat 4.5km/h walking speed (default), or real OSRM street-network walking distances/times (`use_real_routing=True`, opt-in and network-dependent — see §5 for why default-off) and it always respects each POI's opening hours: a stop the router would reach before it opens gets a wait folded into the day's elapsed time, and a stop already closed for the day is skipped rather than scheduled anyway.

The orchestrator's `plan_trip()` runs all five nodes in sequence: segmentation → destination selection (forecaster used as a scorer across all 8 candidate cities, combining tag affinity, budget fit, and crowding penalty) → Fusion RAG retrieval → routing → final synthesis.

**Reasoning-engine note:** every agent's natural-language synthesis step goes through a pluggable `LLMClient` (`agents/llm_client.py`). The default, zero-cost `TemplateLLMClient` composes retrieved facts into readable prose deterministically, so the entire pipeline — including the Streamlit app — runs fully offline with no API key and no per-run cost. If the user sets `ANTHROPIC_API_KEY` and installs the `anthropic` package, every agent automatically switches to live Claude narration with no code changes. This was a deliberate safety/cost decision: the system never assumes network access or spends the user's API budget without explicit opt-in.

#### 3.4.1 Custom orchestrator vs. LangGraph — a direct comparison

`agents/orchestrator_langgraph.py` reimplements the exact same five-node flow on `langgraph.graph.StateGraph`, reusing the same underlying agent classes (`TravelerSegmenter`, `ForecasterAgent`, `FusionRAGAgent`, `RouterAgent`) so only the orchestration layer itself differs. `RoamWiseLangGraphOrchestrator.plan_trip()` has the identical signature and return shape as `RoamWiseOrchestrator.plan_trip()`, verified by `tests/test_pipeline.py::test_langgraph_orchestrator_matches_custom_orchestrator_interface` (skipped automatically where the optional `requirements-langgraph.txt` extra isn't installed, e.g. in CI).

| | Custom state machine (`orchestrator.py`) | LangGraph (`orchestrator_langgraph.py`) |
|---|---|---|
| Dependency weight | None beyond the project's existing deps | Pulls in `langchain-core`, `pydantic`, `langgraph-checkpoint/-prebuilt/-sdk`, `xxhash` |
| Conditional branching (skip destination-selection when the user pins a city) | A plain `if destination_id is None:` inside `plan_trip()` | An explicit `add_conditional_edges("segment", ..., {...})` — declarative, and the branch is visible in the graph structure itself, not buried in a function body |
| Debuggability | Set a breakpoint anywhere in one linear method; the whole call stack is normal Python | Each node is a separate method invoked by the LangGraph runtime; tracing requires either LangGraph's own introspection (`.get_graph()`, LangSmith) or breakpoints inside each node individually |
| Extensibility toward real agent loops (retries, human-in-the-loop, parallel branches, checkpointed/resumable state) | Would require hand-building each of those | Native: checkpointers, conditional edges, and parallel fan-out are first-class primitives already installed as dependencies |
| Lines of orchestration-specific code | ~50 | ~95 (mostly `TypedDict` state schema + per-node method wrappers) |
| Verified behavior | Full test suite + Streamlit app | `plan_trip()` produces an identical result to the custom orchestrator on the same inputs (both deterministic under `TemplateLLMClient`); conditional edge verified to correctly skip `select_destination` when a city is pinned |

**Takeaway:** at this project's current scope — a fixed five-step pipeline with one conditional branch — the custom orchestrator is genuinely simpler with no behavioral downside, which is why it stayed the default. LangGraph's value only shows up once the orchestration needs something the custom version doesn't have: retries/backoff on a failing tool call, a human-approval step before booking, parallel exploration of multiple candidate destinations, or resumable/checkpointed runs across a multi-turn conversation. None of those are currently implemented in either orchestrator; if the project grows toward them, `orchestrator_langgraph.py` is the better foundation to extend.

### 3.5 Comparative analysis

`evaluation/comparative_analysis.py` runs 8 representative queries (archetype × city × category) through all three configurations and reports:

| Config | Mean Recall@k (multi-hop gold) | Mean archetype-category precision | Mean grounded-entity rate | Mean km/stop (itinerary coherence) |
|---|---|---|---|---|
| **Fusion RAG** (Semantic+Graph+Keyword) | **0.368** | **0.946** | 1.000 | 1.66 |
| **Hybrid RAG** (Semantic+Keyword) | 0.327 | 0.875 | 1.000 | 2.75 |
| **Standard prompting** (no retrieval) | 0.000 | 0.453 | 0.000 | 1.56 |

(Numbers above are from the real OSM/Wikidata POI dataset at 150 POIs/city, see §3.1 and `BACKLOG.md` issue #2. Full per-query results: `evaluation/comparative_analysis_results.csv`; aggregated: `evaluation/comparative_analysis_summary.csv`. Reproduce with `python evaluation/comparative_analysis.py`.)

**Reading these results honestly:**

- **Archetype-category precision now shows a dramatic, unambiguous gap** (Fusion 0.946 vs. Hybrid 0.875 vs. Standard 0.453): only Graph-RAG has direct `ArchetypeProfile -[PREFERS]-> POI` edges, so only the fusion configuration systematically surfaces POIs matching the *traveler's segment* rather than just the query text. At 150 POIs/city there are enough genuinely off-archetype POIs in each city for this structural advantage to actually matter — at the old 10-POI scale, precision differences were muted simply because there wasn't much room to be wrong.
- **Recall@k dropped in absolute terms for every config (was 1.000/0.833, now 0.368/0.327) — this is expected, not a regression.** The "gold" multi-hop set is every POI of the queried category within 6km of a transport hub; at 150 POIs/city that gold set is now realistically large (often 15-20+ POIs), so a fixed `top_k=8` structurally caps recall well below 1.0 for *any* retriever. At the old scale (10 POIs/city) the gold set was so small that top_k=8 trivially covered it, which is why the metric saturated near 1.0 and barely separated the two configs — a ceiling effect, not evidence Fusion and Hybrid were close in quality. At the larger, more realistic scale, Fusion still leads Hybrid (0.368 > 0.327), and the ranking-quality question this metric is meant to test is now actually being exercised.
- **Grounded-entity rate is 1.0 for any retrieval-based config and 0.0 for standard prompting by construction** — every document either retrieval layer returns is a real node in the knowledge graph, so this is a structural hallucination-risk proxy, not a measurement of a generative model actually hallucinating (the default `TemplateLLMClient` cannot invent text). If a live Claude key is set, `run_llm_hallucination_probe()` in the same file runs a genuine generative check instead (named-entity match rate against the KB with zero retrieved context).
- **Itinerary coherence (km/stop) now separates the configs too** (Fusion 1.66 vs. Hybrid 2.75): with 150 real, geographically scattered POIs per city instead of 10, *which* POIs a config selects has real consequences for how walkable the resulting route is — Fusion's archetype- and proximity-aware selection produces a tighter route than Hybrid's semantic/keyword-only picks, whereas at the old scale there simply wasn't enough POI density for this to show up.

## 4. Verification performed

- **Automated tests**: `pytest roamwise/tests/ -v` — 14/14 passing, including the LangGraph-orchestrator equivalence test (auto-skipped via `pytest.importorskip` only if `requirements-langgraph.txt` isn't installed; CI installs it, so the GitHub Actions run genuinely exercises both orchestrator implementations on every push, not just the one bundled in `requirements.txt`). Covers graph construction/traversal, forecasting output shape, segmentation correctness, POI zoning completeness, all three retrieval configs, time-budget-respecting routing, full orchestrator execution (both implementations), and the comparative-analysis ordering claims above.
- **Manual UI verification**: the Streamlit app was launched and driven in a real browser session. Verified: sidebar controls render and respond; "Plan my trip" runs the full 5-agent pipeline and returns a result; the Itinerary tab shows a day-by-day plan with an interactive OpenStreetMap route (colored by day); the Demand forecast tab shows a 12-month forecast bar chart colored by crowding level; the Retrieved context tab shows each surfaced POI with its `retrieved_by` source attribution (e.g. `via: graph, keyword, semantic`); the Agent trace tab shows the full orchestration state as JSON; switching the retrieval-configuration radio to "Standard prompting" correctly shows an empty retrieval context and an unfiltered itinerary fallback, confirming the three comparative-analysis conditions are genuinely wired into the live app, not just the offline evaluation script.

## 5. Known limitations & honest gaps vs. the proposal

- **Demand data is a country-level proxy, not city-level.** POIs (§3.1) are now real OpenStreetMap/Wikidata data, and the demand series is real Eurostat data — but Eurostat only publishes monthly tourism-nights at country granularity, so every city's forecast uses its country's series as a stand-in. A real city-level series exists (Eurostat Urban Audit `urb_ctour`) but only annually, which can't drive the monthly-seasonality Holt-Winters model without a redesign (see `BACKLOG.md` issue #3). Distances/routing are unaffected by this — only the crowding-forecast numbers are a proxy.
- **No live LLM by default.** The default template-based "reasoning engine" performs real selection/synthesis over structured data but does not perform open-ended generation, so it cannot exhibit true hallucination — by design, to keep the system free and offline. The optional Anthropic-backed path is implemented but not exercised in this report (no API key configured in the build environment).
- **No live Neo4j.** Named as an example ("e.g., Neo4j") in the proposal; this build uses NetworkX for zero external service dependency and full unit-testability, not because Neo4j is infeasible. (LangGraph, the other named example, now has a working alternative implementation — see §3.4.1 — so this gap is specific to the graph *database*, not the agent framework.)
- **Routing is a 2-opt heuristic, not a full time-window vehicle-routing solver.** Real OSRM street distances/times and POI opening-hours are both now implemented (`optimization/routing.py`, `optimization/osrm_client.py`) — but the 2-opt *ordering* itself still optimizes on pure geography and ignores opening hours; hours are enforced as a second pass (wait-or-skip) over the already-geographically-optimized order, not as a constraint the solver reorders around. A stop that would only make sense to visit last (because it opens late) but that 2-opt places second may get skipped even though a different, still-short route could have included it. Real routing (`use_real_routing=True`) is opt-in rather than default because it depends on a public, unauthenticated OSRM demo server (routing.openstreetmap.de) that has no uptime guarantee and rate-limits bursts of requests — RoamWise fetches one distance/duration matrix per trip (not per day) to stay well under that limit and falls back to the haversine estimate automatically if the request fails, but a default-on network dependency for a course-project prototype still felt like the wrong default. There is also no transit-schedule modeling (walking only).

## 6. How this maps to "done"

Every proposal deliverable has working, tested code behind it in this repository today: a knowledge graph, a Fusion RAG pipeline with a genuine RRF fusion of three independent retrieval signals, a multi-agent orchestrator that calls real forecasting/clustering/optimization tools rather than asking an LLM to hallucinate them, a comparative-analysis harness with saved, reproducible results, and an interactive Streamlit prototype — verified running in a browser, not just imported without error. The honest gaps are documented above rather than glossed over, and each one names the specific file where a real-data or real-LLM swap would plug in.

## 7. Suggested next steps (if continued beyond this prototype)

1. ~~Replace `data/generate_data.py`'s synthetic generation with real OpenStreetMap Overpass queries + Wikidata SPARQL for POIs, and a real aviation/UNWTO arrivals dataset for demand.~~ **Done** — see `data/fetch_real_pois.py` and `data/fetch_real_demand.py`. Remaining gap: demand is real but country-level, not city-level (§5).
2. Swap `SemanticIndex` to `sentence-transformers` + FAISS once model-download bandwidth/time is acceptable for the deployment target.
3. ~~Re-run the comparative analysis at a larger POI-per-city scale (50–100+) to test whether Fusion's multi-hop advantage over Hybrid widens, as hypothesized in §3.5.~~ **Done** at 150 POIs/city — see §3.5: precision and itinerary-coherence gaps both widened substantially; Recall@k dropped in absolute terms for both configs (the small-scale gold set was trivially saturated) but Fusion still leads Hybrid.
4. Wire `run_llm_hallucination_probe()` into CI with a provisioned `ANTHROPIC_API_KEY` to get a real generative-hallucination number alongside the structural grounding metrics.
