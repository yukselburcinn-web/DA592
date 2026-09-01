# evaluation/

Every measurement this project reports, and the file it wrote.

This index exists because a cleanup pass found 14 committed result files that no
document named. None of them was junk — most are the evidence behind a paragraph
of `REPORT.md` — but the only way to tell which measured what was to open each
script and read its docstring. That is a poor property for the directory that
holds a thesis's evidence, so the mapping is written down here instead.

**Nothing in this directory is a runtime input except the three files marked
`app`.** The rest are outputs: a script runs, writes its CSV, and the number
lands in `REPORT.md`. They are committed so a reader can check a quoted figure
without re-running anything — several of the scripts need a live LLM endpoint, a
multi-hour solve, or both.

## Read by the running app

| File | Read by | Why it has to ship |
|---|---|---|
| `retrieval_gold.csv` | `comparative_analysis` at **import** (`RECOMMENDED_POIS`) | Missing it raises `FileNotFoundError` while `views/system_logs.py` is importing, so the whole System logs screen fails — not just the Results tab. Built from Wikivoyage by `pipeline/retrieval_gold.py`. |
| `comparative_analysis_results.csv` | `views/system_logs.py` → Results tab | The committed per-query run; summary and significance are recomputed from it on the fly rather than read. |
| `toptw_ceiling.csv` | `views/system_logs.py` → Results tab | A stop count has no ceiling on its own (#77); this is what it is shown against. |

## The measurements

Ordered by what they measure, not by filename.

### Retrieval — the proposal's core comparison

| Script | Question | Writes |
|---|---|---|
| `comparative_analysis.py` | Fusion vs Hybrid vs standard prompting over 68 queries, five metrics | `comparative_analysis_results.csv`, `_summary.csv`, `_significance.csv`, `_breakdown.csv`, `gold_coverage.csv` |
| `retrieval_coverage.py` | How much of the catalogue can a traveler ever be shown? | `retrieval_coverage_by_category.csv`, `retrieval_coverage_iconic.csv` |
| `category_phrase_sweep.py` | Do the query phrasings match the words the corpus uses, and what does fixing them cost? (#123) | `category_phrase_sweep.csv` |
| `graph_rag_baseline.py` | Pre-rework snapshot and the three checkpoints #126 was gated on | `graph_rag_baseline.json` |
| `chain_threshold_sweep.py` | Where the chain's two thresholds sit (#126, phase 5) | `chain_threshold_sweep.csv` |
| `chain_threshold_weight_sweep.py` | Second leg at 10 or 15 minutes, and at what RRF weight (#144) | `chain_threshold_weight_sweep.csv` |
| `quota_topk_sweep.py` | Can one quota exponent serve three trip lengths? (#143) | `quota_topk_sweep.csv` |
| `quota_plan_impact.py` | Does that exponent reach the traveler's plan? (#143) | `quota_plan_impact.csv` |

### The router

| Script | Question | Writes |
|---|---|---|
| `toptw_measurement.py` | Does a real TOPTW solver buy stops without paying in km per stop? (#72) | `toptw_measurement_results.csv`, `_summary.csv` — **not committed**; the committed pair is the baseline arm below |
| — | The pre-#72 router's own output, kept because that router no longer exists | `toptw_baseline_pre72_results.csv`, `toptw_baseline_pre72_summary.csv` |
| `toptw_scoring_ablation.py` | Does the score belong on the candidates or on the solver? (#72) | `toptw_scoring_ablation.csv` |
| `toptw_ceiling.py` | What a stop count should be read against (#77) | `toptw_ceiling.csv` |
| `iconic_coverage.py` | Do a city's best-known places reach the plan? (#122) | `iconic_coverage.csv` |
| `iconic_penalty_sweep.py` | What making a landmark expensive to drop buys, and spends (#122) | `iconic_penalty_sweep.csv` |
| `crowding_hour_measurement.py` | Does choosing the *hour* put people in emptier rooms? (#33) | `crowding_hour_results.csv`, `_summary.csv` |

### The narrator

Both need a live endpoint to *produce* generations; both re-derive their CSVs
from the committed cache without one.

| Script | Question | Writes |
|---|---|---|
| `hallucination.py` | Geographical hallucination rate on narratives the system actually writes (#132) | `hallucination_results.csv`, `_summary.csv`, `hallucination_generations.json` (the cache), `llm_hallucination_probe.csv` (the zero-context baseline) |
| `geographic_validation.py` | Are the distance and duration claims in the narrative true? (#177) | `geographic_validation.csv`, `_summary.csv` |

### Models and inputs

| Script | Question | Writes |
|---|---|---|
| `forecasting_comparison.py` | Holt-Winters vs Prophet, backtested on the real Eurostat series (#3) | `forecasting_comparison_results.csv`, `_summary.csv` |
| `survey_sensitivity.py` | How much of the personalization is a property of `user_survey.csv`? (#124) | `survey_sensitivity.csv`, `survey_sensitivity_plans.csv` |

## Running them

From `roamwise/`, e.g.:

```bash
HF_HUB_OFFLINE=1 python evaluation/comparative_analysis.py
```

Most take minutes; the ones that plan trips across a grid of cities, archetypes
and trip lengths take considerably longer. Don't run one while the Streamlit app
is up — both hold the embedding model and the graph, and they contend (CLAUDE.md
gotcha 6).
