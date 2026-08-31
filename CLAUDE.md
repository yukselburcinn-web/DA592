# CLAUDE.md

RoamWise — DA592 graduation thesis. An agentic travel-planning prototype: Streamlit UI,
Fusion RAG retrieval, a TOPTW route solver, and a Claude-or-template narrator.

Python 3.11+. `venv/bin/python` is the local interpreter; CI uses 3.11.

## Run it

```bash
cd roamwise && streamlit run app.py            # http://localhost:8501
cd roamwise && HF_HUB_OFFLINE=1 pytest tests/ -q
```

There is **no data-generation step**. Every dataset the app reads is committed, so a fresh
clone runs offline with no API keys. The knowledge graph is built in memory from the CSVs at
startup.

`data/knowledge_graph.gml` is an **export, not a runtime input**: `build_graph.py`'s
`__main__` writes it and `load_graph()` has no callers. Don't delete it — #128 settled this:
it is the concrete artifact behind the proposal's "Knowledge Graph" deliverable. Do
regenerate it (`python -m roamwise.knowledge_graph.build_graph`) whenever the catalogue or
the graph schema changes, or the deliverable silently goes stale.

Never run `data/generate_data.py` or `data/fetch_real_*` to set up: they predate the
two-city migration and overwrite five shipped files with the old eight-city dataset, which
does not start. See README § Rebuilding the datasets.

## Layout

```
roamwise/
  data/              committed catalogue + fetchers; street_network/ (per-city matrices)
  knowledge_graph/   build_graph.py — NetworkX Graph-RAG substrate
  models/            forecasting.py (Holt-Winters), segmentation.py (KMeans)
  retrieval/         semantic_search.py, keyword_search.py, graph_search.py, fusion.py (RRF)
  optimization/      toptw.py (the solver), scoring.py, routing.py, street_network.py, raptor.py
  agents/            ForecasterAgent, FusionRAGAgent, RouterAgent, orchestrator.py
  evaluation/        comparative_analysis.py + committed result CSVs
  views/             itinerary.py (traveler-facing), system_logs.py (operator-facing)
  tests/             one file per subject (#150); helpers.py holds the shared constants
```

## Where the truth lives

| Question | Read |
|---|---|
| What is open / what is next? | **GitHub Issues** (`gh issue list`) — not `BACKLOG.md` |
| What was built and what it measures | `REPORT.md` §3 |
| Known limitations and deliberate trade-offs | `REPORT.md` §5 — read before proposing architecture changes |
| How to run and rebuild things | `README.md` |

`BACKLOG.md` is a synced index that lags GitHub, and it is 81 KB. Don't read it whole and
don't cite it as current; query `gh` instead.

## Reading rules

Some committed files are large enough to be worth avoiding whole:

- `tests/` is one file per subject since #150 — `test_graph.py`, `test_models.py`,
  `test_retrieval.py`, `test_routing.py`, `test_opening_hours.py`, `test_orchestration.py`,
  `test_evaluation.py`, `test_maps.py`, `test_transit.py`, `test_arrival.py`,
  `test_crowding.py`, `test_llm.py`. Read the one you need instead of all of them, and run it
  on its own: `pytest tests/test_opening_hours.py`. The catalogue-derived constants
  (`CITY_CODES`, `MAIN_CITY`, `needs_full_city`, …) live in `tests/helpers.py`; a new test
  file imports them from there rather than re-deriving them.

  **A test file maps to a `roamwise/` subpackage, and that is what keeps parallel work
  apart** (#164). Three people split the open issues by which source files they touch, and
  the split only holds if the tests follow the same line. Two files exist for that reason:
  `test_models.py` (forecasting and segmentation had been folded into `test_retrieval.py`,
  putting the forecast work and the retrieval-tuning work in one file) and
  `test_evaluation.py` (the comparative analysis had been folded into
  `test_orchestration.py`, putting the measurement work and the orchestrator work in one).
  Put a new test where its subject lives, not where a related one happens to sit.
- `data/poi.csv`, `data/crowding.csv`, `evaluation/*_results.csv` — use `head`, `wc -l`, or
  pandas. Never `cat` them.
- `data/knowledge_graph.gml` and `data/street_network/*.npz` are blocked by a `Read` deny
  rule in `.claude/settings.json`. Inspect them through `networkx` / `numpy` if you must.
- `venv/` (1.9 GB) and `roamwise/.cache/` (2.0 GB, osmnx + GTFS downloads) are gitignored and
  skipped by search. Don't walk them with `find` either.

## Gotchas

Each of these cost someone a debugging session.

1. **Set `HF_HUB_OFFLINE=1` for anything that builds a `SemanticIndex`.**
   `sentence-transformers` makes a metadata call to the HF Hub even when the model is cached,
   and it fails intermittently. The flag just stops it asking.

2. **`TemplateLLMClient.complete()` returns the prompt verbatim.** Offline, the prompt *is*
   the output, so a prompt-design flaw is invisible and a wasted generation is free — that is
   how #56 and #57 survived so long. It is also a gift: assert on `final_plan` under the
   template client and you are asserting on exactly what a real model would be shown, with no
   download. See `test_synthesis_prompt_offers_no_place_outside_the_itinerary`.

3. **The catalogue cannot be reproduced by re-running the builder.** `build_catalogue.py`
   ranks on a 60-day rolling pageview window, so a re-run picks a different POI set and
   invalidates REPORT §3.1 and every committed evaluation CSV for unrelated reasons. Fix the
   rule *and* apply it to the committed CSV (what `pipeline/sight_filter.py` does), don't
   rebuild.

4. **Anything derived from `poi.csv` must be regenerated with it.** `city_guides/*.txt` state
   exact counts and enter the retrieval corpus as `guide::<CITY>` — a stale guide is a
   *retrievable false statement*. `city_guide.py` needs `--write`; without it it only prints.
   `evaluation/retrieval_gold.csv` also derives from the catalogue.

5. **Streamlit hot-reloads `app.py` only, not imported modules.** Edit anything under
   `agents/`, `optimization/`, `retrieval/` and the running server keeps the old code. Kill
   and restart. The symptom looks like "my fix did nothing".

6. **Don't run a model-loading script while the app is up.** Both hold ~2 GB and contend; a
   3-day plan that takes 22 s alone took over 10 minutes alongside the app.

7. **Build prompts with `"\n\n".join([...])`, not indented triple-quoted f-strings.**
   Sub-agent facts are flush-left; splicing them into an indented template leaves the
   template's own lines indented, `textwrap.dedent` cannot fix it, and Streamlit renders the
   result as a Markdown code block. This was #22 and it recurred in
   `orchestrator_langgraph.py`.

8. **Match keywords on word boundaries, never as substrings.** `"pub"` is inside
   `"public transit"`, `"bar"` inside `"barrier"`, `"station"` inside
   `"television station"` — that last one is how a TV channel became a Paris landmark (#63).
   Use `_word_re`, as `build_catalogue` and `graph_search.categories_in` do.

9. **`total_minutes` counts waiting time.** A day holding one bar at 18:00 "uses" 660 of 720
   minutes while containing one stop. Use `active_minutes`, or stops per day, when you mean
   fullness.

10. **Tests read the catalogue dynamically — never hardcode a city code.** `CITY_CODES`,
    `MAIN_CITY`, `FULL_CITIES` and `city_with_category()` are derived from the CSVs at
    import. The catalogue has changed size twice; literals turn tests red for reasons
    unrelated to the code under test.

11. **POI dicts are compared by identity (`id(p)`), not equality**, in `routing.py` — two
    distinct POIs can hold equal values. Preserve this if you refactor.

12. **Wikidata 403s without a User-Agent and 429s under load.** Go through
    `pipeline/common.http`, which sets the repo's UA and caches to `roamwise/.cache/`.

13. **`dependence_level` reports query *difficulty*, not circularity**, since #48. Its name
    predates its meaning; its docstring is right.

14. **The corpus embeddings and the knowledge graph are cached per process**
    (`semantic_search._embed_corpus`, `build_graph.shared_graph`). A second `GraphIndex()` or
    `SemanticIndex()` is nearly free, and the objects are shared — nothing may mutate them
    in place. If you rewrite the CSVs inside a live process, call `shared_graph.cache_clear()`.

15. **`SOLUTION_LIMIT = 150` in `toptw.py` is a fixed iteration budget, deliberately**, so the
    same input gives the same itinerary on any machine. Don't "tune" it to make tests faster.

16. **The graph export is guarded now, so regenerating it is not optional.**
    `test_the_committed_graph_export_describes_the_graph_the_code_builds` compares
    `data/knowledge_graph.gml` against what `build_graph()` produces today, per relation and
    per node type. Change the catalogue or the graph schema and that test goes red until you
    run `python -m roamwise.knowledge_graph.build_graph`. It exists because the file went stale
    for nineteen commits after #126 and nothing noticed (#145) — `load_graph` has no callers,
    so a wrong export breaks nothing at runtime.

    (#128's two "the code lies about itself" findings — the unrunnable `backend="neo4j"`
    branches and the "zoning + 2-opt" router label — were fixed in #145. Both are gone.)

17. **`@pytest.mark.slow` is not a hint, it decides what CI runs.** 46 of the 241 tests carry
    it: the ones that run a full trip plan, a multi-day TOPTW solve, a retriever build, or the
    comparative analysis. A pull request that touches nothing under `knowledge_graph/`,
    `retrieval/`, `agents/` or `optimization/` runs `-m "not slow"`; everything else, and
    every push to `main`, runs all 235 (#148). Re-derive both counts rather than trusting
    these two — `pytest tests/ -q --collect-only` and `-m slow --collect-only` — because they
    are the numbers that go stale first (#185). Two consequences: **add the marker when you
    add a test that plans a trip** — an unmarked heavy test silently lengthens every PR — and
    **don't rely on the PR run alone** when you touch those four directories from a branch
    that also edits them, because then you get the full suite anyway. Re-check the split with
    `pytest tests/ -q --durations=25` after adding heavy tests. Measure on an idle machine:
    a busy one doubles the numbers and the ordering is what matters.

## Working conventions

Reviewers here read commit messages closely. These are not style rules.

- **Measure before and after, and put the numbers in the commit/PR.** When a change alters
  the query set or the catalogue, measure the old code against the *new* fixture too, or the
  comparison is not paired — `git worktree add` a second checkout for this.
- **Report regressions you cause, in the same PR, with honest numbers.** #59's PR states the
  metric it made worse and argues the trade; #63 chose the parameter that regressed nothing
  over the one that maximised its headline number.
- **Comments explain *why*, and cite the issue number.** The codebase is dense with "this
  used to do X, which broke Y, so now Z (#NN)". Match that register — it is the single most
  useful property of this code.
- **Say whether you are fixing the data bug or the rule bug.** #59 deliberately left a bad
  OSM `open_hour` alone; #65 is the mirror image — the rows were transcribed correctly and
  the *rule* was wrong.
- **A regression test that passes on `main` guards nothing.** Check new tests against a
  worktree of the pre-fix branch and say in the PR which ones fail there.
- **One branch per issue**, named `issue-NN-short-slug`; PR body ends with `Closes #NN`.
- **Issues and PR bodies are written in Turkish; code, comments and docstrings in English.**
  Keep the split.
