"""Geographical hallucination rate on the narratives RoamWise actually writes
(issue #132).

The proposal names three evaluation targets; two of them were measured and
this one was not. What stood in for it was `grounded_entity_rate` in
`comparative_analysis.py`, which is `1.0 if poi_results else 0.0` -- "did
retrieval return a row". That is 1.0 by construction for every
retrieval-based config, which is why the summary table reports 1.000 / 1.000 /
0.000 and the Wilcoxon test on that column has to answer "identical". It is a
structural risk proxy; it is not a hallucination measurement, and the code's
own comment said so.

It could not have been one, either, for a reason that has nothing to do with
the metric: `run_comparative_analysis()` routes with `narrate=False`, so the
experiment produces no generated text at all. **You cannot measure invention
in a run that invents nothing.** This module is the part that generates, and
then grades what came back.

What is measured
----------------
Every place name the narrative uses is classified against the catalogue and
against what the model was shown:

- `grounded`   -- the name appears in the prompt the model was given.
- `wrong_city` -- a real catalogue place, in the *other* city. This is
                  geographical hallucination in the proposal's own sense.
- `unshown`    -- a real catalogue place in this city that was not in the
                  prompt: either an invention that happens to be right, or a
                  leak from a candidate list the narrator should not have had.

`grounded` deliberately includes off-route names that the prompt itself
introduced. The Humboldt Forum's own description says it stands on Museum
Island, so a narrative correctly describing that stop says "Museum Island"
while recommending nothing off-route. Counting that as hallucination would
make the metric fire hardest on correct output. This is the same distinction
`tests/test_orchestration.py::ungrounded_places` draws, and it is kept
deliberately identical -- one definition of "the model was never shown this",
used by both the per-run test and the systematic measurement.

Matching is on word boundaries, never substrings. "Bar" is inside "barrier"
and "station" inside "television station" -- the latter is how a TV channel
became a Paris landmark in #63. The version of the probe this replaces used
`any(name in line or line in name for name in known_names)`, and the
`line in name` direction is worse than a plain substring test: a short line
matches many long names at once, so the match rate inflates and hallucination
reads *lower* than it is (CLAUDE.md gotcha 8).

What it costs, and why it is not 201 calls
------------------------------------------
67 queries x 3 configs is 201 rows, but not 201 distinct prompts. The
narrative is a function of the prompt, so two queries that route to the same
stops hand the narrator the same facts and the same opportunity to invent.
Measured on the committed catalogue: 52 distinct prompts for fusion, 53 for
hybrid, and 14 for standard -- standard prompting retrieves nothing at all, so
its candidate set is `city_pois(dest)[:top_k]` regardless of the question, and
67 rows collapse onto 2 cities x 7 archetypes. 106 distinct prompts in total.

Generating once per distinct prompt and weighting each result by how many rows
it stands for is not a sample: under deterministic decoding it reproduces
exactly the number a 201-call run would report, for 53% of the quota. The
weights are written into the per-row CSV so the arithmetic is checkable.

Generations are additionally cached to `hallucination_generations.json`,
keyed by (model, system, prompt), and that file is committed. CI cannot run
this -- it needs a key and a live endpoint -- so the cache is what makes the
committed CSV reproducible by someone who does not want to spend their own
quota re-deriving it.
"""
import hashlib
import json
import os
import re
from pathlib import Path

import pandas as pd

from roamwise.agents.llm_client import (
    LLMRequestFailed, TemplateLLMClient, describe_client, get_default_llm_client)
from roamwise.agents.orchestrator import SYNTHESIS_SYSTEM, synthesis_prompt
from roamwise.knowledge_graph.build_graph import GraphIndex

HERE = Path(__file__).parent
CACHE_PATH = HERE / "hallucination_generations.json"
RESULTS_CSV = HERE / "hallucination_results.csv"
SUMMARY_CSV = HERE / "hallucination_summary.csv"
PROBE_CSV = HERE / "llm_hallucination_probe.csv"


class TemplateClientRefused(RuntimeError):
    """Raised rather than returning a number, when no real model is configured.

    A run that fell back to `TemplateLLMClient` reports a hallucination rate of
    0.0 -- the template returns the prompt verbatim, so every place it "names"
    was in the prompt by definition. That is a perfect score produced by having
    no model at all, and it is indistinguishable in a CSV from a genuinely
    well-grounded one. #133 made the fallback audible; this makes it fatal for
    the one measurement it would silently invalidate.
    """


# ---------------------------------------------------------------- gazetteer

def gazetteer(idx: GraphIndex = None) -> dict:
    """Every place name the catalogue knows, mapped to the cities holding it.

    POIs, transport hubs and the cities themselves. A name maps to a *set*
    because two cities may legitimately carry the same one, and a name that is
    ambiguous cannot be evidence of being in the wrong city.
    """
    idx = idx or GraphIndex()
    places = {}
    for _, data in idx.g.nodes(data=True):
        name = (data.get("name") or "").strip()
        if not name or data.get("type") not in ("POI", "Transport", "City"):
            continue
        city = data.get("destination_id") or data.get("city_id") or data.get("id")
        key = _normalise(name)
        places.setdefault(key, {"name": name, "cities": set()})
        if data.get("type") == "City":
            # A city node's own name belongs to that city; the id is the code.
            places[key]["cities"].add(data.get("destination_id") or _city_code(idx, name))
        elif city:
            places[key]["cities"].add(city)
    return places


def _city_code(idx: GraphIndex, city_name: str):
    for node, data in idx.g.nodes(data=True):
        if data.get("type") == "City" and data.get("name") == city_name:
            return node
    return None


def _normalise(name: str) -> str:
    """Lower-cased, whitespace-collapsed, trailing dots removed.

    The trailing dot matters: the catalogue holds a Berlin restaurant called
    "Kila.", and a name ending in punctuation cannot be anchored with `\\b` --
    `\\b` after "." requires a word character next, which never follows it at
    the end of a sentence. Normalising the name and anchoring on
    "not a word character" instead makes the dot irrelevant either way.
    """
    return re.sub(r"\s+", " ", name.strip()).rstrip(".").lower()


def places_pattern(places: dict) -> re.Pattern:
    """One alternation over every known name, longest first.

    Longest-first is what makes the match unambiguous: "Museum of Asian Art"
    must win over "Asian Art", and "Academy of Arts, Berlin" over "Berlin".
    Tokens are joined with `\\s+` so a name that wraps across a line still
    matches, and the anchors are lookarounds rather than `\\b` for the reason
    in `_normalise`.

    The hyphen counts as part of a word here, so "station" does not match
    inside "station-house" and "Berlin" does not match inside "Berlin-Mitte" --
    a hyphenated compound names a different place, which is precisely the
    error class gotcha 8 is about. The apostrophe deliberately does not: a
    narrative saying "Berlin's cafe culture" is naming Berlin.
    """
    alternatives = sorted(places, key=len, reverse=True)
    body = "|".join(r"\s+".join(re.escape(tok) for tok in name.split()) for name in alternatives)
    return re.compile(rf"(?<![A-Za-z0-9-])(?:{body})(?![A-Za-z0-9-])", re.IGNORECASE)


def places_named(text: str, places: dict, pattern: re.Pattern) -> list:
    """Canonical names of every catalogue place `text` mentions, in order.

    De-duplicated: a narrative that names the same stop twice has named one
    place. The rate is over distinct places, not over mentions, or a model that
    repeats itself would score differently from one that does not while
    inventing exactly as much.
    """
    seen, out = set(), []
    for match in pattern.finditer(text or ""):
        key = _normalise(match.group(0))
        if key in places and key not in seen:
            seen.add(key)
            out.append(key)
    return out


# ------------------------------------------------------------- the metric

def classify(narrative: str, prompt: str, destination_id: str,
             places: dict, pattern: re.Pattern) -> dict:
    """Grade one narrative against the prompt that produced it."""
    named = places_named(narrative, places, pattern)
    shown = set(places_named(prompt, places, pattern))

    grounded, wrong_city, unshown = [], [], []
    for key in named:
        if key in shown:
            grounded.append(places[key]["name"])
        elif destination_id not in places[key]["cities"]:
            wrong_city.append(places[key]["name"])
        else:
            unshown.append(places[key]["name"])

    # Two rates, deliberately not summed into one.
    #
    # The proposal asks for a *geographical* hallucination rate, and the issue
    # defines it as names that are (a) not in the catalogue or (b) not in this
    # city. `wrong_city` is (b), measured exactly. Adding `unshown` to it would
    # report a different, stricter thing under the proposal's name: every
    # `unshown` place found in the first run was real and in the right city --
    # "Ile de la Cite", "Pariser Platz", "Latin Quarter" -- so calling them
    # geographical hallucinations would be false. They are the #56 failure
    # instead: the system prompt says "never mention or suggest any other
    # place", and the model did anyway. Correct, and still ungrounded: nothing
    # in the pipeline put those names there or checked them.
    #
    # (a) is NOT measured here. Finding a name the catalogue has never heard of
    # needs entity extraction over open text, which this module deliberately
    # does not attempt -- everything above is exact lookup against a gazetteer.
    # `run_zero_context_probe` gives the only signal available on it, by
    # counting lines a no-context answer produced that match nothing.
    return {
        "places_named": len(named),
        "grounded": len(grounded),
        "wrong_city": len(wrong_city),
        "unshown_same_city": len(unshown),
        # None, not 0.0, when the narrative named no place at all: a rate over
        # an empty denominator is not 0% hallucination, it is no measurement.
        "geographical_hallucination_rate": (len(wrong_city) / len(named)) if named else None,
        "ungrounded_mention_rate": (len(unshown) / len(named)) if named else None,
        "wrong_city_names": "; ".join(wrong_city),
        "unshown_names": "; ".join(unshown),
    }


# ------------------------------------------------------------ generation

def _cache_key(model: str, system: str, prompt: str) -> str:
    return hashlib.sha1(f"{model}\x00{system}\x00{prompt}".encode()).hexdigest()


def load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {}


def save_cache(cache: dict) -> None:
    CACHE_PATH.write_text(json.dumps(cache, indent=1, sort_keys=True, ensure_ascii=False) + "\n")


def require_real_model(llm=None):
    """The configured client, or a refusal. Never a silent template run."""
    llm = llm or get_default_llm_client()
    if isinstance(llm, TemplateLLMClient):
        raise TemplateClientRefused(
            "hallucination measurement needs a generative model; the offline template "
            "returns the prompt verbatim and would report 0.0 by construction. Set "
            "ROAMWISE_LLM (see .env.example) and re-run."
        )
    return llm


def generate(llm, system: str, prompt: str, cache: dict) -> tuple:
    """One generation, cached by (model, system, prompt). Returns (text, hit)."""
    model = getattr(llm, "model", type(llm).__name__)
    key = _cache_key(model, system, prompt)
    if key in cache:
        return cache[key]["text"], True
    completion = llm.complete_verbose(system=system, prompt=prompt)
    cache[key] = {"text": completion.text, "model": model,
                  "truncated": completion.truncated}
    return completion.text, False


# ------------------------------------------------------------- the run

# The forecast line is pinned rather than generated.
#
# In production it is a generated sentence, and generating it here would add
# one call per row for a sentence that cannot contain a place name -- it talks
# about a month and a crowding level, and the whole hallucination question is
# about places. Pinning it also isolates the variable under study: the three
# configs differ in what retrieval put in front of the router, and a forecast
# narrative that varied per row would let the model's own wording move the
# result. The month and level are still the real ones for that city.
FORECAST_TEMPLATE = ("Expected demand in {month} is {level} for this city, "
                     "so plan around the busier hours.")


def _forecast_line(forecaster, destination_id: str, travel_month: str = None) -> str:
    fc = forecaster.run(destination_id, travel_month=travel_month, narrate=False)
    return FORECAST_TEMPLATE.format(month=fc["target_month"], level=fc["crowding_level"])


def build_prompts(top_k: int = 8) -> pd.DataFrame:
    """One row per (query, config), carrying the prompt the narrator would see.

    No model is called here, so this is free and testable offline. The dedup
    the measurement depends on is computed from this frame.
    """
    # Imported inside the function: comparative_analysis builds its query grid
    # at import time, which is slow enough that a test touching only the
    # matching helpers above should not pay for it.
    from roamwise.agents.forecaster_agent import ForecasterAgent
    from roamwise.agents.router_agent import RouterAgent
    from roamwise.evaluation.comparative_analysis import (
        CHAIN_DATE, CONFIGS, TEST_QUERIES, _hub_or_none)
    from roamwise.retrieval.fusion import FusionRetriever

    idx = GraphIndex()
    retriever, router = FusionRetriever(), RouterAgent(idx)
    forecaster = ForecasterAgent(llm=TemplateLLMClient())  # never narrates; see above
    destinations = pd.read_csv(HERE.parent / "data" / "destinations.csv").set_index("destination_id")

    rows = []
    for query_id, query in enumerate(TEST_QUERIES):
        anchor = ({"arrival_hub_id": _hub_or_none(query), "start_date": CHAIN_DATE}
                  if query.chain_anchor else {})
        forecast = _forecast_line(forecaster, query.destination_id)
        for config in CONFIGS:
            results = retriever.retrieve(query.text, config=config,
                                         destination_id=query.destination_id,
                                         archetype=query.archetype, top_k=top_k, **anchor)
            ids = [r["poi_id"] for r in results if r.get("type") == "poi"]
            candidates = [idx.g.nodes[pid] | {"poi_id": pid} for pid in ids]
            if not candidates:  # standard prompting: no retrieval, unfiltered fallback
                candidates = idx.city_pois(query.destination_id)[:top_k]
            routing = router.run(query.destination_id, candidates, n_days=1,
                                 narrate=False, use_real_routing=False)
            city = destinations.loc[query.destination_id, "city"]
            rows.append({
                "query_id": query_id, "tier": query.tier, "config": config,
                "destination_id": query.destination_id, "archetype": query.archetype,
                "prompt": synthesis_prompt(query.archetype, city, query.destination_id,
                                           forecast, routing["facts"]),
            })
    frame = pd.DataFrame(rows)
    frame["prompt_key"] = [hashlib.sha1(p.encode()).hexdigest() for p in frame.prompt]
    return frame


def run_hallucination_measurement(top_k: int = 8, llm=None) -> tuple:
    """Generate once per distinct prompt, grade, and expand back to all rows.

    Returns (per-row frame, per-config summary). Raises TemplateClientRefused
    rather than reporting a number no model produced.
    """
    llm = require_real_model(llm)
    prompts = build_prompts(top_k=top_k)
    places = gazetteer()
    pattern = places_pattern(places)
    cache = load_cache()

    graded, calls, hits = {}, 0, 0
    for key, group in prompts.groupby("prompt_key"):
        row = group.iloc[0]
        text, was_cached = generate(llm, SYNTHESIS_SYSTEM, row.prompt, cache)
        calls += 0 if was_cached else 1
        hits += 1 if was_cached else 0
        graded[key] = classify(text, row.prompt, row.destination_id, places, pattern)
    save_cache(cache)

    out = prompts.drop(columns=["prompt"]).copy()
    # How many rows this one generation stands for. Written out so the
    # weighting is auditable rather than asserted in a docstring.
    out["rows_represented"] = out.groupby("prompt_key").prompt_key.transform("size")
    for column in next(iter(graded.values())):
        out[column] = out.prompt_key.map(lambda k, c=column: graded[k][c])

    measured = out[out.places_named > 0]
    summary = measured.groupby("config").agg(
        queries=("query_id", "size"),
        distinct_prompts=("prompt_key", "nunique"),
        places_named=("places_named", "sum"),
        wrong_city=("wrong_city", "sum"),
        unshown_same_city=("unshown_same_city", "sum"),
    )
    # Proportions over *place names*, not means of per-narrative rates: a
    # narrative naming two places would otherwise weigh as much as one naming
    # twelve. n for any claim about either number is `places_named`.
    summary["geographical_hallucination_rate"] = (summary.wrong_city / summary.places_named).round(4)
    summary["ungrounded_mention_rate"] = (summary.unshown_same_city / summary.places_named).round(4)
    summary["model"] = describe_client(llm)
    print(f"{calls} generation(s), {hits} served from cache")
    return out, summary.reset_index()


def run_zero_context_probe(llm=None) -> pd.DataFrame:
    """The true standard-prompting baseline: no retrieval, no itinerary.

    One call per *city*, not per query -- the prompt depends on nothing else,
    and there are two cities in the catalogue. This is what the old
    `run_llm_hallucination_probe()` did; what it could not do was run. It was
    gated on `ANTHROPIC_API_KEY` and constructed `AnthropicLLMClient` directly,
    so the decision to go through NVIDIA (#7) left it permanently skipped --
    `llm_hallucination_probe.csv` was never written once. It asks the
    configured client now, whatever that is.
    """
    llm = require_real_model(llm)
    idx = GraphIndex()
    places = gazetteer(idx)
    pattern = places_pattern(places)
    cache = load_cache()

    from roamwise.evaluation.comparative_analysis import TEST_QUERIES
    system = "You are a travel assistant."
    rows = []
    for dest_id in sorted({q.destination_id for q in TEST_QUERIES}):
        city = idx.g.nodes[dest_id]["name"]
        prompt = f"List 5 specific points of interest to visit in {city}, one per line, name only."
        text, _ = generate(llm, system, prompt, cache)
        named = places_named(text, places, pattern)
        in_city = [k for k in named if dest_id in places[k]["cities"]]
        # Every line the model produced, whether or not it names a place the
        # catalogue knows. `named` can only ever find catalogue entries, so
        # `lines - len(named)` is the count of places it offered that this
        # knowledge base cannot vouch for either way.
        lines = [l for l in (text or "").splitlines() if l.strip()]
        rows.append({"destination_id": dest_id, "lines": len(lines),
                     "matched_catalogue": len(named), "in_this_city": len(in_city),
                     "unmatched": len(lines) - len(named)})
    save_cache(cache)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    try:
        results, summary = run_hallucination_measurement()
    except TemplateClientRefused as exc:
        raise SystemExit(f"refused: {exc}")
    results.to_csv(RESULTS_CSV, index=False)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(summary.to_string(index=False))
    probe = run_zero_context_probe()
    probe.to_csv(PROBE_CSV, index=False)
    print(probe.to_string(index=False))
