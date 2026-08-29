"""
Baseline snapshot for the Graph-RAG rework (issue #126), and the three
checkpoints that rework is gated on.

Why a script rather than three ad-hoc measurements: #126 changes the knowledge
graph's schema, the retrieval layer's ranking and -- once its flag is flipped --
every itinerary the app produces. Each of those has a "before" that has to be
recorded *now*, because after the first phase merges there is no way back to it.
The same numbers are then re-read at three points during the work, so measuring
them by hand three times would mean three chances for the two sides of a
comparison to be taken differently.

Four sections, matching what #126 asks for:

  graph      node/edge counts by relation, build time, traversal latency.
             This is Phase 0's record and checkpoint KN-1's pass/fail.
  retrieval  fusion retrieve() latency and the composition of the top 8 --
             which retriever surfaced each result. Checkpoint KN-2 reads the
             chain retriever's share here; before #126 lands there is no chain
             retriever and the share is reported as absent rather than zero,
             because "not built yet" and "built and contributing nothing" are
             different findings.
  plan       the stop list of a fixed 3-day plan. The regression check for
             phases that must not change behaviour while the flag is off is
             "this list is identical", which needs the list written down.
  pool       stops/day, km/stop, preference match and categories/day -- the
             frontier `agents/orchestrator.py`'s RETRIEVED_POIS_PER_DAY comment
             documents. Checkpoint KN-3 (the flag gate) compares against this.

Usage:

    python -m roamwise.evaluation.graph_rag_baseline --write     # record the baseline
    python -m roamwise.evaluation.graph_rag_baseline             # measure and print
    python -m roamwise.evaluation.graph_rag_baseline --compare   # diff against the baseline
    python -m roamwise.evaluation.graph_rag_baseline --only graph --compare   # KN-1
    python -m roamwise.evaluation.graph_rag_baseline --full      # 12/15/18h grid, for KN-3

`--full` exists because the quick default (one 12h budget) is what the two
mid-work checkpoints need, while the flag gate has to answer over the same
budget grid `evaluation/toptw_measurement.py` uses. Running the wide grid at
every checkpoint would make the checkpoints expensive enough to skip.
"""
import argparse
import collections
import datetime
import json
import sys
import time
from pathlib import Path

# Run as a script and sys.path[0] is this file's directory, so the repo root --
# where the `roamwise` package lives -- never enters the path. Same bootstrap
# `comparative_analysis.py` uses, for the same reason.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd

from roamwise.agents.orchestrator import RoamWiseOrchestrator
from roamwise.agents.router_agent import RouterAgent
from roamwise.evaluation.toptw_measurement import archetype_preferences, measure
from roamwise.knowledge_graph.build_graph import GraphIndex, build_graph
from roamwise.retrieval.fusion import FusionRetriever
from roamwise.retrieval.query import archetype_query

HERE = Path(__file__).parent
BASELINE_JSON = HERE / "graph_rag_baseline.json"

# The name the chain retriever registers under in `fusion.FusionRetriever`.
# Nothing produces it yet; KN-2 starts reporting once #126 phase 4 lands.
CHAIN_RETRIEVER = "chain"

# Fixed so the snapshot is reproducible. A date rather than a month because
# opening hours are a rule over weekdays (#70) -- without one the plan is not
# determined. 2026-09-25 is a Friday, and it is the date
# `toptw_measurement.START_DATE` already uses, so the two harnesses describe
# trips on the same day.
CITIES = ["PAR", "BER"]
ARCHETYPES = ["Culture Enthusiast", "Family Traveler",
              "Nightlife Seeker", "Nature & Adventure"]
START_DATE = "2026-09-25"
_START = datetime.date.fromisoformat(START_DATE)
N_DAYS = 3
QUICK_BUDGET_HOURS = [12]
FULL_BUDGET_HOURS = [12, 15, 18]

# The plan snapshot's traveler. Written out rather than taken from an archetype
# mean because `archetype_preferences()` reads `user_survey.csv`, which #124 may
# yet change: a baseline that moves when an unrelated file moves cannot be
# compared against. The pool section *does* use the archetype means, because
# there it has to match `toptw_measurement.py`'s convention -- so that section
# records the vectors it used alongside its numbers.
PLAN_PREFERENCES = {"budget": 0.5, "culture": 0.9, "nature": 0.3,
                    "nightlife": 0.2, "relax": 0.3, "adventure": 0.3}
PLAN_CITY = "PAR"
PLAN_BUDGET_MINUTES = 12 * 60

# Tolerances for --compare. Timings are wall-clock and move with the machine, so
# they are reported but never fail a comparison; counts and plans are exact.
LATENCY_KEYS = {"build_graph_s", "graph_index_s", "multi_hop_ms",
                "retriever_init_s", "retrieve_ms"}


# ---------------------------------------------------------------------------
# graph -- Phase 0 record, checkpoint KN-1
# ---------------------------------------------------------------------------

def graph_section() -> dict:
    t = time.perf_counter()
    graph = build_graph()
    build_s = time.perf_counter() - t

    t = time.perf_counter()
    idx = GraphIndex()
    index_s = time.perf_counter() - t

    relations = collections.Counter(
        data.get("relation") for _, _, data in graph.edges(data=True))
    node_types = collections.Counter(
        data.get("type") for _, data in graph.nodes(data=True))

    # The traversal #126 rebuilt on `SERVES` edges. Timed over repeats because a
    # single call is short enough that process noise dominates it.
    reps = 50
    t = time.perf_counter()
    for _ in range(reps):
        hop = idx.multi_hop_transport_to_poi(PLAN_CITY, "museum")
    multi_hop_ms = (time.perf_counter() - t) / reps * 1000

    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "node_types": dict(sorted(node_types.items(), key=lambda kv: str(kv[0]))),
        "relations": dict(sorted(relations.items(), key=lambda kv: str(kv[0]))),
        "build_graph_s": round(build_s, 3),
        "graph_index_s": round(index_s, 3),
        "multi_hop_ms": round(multi_hop_ms, 3),
        # The Louvre case from #126: how many museums the hub relation reaches,
        # and whether the city's best-known ones are among them.
        "multi_hop_museum_count": len(hop),
        "multi_hop_museum_names": sorted(p["name"] for p in hop),
    }


# ---------------------------------------------------------------------------
# retrieval -- Phase 0 record, checkpoint KN-2
# ---------------------------------------------------------------------------

def retrieval_section(retriever: FusionRetriever, init_s: float) -> dict:
    out = {"retriever_init_s": round(init_s, 3), "archetypes": {}}
    latencies = []
    for archetype in ARCHETYPES:
        query = archetype_query(archetype)
        t = time.perf_counter()
        results = retriever.retrieve(query, config="fusion", destination_id=PLAN_CITY,
                                     archetype=archetype, top_k=N_DAYS * 24)
        latencies.append((time.perf_counter() - t) * 1000)

        top8 = results[:8]
        sources = collections.Counter(
            source for r in top8 for source in r.get("retrieved_by", []))
        chain_in_8 = sum(1 for r in top8
                         if CHAIN_RETRIEVER in r.get("retrieved_by", []))
        chain_in_pool = sum(1 for r in results
                            if CHAIN_RETRIEVER in r.get("retrieved_by", []))
        # Absent vs zero: before phase 4 there is no chain retriever at all, and
        # reporting 0 would read as "built and contributing nothing".
        has_chain = any(CHAIN_RETRIEVER in r.get("retrieved_by", []) for r in results)
        out["archetypes"][archetype] = {
            "top8_doc_ids": [r["doc_id"] for r in top8],
            "top8_sources": dict(sorted(sources.items())),
            "chain_in_top8": chain_in_8 if has_chain else None,
            "chain_in_pool": chain_in_pool if has_chain else None,
            "pool_size": len(results),
        }
    out["retrieve_ms"] = round(sum(latencies) / len(latencies), 3)
    return out


def kn2_verdict(section: dict) -> str:
    """KN-2's pass/fail. Target band is 2-4 of the top 8 (see #126, K4): 0 means
    the chain is being drowned in the fusion, 6+ means it is deciding the answer
    on its own, which is the head-counting failure #63 fixed for the other
    retrievers."""
    shares = [v["chain_in_top8"] for v in section["archetypes"].values()]
    if all(s is None for s in shares):
        return "n/a -- no chain retriever yet (expected before #126 phase 4)"
    present = [s for s in shares if s is not None]
    worst = max(present)
    # Nightlife is deliberately excluded from a chain's second leg (#126 phase 2,
    # night-service limit), so a low share there is expected and must not fail
    # the checkpoint on its own -- the band is judged on the highest share.
    if worst == 0:
        return f"FAIL -- chain reaches no top-8 slot (shares {present})"
    if worst > 4:
        return f"FAIL -- chain dominates ({worst}/8); lower RETRIEVER_WEIGHTS['chain']"
    if worst < 2:
        return f"WARN -- chain barely surfaces ({worst}/8); check list order and dedupe"
    return f"PASS -- chain share {present} within the 2-4 band"


# ---------------------------------------------------------------------------
# plan -- the "behaviour did not change while the flag is off" regression check
# ---------------------------------------------------------------------------

def plan_section() -> dict:
    orchestrator = RoamWiseOrchestrator()
    state = orchestrator.plan_trip(
        PLAN_PREFERENCES, destination_id=PLAN_CITY, n_days=N_DAYS,
        daily_minutes_budget=PLAN_BUDGET_MINUTES, start_date=_START)
    days = state["routing"]["itinerary"]
    return {
        "preferences": PLAN_PREFERENCES,
        "archetype": state["segmentation"]["archetype"],
        "day_start_hour": state["routing"]["day_start_hour"],
        # Ids rather than names: a renamed POI is a catalogue change, not a
        # routing change, and the regression check is about routing.
        "days": [[poi["poi_id"] for poi in day["route"]] for day in days],
        "km_per_day": [round(day["distance_km"], 3) for day in days],
    }


# ---------------------------------------------------------------------------
# pool -- the frontier KN-3 (the flag gate) is judged against
# ---------------------------------------------------------------------------

def pool_section(retriever: FusionRetriever, budget_hours: list[int]) -> dict:
    graph = GraphIndex()
    router = RouterAgent(graph)
    preferences = archetype_preferences()
    rows = []
    for city in CITIES:
        for archetype in ARCHETYPES:
            query = archetype_query(archetype)
            for hours in budget_hours:
                budget = hours * 60
                results = retriever.retrieve(query, config="fusion", destination_id=city,
                                             archetype=archetype, top_k=N_DAYS * 24)
                candidates = [graph.g.nodes[r["poi_id"]] | {"poi_id": r["poi_id"]}
                              for r in results if r.get("type") == "poi"]
                routing = router.run(city, candidates, n_days=N_DAYS,
                                     daily_minutes_budget=budget, archetype=archetype,
                                     narrate=False, start_date=_START,
                                     preferences=preferences[archetype])
                metrics = measure(routing["itinerary"], routing["day_start_hour"],
                                  budget, preferences=preferences[archetype])
                rows.append({"city": city, "archetype": archetype,
                             "budget_hours": hours, **metrics})
    df = pd.DataFrame(rows)
    return {
        "budget_hours": budget_hours,
        "cells": len(df),
        # Recorded because these come from `user_survey.csv`, which #124 may
        # change -- a shifted baseline should be explainable, not mysterious.
        "preference_vectors": {a: {k: round(v, 4) for k, v in preferences[a].items()}
                               for a in ARCHETYPES},
        "stops_per_day": round(df.stops_per_day.mean(), 3),
        "km_per_stop": round(df.km_per_stop.mean(), 3),
        "mean_pref_match": round(df.mean_pref_match.mean(), 4),
        "mean_quality": round(df.mean_quality.mean(), 4),
        "categories_per_day": round(df.categories_per_day.mean(), 3),
        "closed_violations": int(df.closed_violations.sum()),
        "budget_overruns": int(df.budget_overruns.sum()),
        "days_with_meals": int(df.days_with_meals.sum()),
        "days_total": int(df.n_days.sum()),
    }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

SECTIONS = ["graph", "retrieval", "plan", "pool"]


def collect(only: list[str], budget_hours: list[int]) -> dict:
    snapshot = {"start_date": START_DATE, "n_days": N_DAYS,
                "cities": CITIES, "archetypes": ARCHETYPES}
    retriever = init_s = None
    if {"retrieval", "pool"} & set(only):
        t = time.perf_counter()
        retriever = FusionRetriever()
        init_s = time.perf_counter() - t

    if "graph" in only:
        snapshot["graph"] = graph_section()
    if "retrieval" in only:
        snapshot["retrieval"] = retrieval_section(retriever, init_s)
    if "plan" in only:
        snapshot["plan"] = plan_section()
    if "pool" in only:
        snapshot["pool"] = pool_section(retriever, budget_hours)
    return snapshot


def _flatten(obj, prefix="") -> dict:
    flat = {}
    for key, value in obj.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten(value, path))
        else:
            flat[path] = value
    return flat


def compare(current: dict, baseline: dict) -> list[str]:
    """Lines describing every difference. Latency keys are reported but never
    counted as a difference -- they move with the machine, not with the code."""
    now, before = _flatten(current), _flatten(baseline)
    lines = []
    for key in sorted(set(now) | set(before)):
        a, b = before.get(key, "<absent>"), now.get(key, "<absent>")
        if a == b:
            continue
        leaf = key.rsplit(".", 1)[-1]
        if leaf in LATENCY_KEYS:
            lines.append(f"  ~ {key}: {a} -> {b}  (timing, informational)")
        else:
            lines.append(f"  ! {key}: {a} -> {b}")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--write", action="store_true",
                        help=f"write the snapshot to {BASELINE_JSON.name}")
    parser.add_argument("--compare", action="store_true",
                        help="diff the current measurement against the committed baseline")
    parser.add_argument("--only", nargs="+", choices=SECTIONS, default=SECTIONS,
                        help="measure only these sections (KN-1 is --only graph)")
    parser.add_argument("--full", action="store_true",
                        help="measure the pool over the 12/15/18h grid (KN-3)")
    args = parser.parse_args()

    budget_hours = FULL_BUDGET_HOURS if args.full else QUICK_BUDGET_HOURS
    snapshot = collect(args.only, budget_hours)

    if "graph" in args.only:
        g = snapshot["graph"]
        print(f"\ngraph      {g['nodes']} nodes, {g['edges']} edges, "
              f"build {g['build_graph_s']}s, multi_hop {g['multi_hop_ms']}ms")
        for relation, count in g["relations"].items():
            print(f"           {str(relation):<16}{count:>8}")
        print(f"           hub->museum reach: {g['multi_hop_museum_count']} museums")
    if "retrieval" in args.only:
        r = snapshot["retrieval"]
        print(f"\nretrieval  init {r['retriever_init_s']}s, retrieve {r['retrieve_ms']}ms")
        for archetype, data in r["archetypes"].items():
            share = data["chain_in_top8"]
            print(f"           {archetype:<20} top-8 by {data['top8_sources']}"
                  f"  chain={'-' if share is None else f'{share}/8'}")
        print(f"           KN-2: {kn2_verdict(r)}")
    if "plan" in args.only:
        p = snapshot["plan"]
        print(f"\nplan       {p['archetype']}, start {p['day_start_hour']}h, "
              f"{sum(len(d) for d in p['days'])} stops over {len(p['days'])} days")
        for i, (day, km) in enumerate(zip(p["days"], p["km_per_day"]), 1):
            print(f"           day {i}: {km:>6.2f} km  {' '.join(day)}")
    if "pool" in args.only:
        p = snapshot["pool"]
        print(f"\npool       {p['cells']} cells over {p['budget_hours']}h budgets")
        print(f"           stops/day {p['stops_per_day']}   km/stop {p['km_per_stop']}   "
              f"pref {p['mean_pref_match']}   cats/day {p['categories_per_day']}")
        print(f"           closed {p['closed_violations']}   overruns {p['budget_overruns']}"
              f"   meals {p['days_with_meals']}/{p['days_total']}")

    if args.compare:
        if not BASELINE_JSON.exists():
            raise SystemExit(f"\n{BASELINE_JSON.name} is missing -- record it first "
                             f"with --write (on the commit you want to compare against).")
        baseline = json.loads(BASELINE_JSON.read_text())
        differences = compare(snapshot, {k: v for k, v in baseline.items() if k in snapshot})
        print(f"\ncompare    against {BASELINE_JSON.name}")
        if not differences:
            print("           identical")
        else:
            print("\n".join(differences))

    if args.write:
        merged = {}
        if BASELINE_JSON.exists():
            # Sections not measured in this run keep whatever the baseline
            # already held, so `--only graph --write` cannot silently drop the
            # pool numbers the flag gate is judged against.
            merged = json.loads(BASELINE_JSON.read_text())
        merged.update(snapshot)
        BASELINE_JSON.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n")
        print(f"\nwrote      {BASELINE_JSON.name}")


if __name__ == "__main__":
    main()
