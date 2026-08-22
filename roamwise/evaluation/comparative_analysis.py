"""
Comparative analysis across the three retrieval configurations the proposal
asks for: Fusion RAG (Semantic+Graph+Keyword) vs Hybrid RAG (Semantic+Keyword)
vs standard prompting (no retrieval).

Design note on metrics: this project's default LLM layer (agents/llm_client.py)
is a deterministic template, not a live generative model, so it cannot
actually "hallucinate" free text -- there is nothing for it to invent. Rather
than fabricate a fake hallucination number, we measure the thing that
determines hallucination risk at the architecture level: whether the content
handed to the generation step is *grounded* (verifiably present in the
knowledge base) or not, plus two task-accuracy metrics that make the value of
each retrieval component concrete:

  1. Multi-hop query resolution accuracy (Recall@k against a graph-computed
     gold set) -- this is the metric that should most clearly separate Fusion
     RAG from Hybrid RAG, since only Fusion RAG has graph traversal.
  2. Archetype/category relevance precision -- fraction of surfaced POIs that
     actually match the requesting traveler archetype's preferred categories.
  3. Grounded-entity rate -- fraction of surfaced content traceable to a real
     KB node (structural hallucination proxy; 1.0 by construction for any
     retrieval-based config, 0.0 for standard prompting with no fallback).
  4. Itinerary coherence -- average per-day walking distance once the
     RouterAgent builds a route from each config's candidate POIs (lower is
     more geographically coherent).

If ANTHROPIC_API_KEY is set, `run_llm_hallucination_probe()` additionally
runs a real generative hallucination check (see its docstring) -- this part
is optional and skipped by default.
"""
import json
import os
from pathlib import Path

import pandas as pd

from roamwise.agents.llm_client import get_default_llm_client
from roamwise.agents.router_agent import RouterAgent
from roamwise.knowledge_graph.build_graph import CATEGORY_AFFINITY, GraphIndex
from roamwise.retrieval.fusion import FusionRetriever

HERE = Path(__file__).parent
CONFIGS = ["fusion", "hybrid", "standard"]

TEST_QUERIES = [
    # (destination_id, archetype, category, query_text)
    ("IST", "Culture Enthusiast", "landmark", "landmarks within walking distance of a transport hub"),
    ("PAR", "Culture Enthusiast", "museum", "museums close to a train station or airport"),
    ("ROM", "Culture Enthusiast", "museum", "museums near a transport hub for history lovers"),
    ("BCN", "Nightlife Seeker", "nightlife", "nightlife spots this traveler would enjoy"),
    ("AMS", "Nature & Adventure", "nature", "parks and nature spots near transport"),
    ("PRG", "Budget Backpacker", "nightlife", "cheap nightlife near the train station"),
    ("VIE", "Luxury Traveler", "shopping", "upscale shopping close to transport hubs"),
    ("LIS", "Beach & Relax", "beach", "relaxing beach spots reachable from the airport"),
    ("IST", "Family Traveler", "food", "we just arrived at the central station with heavy luggage and are starving, where can we get a quick meal without walking much?"),
    ("PAR", "Culture Enthusiast", "museum", "suitable places for the first day after a morning arrival"),
    ("ROM", "Culture Enthusiast", "history", "my elderly parents want to see historical monuments but cannot handle steep hills or long walks from the subway."),
    ("AMS", "Nightlife Seeker", "nightlife", "traveling on a tight budget and want to experience local nightlife that is safely accessible via late-night public transit."),
    ("BCN", "Luxury Traveler", "shopping", "we want to do a full day of luxury shopping and fine dining without needing to hail a taxi between locations."),
    ("LIS", "Nature & Adventure", "nature", "are there any quiet natural escapes in the city that are directly connected to the main train lines?"),
    ("VIE", "Culture Enthusiast", "culture", "where should we head for a relaxed evening out right after spending the entire afternoon at the main art gallery?"),
    ("PRG", "Budget Backpacker", "landmark", "we have a short layover in the city, what is the most iconic landmark we can realistically visit and still catch our flight?"),
    ("IST", "Culture Enthusiast", "history", "which neighborhoods offer the best mix of ancient architecture and modern cafes within a compact walking area?"),
    ("ROM", "Culture Enthusiast", "museum", "if we start our morning at the central square, what is a logical path to hit a museum and a local market before lunch?")
]


def _gold_multi_hop(idx: GraphIndex, destination_id: str, category: str, max_km: float = 6.0) -> set:
    return {r["poi_id"] for r in idx.multi_hop_transport_to_poi(destination_id, category, max_km=max_km)}


def recall_at_k(retrieved_ids: list, gold: set) -> float:
    if not gold:
        return None
    hit = len(set(retrieved_ids) & gold)
    return hit / len(gold)


def run_comparative_analysis(top_k: int = 8) -> pd.DataFrame:
    idx = GraphIndex()
    retriever = FusionRetriever()
    router = RouterAgent(idx)
    rows = []

    for dest_id, archetype, category, query in TEST_QUERIES:
        gold = _gold_multi_hop(idx, dest_id, category)
        preferred_categories = set(CATEGORY_AFFINITY[archetype])

        for config in CONFIGS:
            results = retriever.retrieve(query, config=config, destination_id=dest_id, archetype=archetype, top_k=top_k)
            poi_results = [r for r in results if r.get("type") == "poi"]
            retrieved_ids = [r["poi_id"] for r in poi_results]

            recall = recall_at_k(retrieved_ids, gold)

            candidate_pois = [idx.g.nodes[pid] | {"poi_id": pid} for pid in retrieved_ids]
            if not candidate_pois:  # standard prompting has no retrieval -> unfiltered fallback
                candidate_pois = idx.city_pois(dest_id)[:top_k]

            relevant = sum(1 for p in candidate_pois if p.get("category") in preferred_categories)
            precision = relevant / len(candidate_pois) if candidate_pois else 0.0

            grounded_rate = 1.0 if poi_results else 0.0

            routing = router.run(dest_id, candidate_pois, n_days=1)
            day = routing["itinerary"][0]
            n_stops = len(day["route"])
            km_per_stop = day["distance_km"] / n_stops if n_stops else None

            rows.append({
                "destination_id": dest_id, "archetype": archetype, "category": category,
                "config": config, "recall_at_k": recall, "archetype_precision": round(precision, 3),
                "grounded_entity_rate": grounded_rate, "n_candidate_pois": len(candidate_pois),
                "n_stops_day1": n_stops, "km_per_stop_day1": round(km_per_stop, 2) if km_per_stop else None,
            })

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("config")
        .agg(
            mean_recall_at_k=("recall_at_k", "mean"),
            mean_archetype_precision=("archetype_precision", "mean"),
            mean_grounded_entity_rate=("grounded_entity_rate", "mean"),
            mean_km_per_stop=("km_per_stop_day1", "mean"),
        )
        .reindex(CONFIGS)
        .round(3)
    )


def run_llm_hallucination_probe():
    """Optional, opt-in: if ANTHROPIC_API_KEY is set, ask the live LLM to
    describe a POI in each test city with zero retrieved context (true
    'standard prompting'), then check whether every named entity in the
    response matches a real KB node. Skipped (returns None) without a key
    so the rest of the evaluation stays free/offline."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    from roamwise.agents.llm_client import AnthropicLLMClient
    idx = GraphIndex()
    known_names = {data["name"].lower() for _, data in idx.g.nodes(data=True) if data.get("type") == "POI"}
    llm = AnthropicLLMClient()
    probe_rows = []
    for dest_id, _, _, _ in TEST_QUERIES:
        city = idx.g.nodes[dest_id]["name"]
        prompt = f"List 5 specific points of interest to visit in {city}, one per line, name only."
        text = llm.complete(system="You are a travel assistant.", prompt=prompt)
        lines = [l.strip("-* ").lower() for l in text.splitlines() if l.strip()]
        matched = sum(any(name in line or line in name for name in known_names) for line in lines)
        probe_rows.append({"destination_id": dest_id, "named": len(lines), "matched_kb": matched})
    return pd.DataFrame(probe_rows)


if __name__ == "__main__":
    df = run_comparative_analysis()
    df.to_csv(HERE / "comparative_analysis_results.csv", index=False)
    summary = summarize(df)
    summary.to_csv(HERE / "comparative_analysis_summary.csv")
    print(summary.to_string())

    probe = run_llm_hallucination_probe()
    if probe is not None:
        probe.to_csv(HERE / "llm_hallucination_probe.csv", index=False)
        print("\nLive LLM hallucination probe:\n", probe.to_string())
    else:
        print("\n(No ANTHROPIC_API_KEY set -- skipped the optional live-LLM hallucination probe.)")
