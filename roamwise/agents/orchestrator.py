"""
Top-level agentic orchestration ("Agentic Orchestration -- Core Focus" in the
proposal). This is a lightweight custom state machine rather than LangGraph:
the proposal names LangGraph as one example ("LangGraph or similar
multi-agent systems"), and a hand-rolled orchestrator keeps the whole project
dependency-light and fully deterministic for grading, while still exhibiting
the same shape LangGraph would enforce -- a shared state dict threaded
through named agent nodes in sequence, each of which can be inspected or
swapped independently.

Flow for a user request:
  1. TravelerSegmenter classifies the free-form preferences into an archetype
     (KMeans "tool").
  2. If no destination was pinned, ForecasterAgent scores every candidate
     city by (a) budget/interest match and (b) forecasted low crowding in the
     travel window, and picks the winner.
  3. FusionRAGAgent retrieves grounded, archetype-aware POI context for that
     city (Semantic + Graph + Keyword, fused via RRF).
  4. RouterAgent zones the retrieved POIs geographically and solves the
     day-by-day route with the 2-opt optimization tool.
  5. A final synthesis narrates the full plan from the three agents' outputs.
"""
import json
import pandas as pd

from agents.forecaster_agent import ForecasterAgent
from agents.fusion_rag_agent import FusionRAGAgent
from agents.llm_client import LLMClient, get_default_llm_client
from agents.router_agent import RouterAgent
from knowledge_graph.build_graph import GraphIndex
from models.segmentation import TravelerSegmenter
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


class RoamWiseOrchestrator:
    def __init__(self, llm: LLMClient = None, retrieval_config: str = "fusion"):
        self.llm = llm or get_default_llm_client()
        self.graph = GraphIndex()
        self.segmenter = TravelerSegmenter()
        self.forecaster = ForecasterAgent(llm=self.llm)
        self.fusion_rag = FusionRAGAgent(llm=self.llm)
        self.router = RouterAgent(self.graph, llm=self.llm)
        self.retrieval_config = retrieval_config
        self.destinations = pd.read_csv(DATA_DIR / "destinations.csv")
        self.destinations["tags"] = self.destinations.tags.apply(json.loads)

    def plan_trip(self, preferences: dict, destination_id: str = None, n_days: int = 3,
                  travel_month: str = None, top_k_pois: int = 12) -> dict:
        """preferences: {budget, culture, nature, nightlife, relax, adventure} in [0,1].
        destination_id: pin a city, or leave None to let the orchestrator pick one."""
        state: dict = {"preferences": preferences, "n_days": n_days}

        # --- Node 1: traveler segmentation ---
        seg = self.segmenter.classify(preferences)
        state["archetype"] = seg["archetype"]
        state["segmentation"] = seg

        # --- Node 2: destination selection (Forecaster Agent as scorer) ---
        if destination_id is None:
            destination_id = self._recommend_destination(preferences, travel_month)
        state["destination_id"] = destination_id
        state["forecast"] = self.forecaster.run(destination_id, travel_month=travel_month)

        # --- Node 3: Fusion RAG Agent retrieves grounded, archetype-aware POIs ---
        query = f"best {seg['archetype'].lower()} points of interest and experiences"
        rag = self.fusion_rag.run(
            query, destination_id=destination_id, archetype=seg["archetype"], config=self.retrieval_config, top_k=top_k_pois,
        )
        state["fusion_rag"] = rag
        candidate_pois = [
            self.graph.g.nodes[r["poi_id"]] | {"poi_id": r["poi_id"]}
            for r in rag["results"] if r.get("type") == "poi"
        ]
        if not candidate_pois:  # standard-prompting config: no retrieval, fall back to raw city POIs
            candidate_pois = self.graph.city_pois(destination_id)[:top_k_pois]

        # --- Node 4: Router Agent builds the optimized day-by-day route ---
        routing = self.router.run(destination_id, candidate_pois, n_days=n_days)
        state["routing"] = routing

        # --- Node 5: final synthesis ---
        state["final_plan"] = self._synthesize(state)
        return state

    def _recommend_destination(self, preferences: dict, travel_month: str = None) -> str:
        budget_target = preferences.get("budget", 0.5) * 2 + 1  # map [0,1] -> [1,3]
        scored = []
        for _, d in self.destinations.iterrows():
            tag_overlap = self._tag_affinity(preferences, d.tags)
            budget_fit = 1 - abs(d.budget_level - budget_target) / 2
            fc = self.forecaster.run(d.destination_id, travel_month=travel_month, horizon_months=6)
            crowd_penalty = {"low": 0.0, "medium": 0.15, "high": 0.35}[fc["crowding_level"]]
            score = 0.5 * tag_overlap + 0.3 * budget_fit - crowd_penalty
            scored.append((d.destination_id, score))
        return max(scored, key=lambda t: t[1])[0]

    @staticmethod
    def _tag_affinity(preferences: dict, tags: list[str]) -> float:
        tag_to_pref = {
            "culture": "culture", "history": "culture", "art": "culture",
            "beach": "relax", "nature": "nature", "nightlife": "nightlife",
            "shopping": "budget", "food": "relax", "religion": "culture",
            "luxury": "budget", "budget": "budget", "romance": "relax", "music": "culture",
        }
        vals = [preferences.get(tag_to_pref[t], 0.3) for t in tags if t in tag_to_pref]
        return sum(vals) / len(vals) if vals else 0.3

    def _synthesize(self, state: dict) -> str:
        city = self.destinations.set_index("destination_id").loc[state["destination_id"], "city"]
        prompt = f"""
        Trip plan for a {state['archetype']} traveler visiting {city} ({state['destination_id']}).

        Forecast: {state['forecast']['narrative']}

        Retrieved context ({state['fusion_rag']['config']} configuration): {state['fusion_rag']['narrative']}

        Itinerary: {state['routing']['narrative']}
        """
        return self.llm.complete(
            system="You are RoamWise, an agentic travel-planning assistant. Combine the forecast, "
                   "retrieved context, and itinerary into one coherent recommendation.",
            prompt=prompt,
        )


if __name__ == "__main__":
    orch = RoamWiseOrchestrator()
    prefs = {"budget": 0.3, "culture": 0.9, "nature": 0.4, "nightlife": 0.3, "relax": 0.3, "adventure": 0.3}
    result = orch.plan_trip(prefs, n_days=3)
    print("Archetype:", result["archetype"])
    print("Destination:", result["destination_id"])
    print()
    print(result["final_plan"])
