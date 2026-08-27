"""
LangGraph-based alternative to `orchestrator.py::RoamWiseOrchestrator`.

Issue: the proposal names LangGraph ("e.g., LangGraph or similar multi-agent
systems") as the orchestration framework. `orchestrator.py` deliberately used
a hand-rolled state machine instead (see its module docstring) to keep the
project dependency-light. This module is the other side of that trade-off:
the *same* five-node flow, reimplemented on `langgraph.graph.StateGraph`, so
the two can be compared directly rather than argued about in the abstract.

Design notes:
  - The underlying agents (TravelerSegmenter, ForecasterAgent, FusionRAGAgent,
    RouterAgent) are reused as-is -- LangGraph only replaces the orchestration
    *control flow*, not the agents themselves. That's the honest boundary:
    the proposal's "agentic orchestration" claim is about how nodes are
    sequenced and branched, not about each node's internal implementation.
  - `_select_destination` only runs when no destination was pinned, expressed
    here as a real conditional edge (`add_conditional_edges`) rather than a
    plain `if` inside one big function -- this is the one place LangGraph's
    declarative branching is actually a legible improvement over the custom
    orchestrator's imperative equivalent. See REPORT.md for the full
    comparison.
  - `plan_trip()` has the same return shape as
    `RoamWiseOrchestrator.plan_trip()`, so call sites (app.py, tests,
    evaluation) can swap one for the other with no other code changes. The
    signature is meant to match too and currently lags by `start_date`, which
    is issue #76: a parameter added to one orchestrator and not the other is
    not a compile error, it is a feature that silently does nothing on this
    path.

Requires the optional `requirements-langgraph.txt` dependency group (not
installed by default -- see that file's header comment for why).
"""
import json
from pathlib import Path
from typing import Optional, TypedDict

import pandas as pd
from langgraph.graph import END, StateGraph

from roamwise.agents.forecaster_agent import ForecasterAgent
from roamwise.agents.fusion_rag_agent import FusionRAGAgent
from roamwise.agents.llm_client import LLMClient, get_default_llm_client
from roamwise.agents.orchestrator import MIN_RETRIEVED_POIS, RETRIEVED_POIS_PER_DAY
from roamwise.agents.router_agent import RouterAgent
from roamwise.knowledge_graph.build_graph import GraphIndex
from roamwise.models.segmentation import TravelerSegmenter
from roamwise.retrieval.query import archetype_query
from roamwise.optimization.travel_modes import DEFAULT_MODE

DATA_DIR = Path(__file__).parent.parent / "data"

TAG_TO_PREF = {
    "culture": "culture", "history": "culture", "art": "culture",
    "beach": "relax", "nature": "nature", "nightlife": "nightlife",
    "shopping": "budget", "food": "relax", "religion": "culture",
    "luxury": "budget", "budget": "budget", "romance": "relax", "music": "culture",
}


class PlanState(TypedDict, total=False):
    preferences: dict
    n_days: int
    travel_month: Optional[str]
    top_k_pois: int
    daily_minutes_budget: int
    day_start_hour: float
    use_real_routing: bool
    travel_mode: str
    arrival_hub_id: Optional[str]
    destination_id: Optional[str]
    archetype: str
    segmentation: dict
    forecast: dict
    fusion_rag: dict
    routing: dict
    final_plan: str


class RoamWiseLangGraphOrchestrator:
    def __init__(self, llm: LLMClient = None, retrieval_config: str = "fusion"):
        self.llm = llm or get_default_llm_client()
        self.graph_index = GraphIndex()
        self.segmenter = TravelerSegmenter()
        self.forecaster = ForecasterAgent(llm=self.llm)
        self.fusion_rag = FusionRAGAgent(llm=self.llm)
        self.router = RouterAgent(self.graph_index, llm=self.llm)
        self.retrieval_config = retrieval_config
        self.destinations = pd.read_csv(DATA_DIR / "destinations.csv")
        self.destinations["tags"] = self.destinations.tags.apply(json.loads)
        self._compiled = self._build_graph()

    # ---- graph wiring -----------------------------------------------------

    def _build_graph(self):
        g = StateGraph(PlanState)
        g.add_node("segment", self._segment)
        g.add_node("select_destination", self._select_destination)
        g.add_node("forecast", self._forecast)
        g.add_node("retrieve", self._retrieve)
        g.add_node("route", self._route)
        g.add_node("synthesize", self._synthesize)

        g.set_entry_point("segment")
        g.add_conditional_edges(
            "segment",
            lambda s: "select_destination" if s.get("destination_id") is None else "forecast",
            {"select_destination": "select_destination", "forecast": "forecast"},
        )
        g.add_edge("select_destination", "forecast")
        g.add_edge("forecast", "retrieve")
        g.add_edge("retrieve", "route")
        g.add_edge("route", "synthesize")
        g.add_edge("synthesize", END)
        return g.compile()

    def plan_trip(self, preferences: dict, destination_id: str = None, n_days: int = 3,
                  travel_month: str = None, top_k_pois: int = None,
                  daily_minutes_budget: int = 480, use_real_routing: bool = False,
                  travel_mode: str = DEFAULT_MODE,
                  day_start_hour: float = None, arrival_hub_id: str = None) -> dict:
        """Same return shape as RoamWiseOrchestrator.plan_trip(). The signature
        still lags it by `start_date`, which is issue #76 -- opening hours are
        read as the coarse open/close pair on this path."""
        if top_k_pois is None:
            top_k_pois = max(MIN_RETRIEVED_POIS, n_days * RETRIEVED_POIS_PER_DAY)
        init_state: PlanState = {
            "preferences": preferences, "n_days": n_days, "travel_month": travel_month,
            "top_k_pois": top_k_pois,
            "daily_minutes_budget": daily_minutes_budget, "use_real_routing": use_real_routing,
            "day_start_hour": day_start_hour,
            "travel_mode": travel_mode, "arrival_hub_id": arrival_hub_id,
            "destination_id": destination_id,
        }
        return self._compiled.invoke(init_state)

    # ---- nodes --------------------------------------------------------------
    # Each node takes the accumulated state and returns only the keys it adds
    # or changes; LangGraph merges that into the running state for the next node.

    def _segment(self, state: PlanState) -> dict:
        seg = self.segmenter.classify(state["preferences"])
        return {"archetype": seg["archetype"], "segmentation": seg}

    def _select_destination(self, state: PlanState) -> dict:
        destination_id = self._recommend_destination(state["preferences"], state.get("travel_month"))
        return {"destination_id": destination_id}

    def _forecast(self, state: PlanState) -> dict:
        forecast = self.forecaster.run(state["destination_id"], travel_month=state.get("travel_month"))
        return {"forecast": forecast}

    def _retrieve(self, state: PlanState) -> dict:
        # Kept identical to orchestrator.py's query -- see #63 and the note
        # in HANDOFF about prompt/query drift between the two orchestrators.
        query = archetype_query(state["archetype"])
        rag = self.fusion_rag.run(
            query, destination_id=state["destination_id"], archetype=state["archetype"],
            config=self.retrieval_config, top_k=state.get("top_k_pois", 12),
            narrate=False,
        )
        return {"fusion_rag": rag}

    def _route(self, state: PlanState) -> dict:
        rag = state["fusion_rag"]
        candidate_pois = [
            self.graph_index.g.nodes[r["poi_id"]] | {"poi_id": r["poi_id"]}
            for r in rag["results"] if r.get("type") == "poi"
        ]
        if not candidate_pois:  # standard-prompting config: no retrieval, fall back to raw city POIs
            candidate_pois = self.graph_index.city_pois(state["destination_id"])[: state.get("top_k_pois", 12)]

        # The price filter that used to sit here is gone; it never removed a
        # POI. Kept in step with orchestrator.py, which is the whole point of
        # this file (#67).

        routing = self.router.run(
            state["destination_id"], candidate_pois, n_days=state["n_days"],
            daily_minutes_budget=state.get("daily_minutes_budget", 480),
            day_start_hour=state.get("day_start_hour"), archetype=state.get("archetype"),
            use_real_routing=state.get("use_real_routing", False),
            travel_mode=state.get("travel_mode", DEFAULT_MODE),
            narrate=False, preferences=state.get("preferences"),
            arrival_hub_id=state.get("arrival_hub_id"),
        )
        return {"routing": routing, "travel_mode": routing["travel_mode"]}

    def _synthesize(self, state: PlanState) -> dict:
        """Kept deliberately identical to orchestrator._synthesize -- see that
        method's docstring for why the retrieval context is excluded (#56) and
        why this is a join rather than an indented f-string (#22)."""
        city = self.destinations.set_index("destination_id").loc[state["destination_id"], "city"]
        prompt = "\n\n".join([
            f"Trip plan for a {state['archetype']} traveler visiting {city} ({state['destination_id']}).",
            f"Forecast: {state['forecast']['narrative']}",
            state["routing"]["facts"],
        ])
        final_plan = self.llm.complete(
            system="You are RoamWise, an agentic travel-planning assistant. Write a coherent "
                   "recommendation for the itinerary below, describing its stops in the order "
                   "given and working in the forecast's timing advice. The itinerary is the "
                   "complete plan: describe only the stops it lists, and never mention or "
                   "suggest any other place, attraction or venue.",
            prompt=prompt,
        )
        return {"final_plan": final_plan}

    # ---- helpers (identical to orchestrator.py -- see that file's docstring
    # for why this small overlap is duplicated rather than shared: the two
    # orchestrators are meant to be independently comparable) --------------

    def _recommend_destination(self, preferences: dict, travel_month: str = None) -> str:
        budget_target = preferences.get("budget", 0.5) * 2 + 1
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
        vals = [preferences.get(TAG_TO_PREF[t], 0.3) for t in tags if t in TAG_TO_PREF]
        return sum(vals) / len(vals) if vals else 0.3


if __name__ == "__main__":
    orch = RoamWiseLangGraphOrchestrator()
    prefs = {"budget": 0.3, "culture": 0.9, "nature": 0.4, "nightlife": 0.3, "relax": 0.3, "adventure": 0.3}
    result = orch.plan_trip(prefs, n_days=3)
    print("Archetype:", result["archetype"])
    print("Destination:", result["destination_id"])
    print()
    print(result["final_plan"])
