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
  - `plan_trip()` has the same signature and return shape as
    `RoamWiseOrchestrator.plan_trip()`, so call sites (app.py, tests,
    evaluation) can swap one for the other with no other code changes. Keeping
    the signatures level is the maintenance cost of having two orchestrators,
    and it is not enforced by anything the compiler does: `start_date` was
    added to one and not the other and silently did nothing on this path for
    as long as it took someone to read both files (issue #76). What guards it
    now is a behavioural test, not the interface test that missed it.
  - Every node logs through `app_logging.log_step` under the *same step name*
    as its counterpart in `orchestrator.py`, because both paths are read on
    the same System logs screen. Until #129 this file logged nothing at all,
    which nobody noticed while the only caller was the test suite -- and which
    would have emptied the operator screen the moment the sidebar could select
    this path. The step names are asserted equal by
    `test_both_orchestrators_log_the_same_steps`: observability parity is not
    self-maintaining either, for the same reason #76's signature parity was
    not.

Requires the optional `requirements-langgraph.txt` dependency group (not
installed by default -- see that file's header comment for why).
"""
import datetime
import json
from pathlib import Path
from typing import Optional, TypedDict

import pandas as pd
from langgraph.graph import END, StateGraph

from roamwise.app_logging import get_logger, log_step
from roamwise.agents.forecaster_agent import ForecasterAgent
from roamwise.agents.fusion_rag_agent import FusionRAGAgent
from roamwise.agents.llm_client import LLMClient, get_default_llm_client
# `_free_entry_share` is imported rather than copied, unlike the prompt and the
# helpers below: it computes a statistic the UI displays, not a piece of
# orchestration, so there is nothing to compare between the two paths and a
# second copy would only be one more thing to drift (#76, #129).
from roamwise.agents.orchestrator import (
    MIN_RETRIEVED_POIS, RETRIEVED_POIS_PER_DAY, _free_entry_share,
)
from roamwise.agents.router_agent import RouterAgent
from roamwise.knowledge_graph.build_graph import GraphIndex
from roamwise.models.segmentation import TravelerSegmenter
from roamwise.retrieval.query import archetype_query
from roamwise.optimization.travel_modes import DEFAULT_MODE

DATA_DIR = Path(__file__).parent.parent / "data"

log = get_logger(__name__)

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
    start_date: Optional[datetime.date]
    arrival_hub_id: Optional[str]
    destination_id: Optional[str]
    archetype: str
    segmentation: dict
    forecast: dict
    fusion_rag: dict
    routing: dict
    free_entry_share: Optional[float]
    final_plan: str
    final_plan_truncated: bool


class RoamWiseLangGraphOrchestrator:
    # See RoamWiseOrchestrator.ORCHESTRATOR_ID: this is what the sidebar picker
    # passes to the factory and what the entry log record carries, so the
    # operator screen can name the path whose steps it is showing (#129).
    ORCHESTRATOR_ID = "langgraph"

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
            self._needs_destination,
            {"select_destination": "select_destination", "forecast": "forecast"},
        )
        g.add_edge("select_destination", "forecast")
        g.add_edge("forecast", "retrieve")
        g.add_edge("retrieve", "route")
        g.add_edge("route", "synthesize")
        g.add_edge("synthesize", END)
        return g.compile()

    def _needs_destination(self, state: PlanState) -> str:
        """The one conditional edge -- and the one step the node cannot log.

        `orchestrator.py` keeps its `if destination_id is None` *inside* the
        "Destination selection" step, so that step appears on the operator
        screen either way and says `pinned_by_user=` which way it went. Here the
        branch is the edge, so when the traveler pins a city the node never
        runs and there is nothing to log from -- which is every plan the sidebar
        makes, since it always names a destination. Logging the pinned branch
        from the edge itself keeps the two screens comparable (#129).
        """
        if state.get("destination_id") is None:
            return "select_destination"
        with log_step(log, "Destination selection") as detail:
            detail["pinned_by_user"] = True
            detail["destination_id"] = state["destination_id"]
        return "forecast"

    def plan_trip(self, preferences: dict, destination_id: str = None, n_days: int = 3,
                  travel_month: str = None, top_k_pois: int = None,
                  daily_minutes_budget: int = 480, use_real_routing: bool = False,
                  travel_mode: str = DEFAULT_MODE,
                  day_start_hour: float = None, start_date=None,
                  arrival_hub_id: str = None) -> dict:
        """Same signature and return shape as RoamWiseOrchestrator.plan_trip();
        see that method's docstring for what each parameter does."""
        if top_k_pois is None:
            top_k_pois = max(MIN_RETRIEVED_POIS, n_days * RETRIEVED_POIS_PER_DAY)
        # One control, two consumers, exactly as orchestrator.py does it: the
        # month reaches the forecaster and the weekday reaches the router. This
        # derivation is the half of #76 that is easy to miss, because omitting
        # it leaves opening hours looking fixed *and* the crowding forecast
        # pointed at whatever month the caller happened to pass instead.
        if start_date is not None:
            travel_month = f"{start_date.year:04d}-{start_date.month:02d}"
        # The same opening record orchestrator.py writes, field for field, so
        # the System logs screen reads alike whichever path ran -- `orchestrator`
        # is what tells them apart (#129).
        log.info("Planning a %d-day trip", n_days, extra={"roamwise_fields": {
            "orchestrator": self.ORCHESTRATOR_ID,
            "destination": destination_id or "auto",
            "travel_month": travel_month or "flexible",
            "travel_mode": travel_mode,
            "daily_minutes_budget": daily_minutes_budget,
            "retrieval_config": self.retrieval_config,
            "top_k_pois": top_k_pois,
            "real_routing_requested": use_real_routing,
        }})

        init_state: PlanState = {
            "preferences": preferences, "n_days": n_days, "travel_month": travel_month,
            "top_k_pois": top_k_pois,
            "daily_minutes_budget": daily_minutes_budget, "use_real_routing": use_real_routing,
            "day_start_hour": day_start_hour,
            "travel_mode": travel_mode, "start_date": start_date,
            "arrival_hub_id": arrival_hub_id,
            "destination_id": destination_id,
        }
        return self._compiled.invoke(init_state)

    # ---- nodes --------------------------------------------------------------
    # Each node takes the accumulated state and returns only the keys it adds
    # or changes; LangGraph merges that into the running state for the next node.

    def _segment(self, state: PlanState) -> dict:
        with log_step(log, "Traveler segmentation (KMeans)") as detail:
            seg = self.segmenter.classify(state["preferences"])
            detail["archetype"] = seg["archetype"]
        return {"archetype": seg["archetype"], "segmentation": seg}

    def _select_destination(self, state: PlanState) -> dict:
        # The other half of this step is logged by `_needs_destination` above,
        # which is where the pinned branch is decided.
        with log_step(log, "Destination selection") as detail:
            detail["pinned_by_user"] = False
            destination_id = self._recommend_destination(state["preferences"],
                                                         state.get("travel_month"))
            detail["destination_id"] = destination_id
        return {"destination_id": destination_id}

    def _forecast(self, state: PlanState) -> dict:
        with log_step(log, "Forecaster Agent",
                      destination_id=state["destination_id"]) as detail:
            forecast = self.forecaster.run(state["destination_id"],
                                           travel_month=state.get("travel_month"))
            detail["crowding_level"] = forecast.get("crowding_level")
        return {"forecast": forecast}

    def _retrieve(self, state: PlanState) -> dict:
        with log_step(log, "Fusion RAG retrieval", config=self.retrieval_config) as detail:
            # Kept identical to orchestrator.py's query -- see #63 and the note
            # in HANDOFF about prompt/query drift between the two orchestrators.
            query = archetype_query(state["archetype"])
            rag = self.fusion_rag.run(
                query, destination_id=state["destination_id"], archetype=state["archetype"],
                config=self.retrieval_config, top_k=state.get("top_k_pois", 12),
                narrate=False, arrival_hub_id=state.get("arrival_hub_id"),
                start_date=state.get("start_date"),
            )
            detail["query"] = query
            detail["n_results"] = len(rag["results"])
            # Only when there is one, as orchestrator.py does it: `Arriving at`
            # defaults to "Already in the city", so an unconditional key would
            # print `arrival_hub_id=None` on nearly every plan.
            if state.get("arrival_hub_id"):
                detail["arrival_hub_id"] = state["arrival_hub_id"]
        return {"fusion_rag": rag}

    def _route(self, state: PlanState) -> dict:
        rag = state["fusion_rag"]
        candidate_pois = [
            self.graph_index.g.nodes[r["poi_id"]] | {"poi_id": r["poi_id"]}
            for r in rag["results"] if r.get("type") == "poi"
        ]
        if not candidate_pois:  # standard-prompting config: no retrieval, fall back to raw city POIs
            candidate_pois = self.graph_index.city_pois(state["destination_id"])[: state.get("top_k_pois", 12)]
            log.warning("No POIs retrieved -- falling back to the city's unfiltered top-rated list",
                        extra={"roamwise_fields": {"config": self.retrieval_config,
                                                   "n_fallback_pois": len(candidate_pois)}})

        # The price filter that used to sit here is gone; it never removed a
        # POI. Kept in step with orchestrator.py, which is the whole point of
        # this file (#67).

        travel_mode = state.get("travel_mode", DEFAULT_MODE)
        with log_step(log, "Router Agent (scoring + TOPTW solve)",
                      n_candidate_pois=len(candidate_pois), travel_mode=travel_mode) as detail:
            routing = self.router.run(
                state["destination_id"], candidate_pois, n_days=state["n_days"],
                daily_minutes_budget=state.get("daily_minutes_budget", 480),
                day_start_hour=state.get("day_start_hour"), archetype=state.get("archetype"),
                use_real_routing=state.get("use_real_routing", False),
                travel_mode=travel_mode,
                narrate=False, preferences=state.get("preferences"),
                start_date=state.get("start_date"),
                arrival_hub_id=state.get("arrival_hub_id"),
            )
            got_real_routing = any(d.get("used_real_routing") for d in routing["itinerary"])
            detail["n_pois_routed"] = sum(len(d["route"]) for d in routing["itinerary"])
            detail["day_start_hour"] = routing["day_start_hour"]
            detail["arrives_at"] = routing["itinerary"][0].get("starts_from") if routing["itinerary"] else None
            detail["used_real_routing"] = got_real_routing

        if state.get("use_real_routing", False) and not got_real_routing:
            log.warning("Real street routing was requested but no committed street network "
                        "covers these points -- distances fall back to the straight-line "
                        "estimate")

        # `free_entry_share` is the honest remainder of what the catalogue knows
        # about cost, and the traveler-facing summary reads it off the plan --
        # so a path that omitted it would show a blank where the other shows a
        # share. See orchestrator.py's node 4 for what the number means (#67).
        return {"routing": routing, "travel_mode": routing["travel_mode"],
                "free_entry_share": _free_entry_share(routing["itinerary"])}

    def _synthesize(self, state: PlanState) -> dict:
        """Kept deliberately identical to orchestrator._synthesize -- see that
        method's docstring for why the retrieval context is excluded (#56) and
        why this is a join rather than an indented f-string (#22)."""
        with log_step(log, "Final synthesis (LLM)", llm=type(self.llm).__name__) as detail:
            city = self.destinations.set_index("destination_id").loc[state["destination_id"], "city"]
            prompt = "\n\n".join([
                f"Trip plan for a {state['archetype']} traveler visiting {city} ({state['destination_id']}).",
                f"Forecast: {state['forecast']['narrative']}",
                state["routing"]["facts"],
            ])
            completion = self.llm.complete_verbose(
                system="You are RoamWise, an agentic travel-planning assistant. Write a coherent "
                       "recommendation for the itinerary below, describing its stops in the order "
                       "given and working in the forecast's timing advice. The itinerary is the "
                       "complete plan: describe only the stops it lists, and never mention or "
                       "suggest any other place, attraction or venue.",
                prompt=prompt,
            )
            detail["truncated"] = completion.truncated
        return {"final_plan": completion.text,
                "final_plan_truncated": completion.truncated}

    # ---- helpers (identical to orchestrator.py -- see that file's docstring
    # for why this small overlap is duplicated rather than shared: the two
    # orchestrators are meant to be independently comparable) --------------

    def _recommend_destination(self, preferences: dict, travel_month: str = None) -> str:
        budget_target = preferences.get("budget", 0.5) * 2 + 1
        scored = []
        for _, d in self.destinations.iterrows():
            tag_overlap = self._tag_affinity(preferences, d.tags)
            budget_fit = 1 - abs(d.budget_level - budget_target) / 2
            # Only crowding_level is read here, so this must not narrate.
            fc = self.forecaster.run(d.destination_id, travel_month=travel_month,
                                     horizon_months=6, narrate=False)
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
