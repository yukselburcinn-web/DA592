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

from roamwise.app_logging import get_logger, log_step
from roamwise.agents.forecaster_agent import ForecasterAgent
from roamwise.agents.fusion_rag_agent import FusionRAGAgent
from roamwise.agents.llm_client import LLMClient, get_default_llm_client
from roamwise.agents.router_agent import RouterAgent
from roamwise.knowledge_graph.build_graph import GraphIndex
from roamwise.models.segmentation import TravelerSegmenter
from roamwise.retrieval.query import archetype_query
from roamwise.optimization.travel_modes import DEFAULT_MODE
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

# Retrieval used to return a flat 12 POIs no matter how long the trip was, so
# a 5-day plan started with barely two stops per day and most of each day's
# budget went unused (issue #19). Scale with the trip instead, with headroom
# for the ones the router will drop on opening hours or the time budget.
#
# 24, not the 8 this was until #72. That 8 was never chosen deliberately, and
# it sat at one end of a frontier that trades two things the project already
# reports against each other. Measured on the pre-#72 router over 72 days,
# raising the pool moves them in opposite directions:
#
#   POIs/day    stops/day   km/stop   preference match   categories/day
#        8          6.21      1.582        0.752              1.68
#       16          7.25      1.248        0.648              2.57
#       24          7.82      1.054        0.584              2.86
#       40          8.43      0.908        0.536              3.33
#       60          9.22      0.659        0.485              4.21
#
# km/stop is what the comparative analysis reports as *itinerary coherence*,
# and preference match is what retrieval's own relevance measures cover, so
# the knob was quietly picking one over the other with nobody deciding.
#
# What makes 24 the right point now rather than before: selection used to be
# retrieval's job, because the router could not do it -- it routed whatever
# it was handed. TOPTW plus the score choose from the pool, so a wider pool
# is something to choose *from* rather than something to get through. The
# gain against the pre-#72 router roughly doubles at 24 over 8 (+15.8% vs
# +7.8% stops, -55.3% vs -42.7% km/stop). Past 24 the falling preference
# match stops being worth the diminishing geometry, and the solver's own
# ceiling starts to matter -- node count grows with pool x days.
RETRIEVED_POIS_PER_DAY = 24
MIN_RETRIEVED_POIS = 12

log = get_logger(__name__)



def _free_entry_share(itinerary: list[dict]) -> float | None:
    """Fraction of the plan's stops that cost nothing to enter, or None if the
    plan has no stops. `price_level` is a free/paid flag rather than a tier --
    see `plan_trip`."""
    stops = [poi for day in itinerary for poi in day["route"]]
    if not stops:
        return None
    return sum(1 for poi in stops if poi.get("price_level", 0) == 0) / len(stops)


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
                  travel_month: str = None, top_k_pois: int = None,
                  daily_minutes_budget: int = 480, use_real_routing: bool = False,
                  travel_mode: str = DEFAULT_MODE, day_start_hour: float = None,
                  start_date=None, arrival_hub_id: str = None) -> dict:
        """preferences: {budget, culture, nature, nightlife, relax, adventure} in [0,1].
        destination_id: pin a city, or leave None to let the orchestrator pick one.
        top_k_pois: how many POIs to retrieve; defaults to scaling with trip length
        (see RETRIEVED_POIS_PER_DAY) so a longer trip actually has enough candidates
        to fill its days.
        daily_minutes_budget: sightseeing time available per day, fed to the 2-opt router.
        day_start_hour: what time each day begins. Used to sit only on
        RouterAgent.run()'s signature with nothing able to reach it, so every
        itinerary began at 09:00 whatever the traveler wanted (issue #59).
        Left as None it now comes from the archetype (router_agent.DAY_START_HOURS):
        a 09:00 default gave a Nightlife Seeker nine hours with nothing
        schedulable in them, and a day holding one bar (issue #61).
        travel_mode: "walking", "driving" or "hybrid" -- how legs between stops are
        costed, which decides how much of a day's budget travel consumes.
        use_real_routing: use real street-network distances/times instead of the
        haversine + flat-speed estimate. Computed from OpenStreetMap data
        committed to this repo (issue #32), so it needs no network; it falls
        back automatically for any point no committed city network covers.
        start_date: the trip's first day, as a date. It feeds two things at once --
        the forecaster reads its month, and the router reads its day of the week,
        which is what lets opening hours be honoured per day rather than as one
        open/close pair (issue #70). `travel_month` remains for callers that only
        care about the forecast; a start_date supersedes it.
        arrival_hub_id: a `transport.csv` id -- the gateway the traveler lands at.
        Day 1 then starts from the airport/station rather than from the city
        centre, which is the single arrival leg issue #32 asks for. Left None,
        the trip begins in the city (right for someone already there)."""
        if top_k_pois is None:
            top_k_pois = max(MIN_RETRIEVED_POIS, n_days * RETRIEVED_POIS_PER_DAY)
        # One control, two consumers: a date carries the month the forecaster
        # wants and the weekday the router needs, so the traveler is not asked
        # for both.
        if start_date is not None:
            travel_month = f"{start_date.year:04d}-{start_date.month:02d}"
        state: dict = {"preferences": preferences, "n_days": n_days}

        log.info("Planning a %d-day trip", n_days, extra={"roamwise_fields": {
            "destination": destination_id or "auto",
            "travel_month": travel_month or "flexible",
            "travel_mode": travel_mode,
            "daily_minutes_budget": daily_minutes_budget,
            "retrieval_config": self.retrieval_config,
            "top_k_pois": top_k_pois,
            "real_routing_requested": use_real_routing,
        }})

        # --- Node 1: traveler segmentation ---
        with log_step(log, "Traveler segmentation (KMeans)") as detail:
            seg = self.segmenter.classify(preferences)
            state["archetype"] = seg["archetype"]
            state["segmentation"] = seg
            detail["archetype"] = seg["archetype"]

        # --- Node 2: destination selection (Forecaster Agent as scorer) ---
        with log_step(log, "Destination selection") as detail:
            detail["pinned_by_user"] = destination_id is not None
            if destination_id is None:
                destination_id = self._recommend_destination(preferences, travel_month)
            state["destination_id"] = destination_id
            detail["destination_id"] = destination_id

        with log_step(log, "Forecaster Agent", destination_id=destination_id) as detail:
            state["forecast"] = self.forecaster.run(destination_id, travel_month=travel_month)
            detail["crowding_level"] = state["forecast"].get("crowding_level")

        # --- Node 3: Fusion RAG Agent retrieves grounded, archetype-aware POIs ---
        with log_step(log, "Fusion RAG retrieval", config=self.retrieval_config) as detail:
            # Built from the archetype's preferred categories, not its label:
            # interpolating the label produced "best culture enthusiast points of
            # interest and experiences", which BM25 answered with a television
            # channel whose description says "culture" (#63).
            query = archetype_query(seg["archetype"])
            # narrate=False: nothing downstream reads this agent's prose --
            # the UI shows the retrieved documents themselves, and _synthesize
            # narrates from the itinerary (issues #56, #57).
            rag = self.fusion_rag.run(
                query, destination_id=destination_id, archetype=seg["archetype"], config=self.retrieval_config, top_k=top_k_pois,
                narrate=False,
            )
            state["fusion_rag"] = rag
            detail["query"] = query
            detail["n_results"] = len(rag["results"])

        candidate_pois = [
            self.graph.g.nodes[r["poi_id"]] | {"poi_id": r["poi_id"]}
            for r in rag["results"] if r.get("type") == "poi"
        ]
        if not candidate_pois:  # standard-prompting config: no retrieval, fall back to raw city POIs
            candidate_pois = self.graph.city_pois(destination_id)[:top_k_pois]
            log.warning("No POIs retrieved -- falling back to the city's unfiltered top-rated list",
                        extra={"roamwise_fields": {"config": self.retrieval_config,
                                                   "n_fallback_pois": len(candidate_pois)}})

        # There was a price filter here, dropping POIs above `max_price_level`
        # (documented "1=budget, 3=splurge"). It never removed a single POI and
        # could not: `price_level` only ever holds 0 or 1, and the threshold
        # defaulted to 3, so the condition was true for every row. It is gone
        # rather than repaired because there is nothing to repair it against --
        # see `free_entry_share` below and REPORT.md section 5 (#67).

        # --- Node 4: Router Agent builds the optimized day-by-day route ---
        with log_step(log, "Router Agent (zoning + 2-opt)",
                      n_candidate_pois=len(candidate_pois), travel_mode=travel_mode) as detail:
            # narrate=False for the same reason as the retrieval node above:
            # _synthesize narrates from this agent's `facts` directly, so an
            # LLM paraphrase here would only be read by another LLM (#57).
            routing = self.router.run(destination_id, candidate_pois, n_days=n_days,
                                       daily_minutes_budget=daily_minutes_budget,
                                       day_start_hour=day_start_hour,
                                       archetype=seg["archetype"],
                                       use_real_routing=use_real_routing, travel_mode=travel_mode,
                                       narrate=False, start_date=start_date,
                                       preferences=preferences,
                                       arrival_hub_id=arrival_hub_id)
            state["routing"] = routing
            state["travel_mode"] = routing["travel_mode"]
            # The honest remainder of what the catalogue knows about cost.
            # `price_level` is OSM's `fee` tag, so it says free or paid and
            # nothing finer -- it cannot separate a three-star restaurant from
            # a bistro, and all 61 food POIs carry the same value. Reported
            # rather than filtered on: a share the traveller can see is worth
            # more than a threshold that silently matched everything (#67).
            state["free_entry_share"] = _free_entry_share(routing["itinerary"])
            got_real_routing = any(d.get("used_real_routing") for d in routing["itinerary"])
            detail["n_pois_routed"] = sum(len(d["route"]) for d in routing["itinerary"])
            detail["day_start_hour"] = routing["day_start_hour"]
            detail["arrives_at"] = routing["itinerary"][0].get("starts_from") if routing["itinerary"] else None
            detail["used_real_routing"] = got_real_routing

        if use_real_routing and not got_real_routing:
            log.warning("Real street routing was requested but no committed street network "
                        "covers these points -- distances fall back to the straight-line "
                        "estimate")

        # --- Node 5: final synthesis ---
        with log_step(log, "Final synthesis (LLM)", llm=type(self.llm).__name__):
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
        """Narrate the plan from the itinerary alone.

        This deliberately does *not* include the Fusion RAG context. Retrieval
        returns candidates; the router then drops most of them on opening
        hours, travel time and the day budget -- on a measured 3-day Berlin
        plan, 4 of the 6 retrieved POIs handed to this prompt were not in the
        itinerary at all. Presented with two lists of places and no statement
        of which one was the plan, the model recommended from both, so the
        narrative sent users to stops the route never contained (issue #56).
        The itinerary is the only authoritative list of places, and it now
        carries each stop's own description (RouterAgent._facts), so nothing
        is lost by leaving the candidate set out.

        Built as a join, not an indented triple-quoted f-string: the injected
        facts are already flush-left, so splicing them into a further-indented
        template leaves the template's own lines indented and
        TemplateLLMClient's dedent can't undo that -- which renders as a
        Markdown code block in the UI instead of prose (issue #22).
        """
        city = self.destinations.set_index("destination_id").loc[state["destination_id"], "city"]
        prompt = "\n\n".join([
            f"Trip plan for a {state['archetype']} traveler visiting {city} ({state['destination_id']}).",
            f"Forecast: {state['forecast']['narrative']}",
            state["routing"]["facts"],
        ])
        return self.llm.complete(
            system="You are RoamWise, an agentic travel-planning assistant. Write a coherent "
                   "recommendation for the itinerary below, describing its stops in the order "
                   "given and working in the forecast's timing advice. The itinerary is the "
                   "complete plan: describe only the stops it lists, and never mention or "
                   "suggest any other place, attraction or venue.",
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
