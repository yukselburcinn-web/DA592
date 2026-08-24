"""Router Agent: acts as an algorithmic orchestrator. It does not ask the LLM
to invent a route -- it calls the POIZoner + 2-opt routing tool (a real
optimization method, not the LLM) to solve the routing problem logically,
using graph-enriched POI context, then narrates the resulting itinerary."""
from roamwise.agents.llm_client import LLMClient, get_default_llm_client
from roamwise.knowledge_graph.build_graph import GraphIndex
from roamwise.models.segmentation import POIZoner
from roamwise.optimization.routing import (
    DEFAULT_DAY_START_HOUR, FOOD_CATEGORY, build_multi_day_itinerary)
from roamwise.optimization.travel_modes import DEFAULT_MODE, get_travel_mode

# Lunch and dinner: the itinerary should read like a day a person could
# actually live, not a march between museums (issue #20).
MIN_FOOD_PER_DAY = 2

# POI descriptions are full Wikipedia lead paragraphs -- several hundred words
# each. A dozen of them verbatim would dominate the synthesis prompt and, under
# a local model, cost real generation time for context the narrator only needs
# the gist of (issue #57). One sentence is enough to say what a place is.
_DESCRIPTION_CHARS = 200
# Cutting at the *first* ". " splits on abbreviations rather than sentences:
# "F. W. Borchardt" became "F." and "Pariser Platz (transl. ..." became
# "Pariser Platz (transl.". Real first sentences in this corpus run far longer
# than any abbreviation, so a boundary only counts once past this length.
_MIN_SENTENCE_CHARS = 60


def _clock(hour: float) -> str:
    """Fractional hour (13.5) -> wall clock ("13:30")."""
    hours, minutes = divmod(int(round(hour * 60)), 60)
    return f"{hours % 24:02d}:{minutes:02d}"


def _summarize(description) -> str:
    """First sentence of a POI description, bounded, as an inline clause."""
    if not description or not isinstance(description, str):
        return ""
    text = " ".join(description.split())  # collapse newlines so lines stay one-per-stop
    cutoff = text.find(". ", _MIN_SENTENCE_CHARS)
    if cutoff != -1 and cutoff <= _DESCRIPTION_CHARS:
        return f" -- {text[:cutoff + 1]}"
    if len(text) <= _DESCRIPTION_CHARS:
        return f" -- {text}"
    return f" -- {text[:_DESCRIPTION_CHARS].rstrip()}..."


class RouterAgent:
    def __init__(self, graph_index: GraphIndex = None, llm: LLMClient = None):
        self.graph = graph_index or GraphIndex()
        self.zoner = POIZoner()
        self.llm = llm or get_default_llm_client()

    def run(self, destination_id: str, candidate_pois: list[dict], n_days: int,
            daily_minutes_budget: int = 480, day_start_hour: float = DEFAULT_DAY_START_HOUR,
            respect_opening_hours: bool = True, use_real_routing: bool = False,
            travel_mode=DEFAULT_MODE, min_food_per_day: int = MIN_FOOD_PER_DAY,
            narrate: bool = True) -> dict:
        """narrate=False skips the LLM paraphrase and returns only `facts` --
        see FusionRAGAgent.run()'s docstring and issue #57."""
        n_days = max(1, min(n_days, len(candidate_pois))) if candidate_pois else 1
        mode = get_travel_mode(travel_mode)
        # Days start from the city's own center (proxy for a centrally booked
        # hotel), not the airport -- the airport hub only matters for the
        # single arrival leg, and anchoring every day there would make the
        # optimizer walk everyone back out to the edge of town daily.
        city_node = self.graph.g.nodes[destination_id]
        start_hub = {"lat": city_node["lat"], "lon": city_node["lon"], "name": city_node["name"]}

        # Retrieval can itself surface food-category POIs (markets, bazaars --
        # common for e.g. a Nightlife Seeker query), and those used to flow
        # into the same geography-only zoning/filling pass as everything
        # else. That pass has no notion of "these are meals": it can pack a
        # day with several of them back to back, or -- if a zone's other
        # candidates don't survive opening hours -- with nothing else at all
        # (#29). _ensure_daily_meals below is the only place that reasons
        # about meal spacing and the sightseeing floor, so food POIs are
        # excluded here and left entirely to it; _food_pois() already reuses
        # any of these by identity, so nothing retrieval surfaced is lost or
        # double-booked, just placed through the meal-aware path instead.
        sightseeing_pois = [p for p in candidate_pois if p.get("category") != FOOD_CATEGORY]
        zones = self.zoner.zone(sightseeing_pois, n_zones=n_days)
        itinerary = build_multi_day_itinerary(
            zones, start_hub=start_hub, daily_minutes_budget=daily_minutes_budget,
            day_start_hour=day_start_hour, respect_opening_hours=respect_opening_hours,
            use_real_routing=use_real_routing, travel_mode=mode,
            food_pois=self._food_pois(destination_id, candidate_pois),
            min_food_per_day=min_food_per_day,
        )
        facts = self._facts(destination_id, itinerary, mode)
        return {"destination_id": destination_id, "itinerary": itinerary,
                "travel_mode": mode.key, "facts": facts,
                "narrative": self._narrate(facts) if narrate else None}

    def _food_pois(self, destination_id: str, candidate_pois: list[dict]) -> list[dict]:
        """Meal candidates come straight from the knowledge graph rather than
        from retrieval: retrieval is archetype-driven, so a Culture
        Enthusiast's query surfaces museums and the itinerary came back with
        no meals at all (issue #20). Widening the query to force food in
        would change what the Fusion/Hybrid/standard comparison is measuring,
        so meals are sourced separately and the retrieval layer is left
        alone. Any food POI retrieval *did* surface is reused by identity, so
        the router doesn't schedule the same restaurant twice under two
        different dict objects."""
        already = {p.get("poi_id"): p for p in candidate_pois
                   if p.get("category") == FOOD_CATEGORY}
        pois = self.graph.city_pois(destination_id, category=FOOD_CATEGORY)
        return [already.get(p.get("poi_id"), p) for p in pois]

    def _facts(self, destination_id: str, itinerary: list[dict], mode) -> str:
        """The itinerary as grounded, deterministic text -- one line per stop.

        Each stop carries its own description rather than leaving downstream
        text to source those separately: the final synthesis used to be handed
        the *retrieval* context alongside this, which is a different and wider
        set of places (most retrieved candidates never survive the router's
        time/opening-hour constraints), and the model treated both lists as
        things to recommend (issue #56). Everything a narrator needs to
        describe this plan is therefore here, tied to a stop that is actually
        in it.
        """
        lines = [f"Optimized itinerary for {destination_id} ({mode.label.lower()}):"]
        for day in itinerary:
            routing_note = "real street routing" if day.get("used_real_routing") else "straight-line estimate"
            hours, minutes = divmod(day["total_minutes"], 60)
            lines.append(f"\nDay {day['day']} ({day['distance_km']}km, {hours}h {minutes}m, {routing_note}):")
            if not day["route"]:
                lines.append("  (no stops fit the time budget)")
                continue
            schedule = day.get("schedule", [])
            for i, poi in enumerate(day["route"], 1):
                slot = schedule[i - 1] if i <= len(schedule) else None
                when = f"{_clock(slot['arrival'])} " if slot else ""
                lines.append(f"  {i}. {when}{poi['name']} ({poi.get('category', 'stop')})"
                             f"{_summarize(poi.get('description'))}")
        return "\n".join(lines)

    def _narrate(self, facts: str) -> str:
        return self.llm.complete(system="Present the itinerary clearly, day by day.", prompt=facts)


if __name__ == "__main__":
    graph = GraphIndex()
    agent = RouterAgent(graph)
    pois = graph.city_pois("PAR")
    result = agent.run("PAR", pois, n_days=3)
    print(result["narrative"])
