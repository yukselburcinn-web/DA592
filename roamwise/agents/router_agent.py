"""Router Agent: acts as an algorithmic orchestrator. It does not ask the LLM
to invent a route -- it scores the candidates against the traveler's own
preference vector, hands the best of them to a TOPTW solver (a real
optimization method, not the LLM), and narrates what comes back.

Two decisions, in that order, and the split is measured rather than assumed
(see optimization/scoring.py). The score chooses *which* places are worth
the traveler's time; the solver then decides which of those actually fit,
on which day, in which order and at what hour, optimising geometry and time
uniformly over them. Weighting the solver with the same scores as well was
tried and cost stops and distance while barely moving preference match --
the preference signal is spent once, at selection."""
from roamwise.agents.llm_client import LLMClient, get_default_llm_client
from roamwise.knowledge_graph.build_graph import GraphIndex
from roamwise.optimization.routing import DEFAULT_DAY_START_HOUR, FOOD_CATEGORY
from roamwise.optimization.scoring import (
    MAX_WORKING_SET, SELECTION_PER_DAY, select_by_score)
from roamwise.optimization.toptw import build_multi_day_itinerary
from roamwise.optimization.travel_modes import DEFAULT_MODE, get_travel_mode

# Lunch and dinner: the itinerary should read like a day a person could
# actually live, not a march between museums (issue #20).
MIN_FOOD_PER_DAY = 2

# What time of day a traveler's day begins, by archetype. #59 made the start
# hour settable and plumbed it through; this decides what it should *default*
# to, which is what the reported symptom turned on (#61). Nightlife is never
# scheduled before NIGHTLIFE_EARLIEST_HOUR, so a Nightlife Seeker whose day
# opens at 09:00 spends its first nine hours with nothing schedulable in them
# and comes back holding one bar: measured over Paris and Berlin, a 3-day
# 12-hour trip returned [1, 1, 1] stops from 09:00 and [4, 4, 3] from 14:00.
# The archetype already knows this; the traveler should not have to work it out
# from a slider.
DAY_START_HOURS = {
    "Nightlife Seeker": 15.0,
    "Culture Enthusiast": 9.0,
    "Beach & Relax": 10.0,
    "Budget Backpacker": 8.0,
    "Luxury Traveler": 10.0,
    "Nature & Adventure": 7.0,
    "Family Traveler": 9.0,
}


def start_hour_for(archetype: str = None, override: float = None) -> float:
    """The hour a day begins: the traveler's own choice if they made one,
    otherwise their archetype's, otherwise the ordinary morning default."""
    if override is not None:
        return float(override)
    return DAY_START_HOURS.get(archetype, DEFAULT_DAY_START_HOUR)

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
        self.llm = llm or get_default_llm_client()

    def run(self, destination_id: str, candidate_pois: list[dict], n_days: int,
            daily_minutes_budget: int = 480, day_start_hour: float = None,
            archetype: str = None,
            respect_opening_hours: bool = True, use_real_routing: bool = False,
            travel_mode=DEFAULT_MODE, min_food_per_day: int = MIN_FOOD_PER_DAY,
            narrate: bool = True, start_date=None, preferences: dict = None,
            arrival_hub_id: str = None, hour_aware: bool = True) -> dict:
        """narrate=False skips the LLM paraphrase and returns only `facts` --
        see FusionRAGAgent.run()'s docstring and issue #57.

        day_start_hour=None lets the archetype pick it (see start_hour_for);
        pass a number to override that with the traveler's own choice.

        start_date is the trip's first day. It is what makes opening hours
        answerable -- a POI's hours are a rule over days of the week, and
        without a date the router can only read the coarse open/close pair and
        will happily schedule a Monday-closed museum on a Monday (issue #70).

        arrival_hub_id is a `transport.csv` id -- the airport, station or
        terminal the traveler lands at. Day 1 then starts there instead of at
        the city centre (issue #32). An id the city does not have is ignored
        rather than raised, the same way an unknown travel mode is, so a stale
        value from the UI can never break planning."""
        n_days = max(1, min(n_days, len(candidate_pois))) if candidate_pois else 1
        mode = get_travel_mode(travel_mode)
        day_start_hour = start_hour_for(archetype, day_start_hour)
        # Days start from the city's own center (proxy for a centrally booked
        # hotel), not the airport -- the airport hub only matters for the
        # single arrival leg, and anchoring every day there would make the
        # optimizer walk everyone back out to the edge of town daily.
        city_node = self.graph.g.nodes[destination_id]
        start_hub = {"lat": city_node["lat"], "lon": city_node["lon"], "name": city_node["name"]}
        # ...except day 1, if the traveler says where they are arriving. The
        # gateways are already in the graph as Transport nodes; this is the
        # single arrival leg the comment above has always meant.
        arrival_hub = None
        if arrival_hub_id:
            hub = next((h for h in self.graph.city_transport(destination_id)
                        if h["transport_id"] == arrival_hub_id), None)
            if hub:
                arrival_hub = {"lat": hub["lat"], "lon": hub["lon"], "name": hub["name"]}

        # Retrieval can itself surface food-category POIs (markets, bazaars --
        # common for e.g. a Nightlife Seeker query). They are still kept apart
        # from the sightseeing pool, because "this is a meal" is a fact the
        # optimizer needs and the category is the only thing carrying it;
        # _food_pois() reuses any retrieval surfaced by identity, so nothing is
        # lost or double-booked.
        sightseeing_pois = [p for p in candidate_pois if p.get("category") != FOOD_CATEGORY]
        food_pois = self._food_pois(destination_id, candidate_pois)
        sightseeing_pois, food_pois = self._select(
            sightseeing_pois, food_pois, n_days, preferences, min_food_per_day)

        # One model decides which of these to visit, on which day, in which
        # order and at what hour (issue #72). It replaces KMeans zoning plus
        # six passes -- day filling, day rebalancing, meal insertion, evening
        # insertion, a nightlife hour floor and a nightlife-last reorder --
        # which ran in sequence over each other's output and interacted: the
        # two-meal guarantee held on 28 of 72 measured days, and raising
        # min_food_per_day could *remove* a nightlife stop. See
        # optimization/toptw.py.
        itinerary = build_multi_day_itinerary(
            sightseeing_pois, n_days, start_hub=start_hub, arrival_hub=arrival_hub,
            daily_minutes_budget=daily_minutes_budget,
            day_start_hour=day_start_hour, respect_opening_hours=respect_opening_hours,
            use_real_routing=use_real_routing, travel_mode=mode,
            food_pois=food_pois, min_food_per_day=min_food_per_day,
            start_date=start_date, hour_aware=hour_aware,
        )
        facts = self._facts(destination_id, itinerary, mode)
        return {"destination_id": destination_id, "itinerary": itinerary,
                "travel_mode": mode.key, "day_start_hour": day_start_hour, "facts": facts,
                "start_date": start_date,
                "narrative": self._narrate(facts) if narrate else None}

    def _select(self, sightseeing: list[dict], food: list[dict], n_days: int,
                preferences: dict, min_food_per_day: int):
        """Shortlist both pools down to a working set the solver can hold.

        The score does the choosing, and it reads the traveler's six sliders
        directly rather than the archetype label they collapse to -- which is
        the point: two travelers who both land on "Culture Enthusiast" have
        different vectors and now get different itineraries, where before the
        label was the only thing that reached this agent.

        Measured against selecting by `archetype_query` retrieval at equal
        candidate count, scoring this way returned more stops a day at the
        same distance per stop, better-known places and more categories per
        day. Weighting the *solver* with the same scores, by contrast, cost
        stops and distance and barely moved preference match -- so the score
        selects, and the solver then optimises geometry and time uniformly
        over what it selected (see optimization/scoring.py).

        Without a preference vector there is nothing to score with, so both
        pools pass through trimmed only to what the solver can carry.
        """
        # Enough restaurants to have a real choice at each sitting without
        # crowding the sights out of the working set.
        food_limit = min(len(food), max(n_days * 4, 8)) if min_food_per_day > 0 else 0
        sight_limit = max(MAX_WORKING_SET - food_limit, n_days)
        sight_limit = min(sight_limit, SELECTION_PER_DAY * n_days)
        if not preferences:
            return sightseeing[:sight_limit], food[:food_limit]
        return (select_by_score(sightseeing, preferences, sight_limit),
                select_by_score(food, preferences, food_limit))

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
