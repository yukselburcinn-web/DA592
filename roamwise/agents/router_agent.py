"""Router Agent: acts as an algorithmic orchestrator. It does not ask the LLM
to invent a route -- it calls the POIZoner + 2-opt routing tool (a real
optimization method, not the LLM) to solve the routing problem logically,
using graph-enriched POI context, then narrates the resulting itinerary."""
from agents.llm_client import LLMClient, get_default_llm_client
from knowledge_graph.build_graph import GraphIndex
from models.segmentation import POIZoner
from optimization.routing import build_multi_day_itinerary


class RouterAgent:
    def __init__(self, graph_index: GraphIndex = None, llm: LLMClient = None):
        self.graph = graph_index or GraphIndex()
        self.zoner = POIZoner()
        self.llm = llm or get_default_llm_client()

    def run(self, destination_id: str, candidate_pois: list[dict], n_days: int,
            daily_minutes_budget: int = 480) -> dict:
        n_days = max(1, min(n_days, len(candidate_pois))) if candidate_pois else 1
        # Days start from the city's own center (proxy for a centrally booked
        # hotel), not the airport -- the airport hub only matters for the
        # single arrival leg, and anchoring every day there would make the
        # optimizer walk everyone back out to the edge of town daily.
        city_node = self.graph.g.nodes[destination_id]
        start_hub = {"lat": city_node["lat"], "lon": city_node["lon"], "name": city_node["name"]}

        zones = self.zoner.zone(candidate_pois, n_zones=n_days)
        itinerary = build_multi_day_itinerary(zones, start_hub=start_hub, daily_minutes_budget=daily_minutes_budget)
        narrative = self._narrate(destination_id, itinerary)
        return {"destination_id": destination_id, "itinerary": itinerary, "narrative": narrative}

    def _narrate(self, destination_id: str, itinerary: list[dict]) -> str:
        lines = []
        for day in itinerary:
            stops = " -> ".join(p["name"] for p in day["route"]) or "(no stops fit the time budget)"
            lines.append(f"Day {day['day']}: {stops}  [{day['distance_km']}km, ~{day['total_minutes']}min]")
        prompt = "Optimized itinerary for " + destination_id + ":\n" + "\n".join(lines)
        return self.llm.complete(system="Present the itinerary clearly, day by day.", prompt=prompt)


if __name__ == "__main__":
    graph = GraphIndex()
    agent = RouterAgent(graph)
    pois = graph.city_pois("PAR")
    result = agent.run("PAR", pois, n_days=3)
    print(result["narrative"])
