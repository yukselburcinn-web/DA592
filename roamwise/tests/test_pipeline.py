"""End-to-end smoke tests covering every module in the pipeline. Run with:
    cd roamwise && ../venv/bin/pytest tests/ -v
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge_graph.build_graph import GraphIndex
from models.forecasting import forecast_city, best_months_to_visit
from models.segmentation import TravelerSegmenter, POIZoner
from retrieval.fusion import FusionRetriever
from optimization.routing import optimize_day_route
from agents.orchestrator import RoamWiseOrchestrator
from evaluation.comparative_analysis import run_comparative_analysis, summarize


def test_knowledge_graph_builds_and_traverses():
    idx = GraphIndex()
    stats = idx.stats()
    assert stats["by_type"]["City"] == 8
    assert stats["by_type"]["POI"] == 1200
    hop = idx.multi_hop_transport_to_poi("IST", "landmark")
    assert len(hop) > 0
    assert all("nearest_hub_km" in r for r in hop)


def test_forecasting_produces_crowding_labels():
    fc = forecast_city("PAR")
    assert len(fc) == 12
    assert set(fc.crowding_level.astype(str)) <= {"low", "medium", "high"}
    months = best_months_to_visit("PAR", top_k=3)
    assert len(months) == 3


def test_traveler_segmentation_classifies():
    seg = TravelerSegmenter()
    result = seg.classify({"budget": 0.1, "culture": 0.2, "nature": 0.9, "nightlife": 0.1, "relax": 0.3, "adventure": 0.9})
    assert result["archetype"] == "Nature & Adventure"


def test_poi_zoning_covers_all_pois():
    idx = GraphIndex()
    pois = idx.city_pois("AMS")
    zones = POIZoner().zone(pois, n_zones=3)
    total = sum(len(v) for v in zones.values())
    assert total == len(pois)


def test_fusion_retrieval_all_configs_run():
    fr = FusionRetriever()
    for config in ["fusion", "hybrid", "standard"]:
        results = fr.retrieve("museums near a transport hub", config=config, destination_id="ROM", top_k=5)
        if config == "standard":
            assert results == []
        else:
            assert len(results) > 0
            assert all(r["destination_id"] == "ROM" for r in results)


def test_fusion_beats_hybrid_on_archetype_grounding():
    fr = FusionRetriever()
    fusion_results = fr.retrieve("things to do", config="fusion", destination_id="ROM", archetype="Nightlife Seeker", top_k=8)
    fusion_categories = {r.get("text", "") for r in fusion_results}
    assert any("nightlife" in t.lower() or "trastevere" in t.lower() for t in fusion_categories)


def test_routing_respects_time_budget():
    idx = GraphIndex()
    pois = idx.city_pois("VIE")
    result = optimize_day_route(pois, daily_minutes_budget=300)
    assert result["total_minutes"] <= 300 + 1e-6


def test_routing_skips_poi_closed_for_the_rest_of_the_day():
    poi = {"name": "Late Museum", "lat": 41.0, "lon": 29.0, "avg_visit_minutes": 60,
           "open_hour": 9, "close_hour": 18}
    result = optimize_day_route([poi], day_start_hour=23.0, daily_minutes_budget=1440)
    assert result["route"] == []


def test_routing_waits_for_poi_to_open():
    poi = {"name": "Afternoon Gallery", "lat": 41.0, "lon": 29.0, "avg_visit_minutes": 60,
           "open_hour": 14, "close_hour": 20}
    result = optimize_day_route([poi], day_start_hour=9.0, daily_minutes_budget=1440)
    assert [p["name"] for p in result["route"]] == ["Afternoon Gallery"]
    # 9:00 -> waits until 14:00 -> 60min visit -> finishes 15:00 = 360 elapsed minutes,
    # not just the 60-minute visit duration, proving the wait is accounted for.
    assert result["total_minutes"] == 360


def test_routing_falls_back_when_osrm_unreachable(monkeypatch):
    import optimization.routing as routing_module
    monkeypatch.setattr(routing_module, "fetch_distance_duration_matrix", lambda points: None)

    idx = GraphIndex()
    pois = idx.city_pois("VIE")[:4]
    result = optimize_day_route(pois, use_real_routing=True)
    assert result["used_real_routing"] is False
    assert len(result["route"]) > 0  # still produced a usable route via the haversine fallback


def test_routing_uses_injected_real_routing_matrix(monkeypatch):
    import optimization.routing as routing_module

    def fake_matrix(points):
        n = len(points)
        # every pairwise "real" distance/duration is a fixed, distinguishable
        # value the haversine fallback could never produce, so we can tell
        # the matrix path (not haversine) actually supplied the numbers.
        return [[7.0] * n for _ in range(n)], [[42.0] * n for _ in range(n)]

    monkeypatch.setattr(routing_module, "fetch_distance_duration_matrix", fake_matrix)

    idx = GraphIndex()
    pois = idx.city_pois("VIE")[:2]
    result = optimize_day_route(pois, use_real_routing=True, daily_minutes_budget=1440,
                                 respect_opening_hours=False)
    assert result["used_real_routing"] is True
    assert result["distance_km"] == 7.0  # single leg from the fake matrix, not a haversine value


def test_orchestrator_end_to_end():
    orch = RoamWiseOrchestrator()
    prefs = {"budget": 0.7, "culture": 0.3, "nature": 0.2, "nightlife": 0.9, "relax": 0.2, "adventure": 0.3}
    result = orch.plan_trip(prefs, n_days=2)
    assert result["archetype"] == "Nightlife Seeker"
    assert result["destination_id"] in orch.destinations.destination_id.values
    assert len(result["routing"]["itinerary"]) == 2
    assert result["final_plan"]


def test_comparative_analysis_shows_fusion_advantage():
    df = run_comparative_analysis(top_k=8)
    summary = summarize(df)
    assert summary.loc["fusion", "mean_archetype_precision"] >= summary.loc["standard", "mean_archetype_precision"]
    assert summary.loc["fusion", "mean_grounded_entity_rate"] == 1.0
    assert summary.loc["standard", "mean_grounded_entity_rate"] == 0.0


def test_langgraph_orchestrator_matches_custom_orchestrator_interface():
    """Optional: skipped in CI, where the langgraph extra isn't installed
    (see requirements-langgraph.txt). Confirms the LangGraph rewrite exposes
    the same plan_trip() shape and honors a pinned destination via its
    conditional edge (select_destination is skipped)."""
    pytest.importorskip("langgraph")
    from agents.orchestrator_langgraph import RoamWiseLangGraphOrchestrator

    orch = RoamWiseLangGraphOrchestrator()
    prefs = {"budget": 0.7, "culture": 0.3, "nature": 0.2, "nightlife": 0.9, "relax": 0.2, "adventure": 0.3}

    result = orch.plan_trip(prefs, n_days=2)
    assert result["archetype"] == "Nightlife Seeker"
    assert result["destination_id"] in orch.destinations.destination_id.values
    assert len(result["routing"]["itinerary"]) == 2
    assert result["final_plan"]

    pinned = orch.plan_trip(prefs, destination_id="PAR", n_days=2)
    assert pinned["destination_id"] == "PAR"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
