"""The orchestrators end to end: what the narrator is shown (#56/#57), what a
generation costs, the LangGraph path, the router's stop ceiling, and the import
paths the documented commands rely on.

The comparative analysis that used to live here moved to `test_evaluation.py`
in #164 -- it tests `evaluation/`, not the orchestrators.
"""

import datetime
import importlib
import os
import subprocess
import sys

import pytest

from pathlib import Path
from roamwise.agents.orchestrator import RoamWiseOrchestrator
from roamwise.agents.router_agent import RouterAgent
from roamwise.knowledge_graph.build_graph import GraphIndex
from roamwise.optimization.routing import FOOD_CATEGORY, _opening_intervals
from roamwise.tests.helpers import CITY_CODES, MAIN_CITY, MONDAY


# --- issues #56 / #57: the final narrative must describe the itinerary and
# nothing else, and must not cost a generation per intermediate agent ---

def _routed_names(result):
    return {p["name"] for day in result["routing"]["itinerary"] for p in day["route"]}


def _retrieved_names(result):
    return {r["name"] for r in result["fusion_rag"]["results"] if r.get("name")}


def ungrounded_places(result) -> list[str]:
    """Retrieved-but-unrouted places the narrative names without cover.

    "Without cover" is the operative part. A stop's own grounded description
    may legitimately name a neighbour -- the Humboldt Forum's says it sits on
    Museum Island, so a narrative describing that stop says "Museum Island"
    while recommending nothing off-route. Flagging that would make the check
    fire on correct output. What must never happen is a place appearing that
    the model was never shown: that is either an invention or a leak from the
    candidate list issue #56 removed.
    """
    off_route = _retrieved_names(result) - _routed_names(result)
    shown = result["routing"]["facts"]
    return sorted(name for name in off_route
                  if name in result["final_plan"] and name not in shown)


@pytest.mark.slow
@pytest.mark.parametrize("city", CITY_CODES)
def test_synthesis_prompt_offers_no_place_outside_the_itinerary(city):
    """Issue #56: retrieval returns candidates, and the router drops most of
    them on opening hours and the time budget. Handing both lists to the
    narrator made it recommend stops the route never contained.

    Under TemplateLLMClient `final_plan` is the prompt verbatim, so asserting
    on it asserts on exactly what a real model would be shown -- the property
    that actually has to hold, and one no local-model download is needed to
    check. Every city is swept because whether a candidate survives routing
    is a per-city accident of geography and opening hours.
    """
    orch = RoamWiseOrchestrator()
    prefs = {"budget": 0.6, "culture": 0.9, "nature": 0.2, "nightlife": 0.2, "relax": 0.3, "adventure": 0.2}
    result = orch.plan_trip(prefs, destination_id=city, n_days=3)

    off_route = _retrieved_names(result) - _routed_names(result)
    assert off_route, f"{city}: retrieval should surface candidates the router drops -- else this proves nothing"
    assert not ungrounded_places(result), \
        f"{city}: places not in the itinerary were offered to the narrator: {ungrounded_places(result)}"


@pytest.mark.slow
def test_planning_spends_one_generation_per_user_visible_narrative():
    """Issue #57: a trip used to cost four sequential generations, two of
    which produced text no user ever sees -- the retrieval and routing
    paraphrases existed only to be pasted into the synthesis prompt. Only the
    forecast blurb and the final plan are rendered, so only those two may
    cost a model pass.

    This pins `destination_id`. The unpinned path -- what the app does by
    default -- is covered by test_choosing_a_destination_costs_no_generations,
    which is where the cost came back after #57 (issue #125).
    """
    from roamwise.agents.llm_client import Completion, LLMClient

    class CountingLLM(LLMClient):
        def __init__(self):
            self.calls = []

        def complete_verbose(self, system, prompt, max_tokens=None):
            self.calls.append(system)
            return Completion(prompt, truncated=False)

    llm = CountingLLM()
    orch = RoamWiseOrchestrator(llm=llm)
    prefs = {"budget": 0.6, "culture": 0.9, "nature": 0.2, "nightlife": 0.2, "relax": 0.3, "adventure": 0.2}
    orch.plan_trip(prefs, destination_id=MAIN_CITY, n_days=3)

    assert len(llm.calls) == 2, f"expected forecast + synthesis only, got {len(llm.calls)}: {llm.calls}"


def test_stop_descriptions_are_not_cut_at_an_abbreviation():
    """Splitting on the first ". " turned "F. W. Borchardt" into "F." and
    "Pariser Platz (transl. ..." into "Pariser Platz (transl.", so a stop's
    one-line description said nothing about it."""
    from roamwise.agents.router_agent import _summarize

    assert _summarize("F. W. Borchardt is a restaurant in Berlin, open since 1853. It moved twice.") \
        .startswith(" -- F. W. Borchardt is a restaurant")
    assert "Paris Square" in _summarize(
        "Pariser Platz (transl. Paris Square) is a square in central Berlin, by the "
        "Brandenburg Gate. It is named after the French capital.")
    assert _summarize(None) == ""
    assert _summarize("") == ""


# --- issue #173: the narrator invented walking times, because the only
# durations it was shown were arrival clocks ---

def test_a_stop_line_names_the_leg_that_reached_it_rather_than_a_bare_arrival():
    """The Reichstag case, reduced to the two schedule entries that produced
    it.

    The narrative said "a 64-minute walk brings you to the Brandenburg Gate":
    280 m, about four minutes. The 64 was 13:11 minus 12:07 -- an hour inside
    the Reichstag plus the walk -- because the facts block gave arrival times
    and nothing else, and a time difference with no unit named is the only
    thing the model had to work with.

    Every span the line reports now says what it is spent on, so there is no
    unattributed difference left to misread.
    """
    from roamwise.agents.router_agent import _leg_clause
    from roamwise.optimization.travel_modes import get_travel_mode

    walking = get_travel_mode("walking")
    reichstag = {"arrival": 12.116, "finish": 13.116, "leg_km": 0.9,
                 "leg_minutes": 12.0, "leg_mode": "walking"}
    gate = {"arrival": 13.186, "finish": 13.686, "leg_km": 0.28,
            "leg_minutes": 4.19, "leg_mode": "walking"}

    clause = _leg_clause(gate, reichstag, walking)
    assert "4 min walk" in clause and "0.3 km" in clause
    assert "from the previous stop" in clause
    assert "visit until 13:41" in clause
    assert "64" not in clause


def test_the_leg_clause_accounts_for_the_wait_as_well_as_the_walk():
    """Leg plus visit does not always add up to the next arrival: a venue that
    is not open yet makes the day wait (`_next_open_hour`, and the solver's
    time windows on the TOPTW path). Left unnamed, that remainder is one more
    span for a narrator to attribute to walking, so it is labelled too.

    Below the floor it is dropped -- a one-second gap is rounding, and
    "0 min of waiting" on every line is noise.
    """
    from roamwise.agents.router_agent import _leg_clause
    from roamwise.optimization.travel_modes import get_travel_mode

    walking = get_travel_mode("walking")
    previous = {"arrival": 9.0, "finish": 10.0, "leg_km": 0.0, "leg_minutes": 0.0, "leg_mode": None}
    waited = {"arrival": 11.0, "finish": 12.0, "leg_km": 0.5, "leg_minutes": 6.0, "leg_mode": "walking"}
    prompt_arrival = {"arrival": 10.1, "finish": 11.0, "leg_km": 0.5,
                      "leg_minutes": 6.0, "leg_mode": "walking"}

    assert "then 54 min of waiting" in _leg_clause(waited, previous, walking)
    assert "waiting" not in _leg_clause(prompt_arrival, previous, walking)


def test_the_leg_clause_reports_the_mode_each_leg_was_actually_priced_on():
    """Hybrid walks short legs and drives long ones, per leg (#159), and a
    transit leg's kilometres are the straight line rather than the route a
    service takes -- `_leg_caption` drops them in the UI for that reason, and
    a narrator that divides one by the other would be worse than a caption
    that invites it. `leg_mode` is None where `solve` was called without one,
    which is when the trip's own mode is the honest answer."""
    from roamwise.agents.router_agent import _leg_clause
    from roamwise.optimization.travel_modes import get_travel_mode

    hybrid, walking = get_travel_mode("hybrid"), get_travel_mode("walking")
    driven = {"arrival": 10.0, "finish": 11.0, "leg_km": 4.0, "leg_minutes": 17.6, "leg_mode": "driving"}
    ridden = {"arrival": 10.0, "finish": 11.0, "leg_km": 4.0, "leg_minutes": 22.0, "leg_mode": "transit"}
    unnamed = {"arrival": 10.0, "finish": 11.0, "leg_km": 0.9, "leg_minutes": 12.0, "leg_mode": None}

    assert "18 min drive, 4.0 km" in _leg_clause(driven, driven, hybrid)
    assert "22 min ride by transit" in _leg_clause(ridden, ridden, hybrid)
    assert "km" not in _leg_clause(ridden, ridden, hybrid)
    assert "12 min walk" in _leg_clause(unnamed, unnamed, walking)


def test_the_first_stop_of_a_day_is_reached_from_somewhere_or_from_nothing():
    """Day one of an arrival-hub trip starts at the hub, and the leg in from
    it is a real cost the traveler pays (#126). Every other day's first stop
    has no leg priced before it, and inventing "0 min" for it would be the
    same class of made-up duration this issue is about."""
    from roamwise.agents.router_agent import _leg_clause
    from roamwise.optimization.travel_modes import get_travel_mode

    walking = get_travel_mode("walking")
    from_hub = {"arrival": 9.4, "finish": 10.4, "leg_km": 1.8, "leg_minutes": 24.0,
                "leg_mode": "walking"}
    cold_start = {"arrival": 9.0, "finish": 10.0, "leg_km": 0.0, "leg_minutes": 0.0,
                  "leg_mode": None}

    assert "24 min walk, 1.8 km from Gare du Nord" in _leg_clause(
        from_hub, None, walking, starts_from="Gare du Nord")
    assert _leg_clause(cold_start, None, walking) == " [visit until 10:00]"


@pytest.mark.slow
def test_the_prompt_states_every_leg_the_schedule_priced(city=MAIN_CITY):
    """The end of the chain: what the schedule holds reaches the text a real
    model is shown.

    Under TemplateLLMClient `final_plan` is the synthesis prompt verbatim
    (CLAUDE.md gotcha 2), so this asserts on exactly what a model would see
    with no download and no key -- the same gift
    `test_synthesis_prompt_offers_no_place_outside_the_itinerary` uses.

    The assertion is per leg against `leg_minutes` rather than against a
    regex for "min walk": a format that printed the right words and the wrong
    number would satisfy the second and is precisely the failure #173 is.
    """
    from roamwise.agents.router_agent import _clock

    orch = RoamWiseOrchestrator()
    prefs = {"budget": 0.6, "culture": 0.9, "nature": 0.2, "nightlife": 0.2, "relax": 0.3, "adventure": 0.2}
    result = orch.plan_trip(prefs, destination_id=city, n_days=2)

    plan = result["final_plan"]
    legs = 0
    for day in result["routing"]["itinerary"]:
        for poi, slot in zip(day["route"], day["schedule"]):
            line = next(l for l in plan.splitlines() if poi["name"] in l and l.strip()[0].isdigit())
            assert f"visit until {_clock(slot['finish'])}" in line, line
            if slot["leg_minutes"] <= 0:
                continue
            legs += 1
            assert f"{round(slot['leg_minutes'])} min" in line, line
            assert f"{slot['leg_km']:.1f} km" in line, line
    assert legs, "a plan with no priced leg cannot show one"


@pytest.mark.slow
def test_itinerary_facts_describe_every_routed_stop():
    """The narrator can only describe stops from what it is given, so dropping
    the retrieval context (#56) is only safe if the itinerary carries each
    stop's own description."""
    orch = RoamWiseOrchestrator()
    prefs = {"budget": 0.6, "culture": 0.9, "nature": 0.2, "nightlife": 0.2, "relax": 0.3, "adventure": 0.2}
    result = orch.plan_trip(prefs, destination_id=MAIN_CITY, n_days=2)

    facts = result["routing"]["facts"]
    for name in _routed_names(result):
        assert name in facts, f"{name} is routed but missing from the itinerary facts"


@pytest.mark.slow
def test_langgraph_orchestrator_matches_custom_orchestrator_interface():
    """Skipped only where the langgraph extra is absent (see
    requirements-langgraph.txt); CI installs it, so this runs there. Confirms
    the LangGraph rewrite exposes the same plan_trip() shape and honors a
    pinned destination via its conditional edge (select_destination is
    skipped).

    Shape is all this checks. That is exactly why issue #76 survived it: a
    parameter missing from one orchestrator changes the itinerary, not the
    return type. The behavioural half is the test below."""
    pytest.importorskip("langgraph")
    from roamwise.agents.orchestrator_langgraph import RoamWiseLangGraphOrchestrator

    orch = RoamWiseLangGraphOrchestrator()
    prefs = {"budget": 0.7, "culture": 0.3, "nature": 0.2, "nightlife": 0.9, "relax": 0.2, "adventure": 0.3}

    result = orch.plan_trip(prefs, n_days=2)
    assert result["archetype"] == "Nightlife Seeker"
    assert result["destination_id"] in orch.destinations.destination_id.values
    assert len(result["routing"]["itinerary"]) == 2
    assert result["final_plan"]

    pinned = orch.plan_trip(prefs, destination_id=MAIN_CITY, n_days=2)
    assert pinned["destination_id"] == MAIN_CITY


def _stops_scheduled_while_closed(itinerary, start_date):
    """Stops whose arrival hour falls outside the venue's hours *for the
    weekday that day actually falls on*, read from the same helper the router
    schedules with.

    The date comes from `start_date` plus the day's index rather than from the
    itinerary's own `date` field, deliberately: an orchestrator that never
    passed the date leaves that field None, and reading it would make this
    check skip exactly the itineraries it exists to catch."""
    closed = []
    for offset, day in enumerate(itinerary):
        day_date = start_date + datetime.timedelta(days=offset)
        for poi, slot in zip(day["route"], day["schedule"]):
            intervals = _opening_intervals(poi, day_date=day_date)
            if not any(low <= slot["arrival"] <= high for low, high in intervals):
                closed.append((day_date, poi["name"]))
    return closed


@pytest.mark.slow
def test_the_router_reports_its_stops_against_a_ceiling():
    """Issue #77: a stop count has no scale on its own. `with_ceiling=True`
    re-solves the same shortlist with travel free and reports the trip against
    that bound.

    Two things have to hold for the number to mean anything. The ceiling is an
    upper bound, so it can never sit below the trip it bounds -- if it did, the
    "ratio" would be a comparison of two different problems. And it has to be
    opt-in: the relaxed solve explores a denser problem than the real one and
    costs more than the trip, so a caller that did not ask for it must not pay
    for it."""
    idx = GraphIndex()
    pois = idx.city_pois(MAIN_CITY)[:40]
    router = RouterAgent(idx)
    kwargs = dict(n_days=2, daily_minutes_budget=600, narrate=False, start_date=MONDAY)

    plain = router.run(MAIN_CITY, pois, **kwargs)
    assert plain["stops_ratio"] is None and plain["ceiling_stops"] is None
    assert "would fit with travel between them free" not in plain["facts"]

    annotated = router.run(MAIN_CITY, pois, with_ceiling=True, **kwargs)
    stops = sum(len(d["route"]) for d in annotated["itinerary"])
    assert stops > 0, "a trip with no stops cannot exercise the ratio"
    assert annotated["ceiling_stops"] >= stops
    assert annotated["stops_ratio"] == pytest.approx(stops / annotated["ceiling_stops"])
    assert 0 < annotated["stops_ratio"] <= 1
    # The denominator reaches the text the narrator is grounded on, which is
    # the point of carrying it at all.
    assert f"of the {annotated['ceiling_stops']} stops" in annotated["facts"]


@pytest.mark.slow
def test_the_ceiling_is_solved_over_the_shortlist_not_the_whole_pool():
    """The shortlist is what `select_by_score` leaves, and what it leaves out
    it leaves out deliberately -- that is the only place the traveler's sliders
    reach the plan (issue #80). Solving the whole pool would fold that choice
    into the ratio and make personalization read as solver waste, so the
    ceiling has to move with the shortlist rather than with the pool."""
    idx = GraphIndex()
    router = RouterAgent(idx)
    kwargs = dict(n_days=2, daily_minutes_budget=600, narrate=False,
                  start_date=MONDAY, with_ceiling=True)

    pool = idx.city_pois(MAIN_CITY)
    plan = router.run(MAIN_CITY, pool, **kwargs)

    # Rebuild the shortlist the way run() does and solve it with travel free.
    # The reported ceiling has to be that number: solving the pool instead
    # would count stops the scorer deliberately left out.
    from roamwise.optimization.toptw import solve as toptw_solve
    from roamwise.agents.router_agent import MIN_FOOD_PER_DAY

    sights = [p for p in pool if p.get("category") != FOOD_CATEGORY]
    short_sights, short_food = router._select(
        sights, router._food_pois(MAIN_CITY, pool), 2, None, MIN_FOOD_PER_DAY)
    node = idx.g.nodes[MAIN_CITY]
    expected = toptw_solve(
        list(short_sights) + list(short_food), 2,
        start_hub={"lat": node["lat"], "lon": node["lon"], "name": node["name"]},
        daily_minutes_budget=600, day_start_hour=plan["day_start_hour"],
        start_date=MONDAY, distance_fn=lambda a, b: 0.0, duration_fn=lambda a, b: 0.0,
        min_food_per_day=MIN_FOOD_PER_DAY)

    assert plan["ceiling_stops"] == sum(len(d["route"]) for d in expected)
    assert plan["ceiling_stops"] >= sum(len(d["route"]) for d in plan["itinerary"])
    # And it is well below the pool it was shortlisted from, which is the
    # whole reason the distinction matters.
    assert plan["ceiling_stops"] < len(pool)


@pytest.mark.slow
def test_both_orchestrators_honour_the_weekday_a_start_date_names():
    """Issue #76: `orchestrator.py` passed `start_date` to the router and
    `orchestrator_langgraph.py` did not, so the verbatim OSM tag could not be
    resolved against a weekday on that path and fell back to the coarse
    open/close pair -- `Tu-Su 10:00-18:00; Mo off` read as open on Monday.

    The interface test above could not see this: both orchestrators returned
    the same shape while one of them scheduled Monday-closed museums on a
    Monday. So this asserts the itinerary instead, on a real Monday, where the
    catalogue's signal is sharpest -- 70 of its POIs are shut that day.

    Both paths are asserted, not just the fixed one: the bug was a divergence
    between two files, and a test that only reads one of them cannot see the
    next divergence either."""
    pytest.importorskip("langgraph")
    from roamwise.agents.orchestrator_langgraph import RoamWiseLangGraphOrchestrator

    prefs = {"budget": 0.5, "culture": 0.9, "nature": 0.3,
             "nightlife": 0.2, "relax": 0.4, "adventure": 0.3}
    kwargs = dict(destination_id=MAIN_CITY, n_days=2, start_date=MONDAY)

    custom = RoamWiseOrchestrator().plan_trip(prefs, **kwargs)["routing"]["itinerary"]
    graph = RoamWiseLangGraphOrchestrator().plan_trip(prefs, **kwargs)["routing"]["itinerary"]

    assert sum(len(d["route"]) for d in graph) > 0, "a trip with no stops proves nothing"

    # The consequence first: this is what the bug actually did to travelers.
    assert _stops_scheduled_while_closed(custom, MONDAY) == []
    assert _stops_scheduled_while_closed(graph, MONDAY) == []
    # And the date reaches the itinerary, so the UI can show which day is which.
    assert [d["date"] for d in graph] == [d["date"] for d in custom] == [
        MONDAY, MONDAY + datetime.timedelta(days=1)]


def test_a_start_date_also_reaches_the_forecaster_on_the_langgraph_path():
    """The other half of #76, and the easier half to leave out: `start_date`
    supersedes `travel_month` in `orchestrator.py`, so a caller who gives a
    date gets crowding for that date's month. Without the same derivation the
    LangGraph path would forecast whatever month it was handed -- here, none at
    all."""
    pytest.importorskip("langgraph")
    from roamwise.agents.orchestrator_langgraph import RoamWiseLangGraphOrchestrator

    prefs = {"budget": 0.5, "culture": 0.9, "nature": 0.3,
             "nightlife": 0.2, "relax": 0.4, "adventure": 0.3}
    plan = RoamWiseLangGraphOrchestrator().plan_trip(
        prefs, destination_id=MAIN_CITY, n_days=1, start_date=MONDAY)

    assert plan["forecast"]["target_month"] == f"{MONDAY.year:04d}-{MONDAY.month:02d}"


def test_local_llm_is_opt_in_and_falls_back_safely(monkeypatch):
    """Issue #54: constructing LocalHuggingFaceLLMClient can trigger a
    multi-gigabyte model download, so get_default_llm_client() must never
    reach for it just because mlx-lm happens to be importable -- only on the
    explicit ROAMWISE_LOCAL_LLM opt-in. Runs regardless of whether mlx-lm is
    actually installed, unlike the test below."""
    from roamwise.agents.llm_client import TemplateLLMClient, get_default_llm_client

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ROAMWISE_LOCAL_LLM", raising=False)
    assert isinstance(get_default_llm_client(), TemplateLLMClient)


def test_local_llm_client_produces_real_generation():
    """Optional: skipped unless mlx-lm is installed (see
    requirements-local-llm.txt) -- this actually loads the model and runs
    generation, so it also downloads several GB on first run."""
    pytest.importorskip("mlx_lm")
    from roamwise.agents.llm_client import LocalHuggingFaceLLMClient

    client = LocalHuggingFaceLLMClient()
    out = client.complete(system="Reply with one short sentence.", prompt="Name a city in France.")
    assert out.strip()


# app.py is only the router since the System logs screen landed (#41); the
# import chain that reaches the agent/retrieval modules now hangs off the page
# scripts, so both have to be loaded to prove the app can actually start.
STREAMLIT_MODULES = ["roamwise.app", "roamwise.views.itinerary", "roamwise.views.system_logs"]


@pytest.mark.parametrize("module", STREAMLIT_MODULES)
def test_streamlit_app_imports(module):
    """Guards the failure mode where the whole suite is green but the app does
    not start at all: these modules' import chain reaches every agent/retrieval
    module, so a half-migrated import path breaks here instead of at launch."""
    importlib.import_module(module)


def test_internal_modules_are_loaded_under_a_single_import_path():
    """Every internal module must be reachable as `roamwise.<pkg>.<mod>` only.
    A bare `<pkg>.<mod>` entry in sys.modules means the same file was loaded
    twice under two names, which silently splits module state and breaks
    isinstance/monkeypatch across the two copies."""
    for module in STREAMLIT_MODULES:
        importlib.import_module(module)

    internal_packages = {
        "agents", "models", "optimization", "retrieval",
        "knowledge_graph", "evaluation", "data", "views",
    }
    duplicated = sorted(
        name for name in sys.modules
        if name.split(".")[0] in internal_packages
    )
    assert not duplicated, f"modules loaded outside the roamwise.* path: {duplicated}"


# Every file the README/Dockerfile/launch.json tell a user to run directly.
# Importing them the normal way (as `roamwise.<pkg>.<mod>`) proves nothing about
# these commands, because that path already has the repo root on sys.path.
DOCUMENTED_ENTRY_POINT_SCRIPTS = [
    "app.py",
    "evaluation/comparative_analysis.py",
    "evaluation/forecasting_comparison.py",
]

# Load the file the way `python <script>` does -- under a throwaway module name,
# so any `if __name__ == "__main__":` body stays unexecuted and only the imports
# and module-level code run.
_IMPORT_PROBE = (
    "import importlib.util, sys\n"
    "spec = importlib.util.spec_from_file_location('_entry_point_probe', sys.argv[1])\n"
    "spec.loader.exec_module(importlib.util.module_from_spec(spec))\n"
)


@pytest.mark.parametrize("script", DOCUMENTED_ENTRY_POINT_SCRIPTS)
def test_documented_entry_point_scripts_resolve_roamwise_imports(script):
    """Run as a script, sys.path[0] is the script's own directory, so the repo
    root -- where the `roamwise` package lives -- never enters the path unless
    the script puts it there. This is the exact reason the app used to die with
    `ModuleNotFoundError: No module named 'roamwise'` while the suite was green,
    so reproduce that sys.path instead of relying on pytest's.

    Only the roamwise import path is asserted on: these scripts may still exit
    non-zero for unrelated reasons (a missing optional dependency, no network),
    and that is not what this test is guarding.
    """
    path = Path(__file__).resolve().parents[1] / script
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}

    # cwd == the script's directory makes the child's sys.path[0] ('') resolve
    # to that directory, matching how the interpreter launches a script.
    proc = subprocess.run(
        [sys.executable, "-c", _IMPORT_PROBE, str(path)],
        cwd=path.parent, env=env, capture_output=True, text=True,
    )

    assert "No module named 'roamwise" not in proc.stderr, (
        f"`python {script}` cannot import the roamwise package:\n{proc.stderr}"
    )
