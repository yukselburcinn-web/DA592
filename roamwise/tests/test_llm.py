"""The LLM layer's output budget, truncation reporting and cost (#125), and an
opt-in that did not take effect being audible rather than silent (#133).
"""

import pytest

from roamwise.agents.llm_client import CHARS_PER_TOKEN, budget_for
from roamwise.agents.orchestrator import RoamWiseOrchestrator
from roamwise.tests.helpers import MAIN_CITY, MONDAY


# --- issue #125: the LLM layer's output budget, truncation reporting and cost ---

def _synthesis_prompt(n_days, city=None):
    """The exact text the narrator is handed, via TemplateLLMClient's verbatim
    echo -- the same trick test_synthesis_prompt_offers_no_place_outside_the_
    itinerary uses to assert on a real model's input with no model."""
    orch = RoamWiseOrchestrator()
    prefs = {"budget": 0.5, "culture": 0.9, "nature": 0.5,
             "nightlife": 0.2, "relax": 0.2, "adventure": 0.3}
    return orch.plan_trip(prefs, destination_id=city or MAIN_CITY,
                          n_days=n_days)["final_plan"]


@pytest.mark.slow
@pytest.mark.parametrize("n_days", [1, 2, 3, 4, 5])
def test_the_synthesis_budget_covers_every_trip_length(n_days):
    """Issue #125: the narrative described two days of a three-day route.

    The output cap was a flat 1024 tokens while the synthesis prompt grows with
    the trip -- ~520 tokens at one day, 1348 at three, 1982 at five. Three days
    of facts therefore could not even be restated inside the cap, and the
    generation stopped mid-sentence before reaching day 3. Two days measured
    966 tokens, just under the cap, which is exactly why the symptom was
    'the last day is missing' rather than 'the narrative is empty'.

    This is checkable with no model at all: whatever the narrator is asked to
    describe, its budget has to exceed it. The 5-day case is the boundary the
    old constant failed at worst, so the sweep goes to the trip length the UI
    allows, not just the one that was reported.
    """
    prompt = _synthesis_prompt(n_days)
    assert f"Day {n_days}" in prompt, \
        f"{n_days}-day plan should reach day {n_days} before the narrator sees it"
    # Estimated the way budget_for does, so the assertion tests the rule the
    # client actually applies rather than a parallel guess at tokenisation.
    approx_prompt_tokens = len(prompt) / CHARS_PER_TOKEN
    assert budget_for(prompt) > approx_prompt_tokens, (
        f"{n_days}-day synthesis gets {budget_for(prompt)} output tokens for a prompt of "
        f"~{approx_prompt_tokens:.0f} -- the narrative cannot restate every day it was given")


@pytest.mark.slow
def test_a_generation_that_hits_its_cap_is_reported_not_returned_as_finished():
    """Issue #125's second half: raising the cap alone would leave the same bug
    waiting at some longer trip. A cut generation returns well-formed prose
    that stops mid-sentence, so unless the client says it ran out of room,
    nothing downstream can tell a finished answer from half of one."""
    from roamwise.agents.llm_client import Completion, LLMClient

    class CutOffLLM(LLMClient):
        def complete_verbose(self, system, prompt, max_tokens=None):
            return Completion("Day 1: the Louvre, then lunch at", truncated=True)

    orch = RoamWiseOrchestrator(llm=CutOffLLM())
    prefs = {"budget": 0.6, "culture": 0.9, "nature": 0.2,
             "nightlife": 0.2, "relax": 0.3, "adventure": 0.2}
    result = orch.plan_trip(prefs, destination_id=MAIN_CITY, n_days=3)

    assert result["final_plan_truncated"] is True, \
        "a truncated narrative must be flagged so the UI can say the last days are missing"
    # And the ordinary path must not cry wolf.
    assert RoamWiseOrchestrator().plan_trip(
        prefs, destination_id=MAIN_CITY, n_days=3)["final_plan_truncated"] is False


@pytest.mark.slow
def test_choosing_a_destination_costs_no_generations():
    """Issue #125: #57 cut the two paraphrases nobody reads, but its test pins
    `destination_id`, which skips destination selection entirely -- and that is
    the path the app actually takes, since the sidebar defaults to letting
    RoamWise pick the city.

    Unpinned, `_recommend_destination` runs the forecaster over every city in
    the catalogue to read one field, `crowding_level`. The forecaster narrated
    unconditionally, so each candidate city cost a full generation whose prose
    was then discarded: N+1 generations per request, N of them invisible. Only
    the forecast blurb and the final plan are ever rendered.
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
    prefs = {"budget": 0.6, "culture": 0.9, "nature": 0.2,
             "nightlife": 0.2, "relax": 0.3, "adventure": 0.2}
    orch.plan_trip(prefs, n_days=3)  # unpinned: the app's own default

    assert len(llm.calls) == 2, (
        f"auto-selecting a destination should still cost only the forecast blurb and the "
        f"final plan, got {len(llm.calls)}: {llm.calls}")


def test_forecaster_can_score_without_narrating():
    """The flag the above relies on, asserted directly: scoring reads a number,
    so it must not spend a generation to get one."""
    from roamwise.agents.forecaster_agent import ForecasterAgent

    agent = ForecasterAgent()
    scored = agent.run(MAIN_CITY, narrate=False)
    assert scored["narrative"] is None
    assert scored["crowding_level"], "the field the scorer actually reads must survive"
    assert agent.run(MAIN_CITY)["narrative"], "narration stays on by default"


@pytest.mark.slow
def test_both_orchestrators_issue_the_same_synthesis_call():
    """HANDOFF's standing warning, made checkable. `orchestrator_langgraph.py`
    reimplements the same five nodes and a test already asserts interface
    parity -- but not that the two send the narrator the same thing, so drift
    in the prompt or in the call's parameters is silent. It has bitten three
    times (most recently the retrieval query, #63).

    Issue #125 changed this call in both files at once (complete -> the
    truncation-aware complete_verbose); this is what keeps the next such change
    from landing in only one of them.
    """
    pytest.importorskip("langgraph")
    from roamwise.agents.llm_client import Completion, LLMClient
    from roamwise.agents.orchestrator_langgraph import RoamWiseLangGraphOrchestrator

    class RecordingLLM(LLMClient):
        def __init__(self):
            self.synthesis = None

        def complete_verbose(self, system, prompt, max_tokens=None):
            if "agentic travel-planning assistant" in system:
                self.synthesis = {"system": system, "prompt": prompt,
                                  "max_tokens": max_tokens}
            return Completion(prompt, truncated=False)

    prefs = {"budget": 0.5, "culture": 0.9, "nature": 0.3,
             "nightlife": 0.2, "relax": 0.4, "adventure": 0.3}
    kwargs = dict(destination_id=MAIN_CITY, n_days=3, start_date=MONDAY)

    custom, graph = RecordingLLM(), RecordingLLM()
    RoamWiseOrchestrator(llm=custom).plan_trip(prefs, **kwargs)
    RoamWiseLangGraphOrchestrator(llm=graph).plan_trip(prefs, **kwargs)

    assert custom.synthesis is not None and graph.synthesis is not None, \
        "both orchestrators must reach the synthesis step"
    assert custom.synthesis == graph.synthesis, \
        "the two orchestrators send the narrator different calls -- they have drifted"


@pytest.mark.slow
def test_the_langgraph_path_also_reports_a_truncated_narrative():
    """The truncation flag is only useful if both orchestrators set it; the
    LangGraph twin returns its state as a dict of node outputs, which is
    exactly the kind of place a new field gets dropped."""
    pytest.importorskip("langgraph")
    from roamwise.agents.llm_client import Completion, LLMClient
    from roamwise.agents.orchestrator_langgraph import RoamWiseLangGraphOrchestrator

    class CutOffLLM(LLMClient):
        def complete_verbose(self, system, prompt, max_tokens=None):
            return Completion("Day 1: the Louvre, then lunch at", truncated=True)

    prefs = {"budget": 0.5, "culture": 0.9, "nature": 0.3,
             "nightlife": 0.2, "relax": 0.4, "adventure": 0.3}
    plan = RoamWiseLangGraphOrchestrator(llm=CutOffLLM()).plan_trip(
        prefs, destination_id=MAIN_CITY, n_days=3)

    assert plan["final_plan_truncated"] is True


def test_hitting_the_cap_logs_a_warning_an_operator_can_see(caplog):
    """The System logs screen reads this logger, so a cut narrative shows up
    there as a WARNING instead of being something a reader has to catch by
    noticing a sentence that stops mid-word."""
    import logging
    from roamwise.agents.llm_client import _warn_if_truncated

    with caplog.at_level(logging.WARNING, logger="roamwise.agents.llm_client"):
        _warn_if_truncated(True, "LocalHuggingFaceLLMClient", 2048)
    assert any(r.levelno == logging.WARNING and "2048" in r.getMessage()
               for r in caplog.records), "a truncated generation must be logged"

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="roamwise.agents.llm_client"):
        _warn_if_truncated(False, "LocalHuggingFaceLLMClient", 2048)
    assert not caplog.records, "a generation that finished must not warn"


# --- issue #133: an opt-in that did not take effect must be audible ---

def test_a_failed_local_llm_optin_warns_instead_of_falling_back_silently(monkeypatch, caplog):
    """Issue #133: `except Exception: pass` turned a missing package or a
    broken model cache into "the app works fine". TemplateLLMClient returns
    the prompt verbatim, so the run still produces confident-looking text --
    and a hallucination measurement taken on it would report exactly 0.0
    (#132). Asking for a model and not getting one has to be audible.
    """
    import logging
    from roamwise.agents import llm_client as mod

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ROAMWISE_LOCAL_LLM", "1")

    def explode(*a, **kw):
        raise ImportError("No module named 'mlx_lm'")

    monkeypatch.setattr(mod, "LocalHuggingFaceLLMClient", explode)

    with caplog.at_level(logging.WARNING, logger="roamwise.agents.llm_client"):
        client = mod.get_default_llm_client()

    assert isinstance(client, mod.TemplateLLMClient), "it must still fall back, just not quietly"
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "mlx_lm" in messages, "the swallowed exception must reach the log"
    assert "ROAMWISE_LOCAL_LLM" in messages, "the log must name the opt-in that did not take"
    # And the consequence, not only the fact -- this is what stops a template
    # run from being written up as a model run.
    assert "echoes the prompt" in messages


def test_the_deliberate_template_default_stays_silent(monkeypatch, caplog):
    """The other half, and the one that keeps the warning worth reading:
    running offline with no opt-in is the intended default (#54), so it must
    not warn. A warning on every start would make the real one invisible."""
    import logging
    from roamwise.agents.llm_client import TemplateLLMClient, get_default_llm_client

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ROAMWISE_LOCAL_LLM", raising=False)

    with caplog.at_level(logging.WARNING, logger="roamwise.agents.llm_client"):
        client = get_default_llm_client()

    assert isinstance(client, TemplateLLMClient)
    assert not caplog.records, f"the default path must not warn, got: {[r.getMessage() for r in caplog.records]}"


def test_the_ui_can_tell_a_silent_fallback_from_the_intended_default(monkeypatch):
    """`fallback_reason` is what the itinerary page branches on, so the
    distinction it draws is worth asserting directly: template-because-nobody-
    asked is normal, template-despite-an-opt-in is a failure to surface."""
    from roamwise.agents.llm_client import (TemplateLLMClient, describe_client,
                                            fallback_reason)

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ROAMWISE_LOCAL_LLM", raising=False)
    assert fallback_reason(TemplateLLMClient()) is None

    monkeypatch.setenv("ROAMWISE_LOCAL_LLM", "1")
    assert fallback_reason(TemplateLLMClient()) == "ROAMWISE_LOCAL_LLM is set"
    # A real client is never a fallback, whatever the environment says.

    class NotATemplate(TemplateLLMClient.__bases__[0]):
        def complete_verbose(self, system, prompt, max_tokens=None):
            raise NotImplementedError

    assert fallback_reason(NotATemplate()) is None
    assert "template" in describe_client(TemplateLLMClient())
