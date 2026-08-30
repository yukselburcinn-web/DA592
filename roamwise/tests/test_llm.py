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


@pytest.mark.slow
def test_the_comparative_analysis_spends_no_generations(monkeypatch):
    """The 201-call trap (issue #7).

    `run_comparative_analysis` runs every query through three configurations
    and routes a day from each one's candidates. It passes `narrate=False` at
    every step, so the whole measurement costs zero generations -- and the
    comment above that argument says why. But a comment is all that guards it:
    drop the argument and the table quietly starts firing one generation per
    query per configuration, which on the committed query set is 67 x 3 = 201
    calls. Under a 40-per-minute quota that is five minutes of a rebuild spent
    on prose nothing reads, and offline it is invisible, because
    TemplateLLMClient answers instantly and for free (see CLAUDE.md).

    The agents bind `get_default_llm_client` at import, so each module is
    patched where it looks the name up rather than at its source.
    """
    from roamwise.agents import fusion_rag_agent, router_agent
    from roamwise.agents.llm_client import Completion, LLMClient
    from roamwise.evaluation import comparative_analysis as ca

    calls = []

    class CountingLLM(LLMClient):
        def complete_verbose(self, system, prompt, max_tokens=None):
            calls.append(system)
            return Completion(prompt, truncated=False)

    for module in (router_agent, fusion_rag_agent):
        monkeypatch.setattr(module, "get_default_llm_client", lambda: CountingLLM())

    # One query is enough: the rule is per-step, not per-query, and the full
    # set costs minutes. Sliced rather than rewritten so the row still carries
    # a real query with a real answer key.
    committed_queries = len(ca.TEST_QUERIES)
    monkeypatch.setattr(ca, "TEST_QUERIES", ca.TEST_QUERIES[:1])
    results = ca.run_comparative_analysis()

    assert len(results) == len(ca.CONFIGS), "the slice should still measure every configuration"
    assert not calls, (
        f"the comparative analysis must not generate prose -- got {len(calls)} call(s) for "
        f"one query, which is {len(calls) * committed_queries} for the committed set")


# --- issue #7: a hosted model behind an explicit switch, a quota, and retries ---

def test_a_key_in_the_environment_is_not_permission_to_spend_it(monkeypatch):
    """Issue #7. Holding a key and asking to use it are different statements,
    and only one of them belongs in `.zshrc`.

    Before ROAMWISE_LLM, `get_default_llm_client()` routed to a paid API the
    moment ANTHROPIC_API_KEY existed anywhere in the environment -- so
    exporting a key, which is the normal way to configure the integration,
    silently billed every `pytest` run (42 calls), every evaluation script and
    every `python -m` invocation. The switch says what to use; the key says
    what is available.
    """
    from roamwise.agents.llm_client import TemplateLLMClient, get_default_llm_client

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-not-consent")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-not-consent")
    monkeypatch.delenv("ROAMWISE_LLM", raising=False)
    monkeypatch.delenv("ROAMWISE_LOCAL_LLM", raising=False)

    assert isinstance(get_default_llm_client(), TemplateLLMClient), \
        "a key alone must not select a paid client"


def test_a_misspelled_engine_is_refused_out_loud(monkeypatch, caplog):
    """A typo in ROAMWISE_LLM must not read as 'template, then'. The template
    echoes the prompt back, so a silent fall-through produces a run that looks
    like model output and is not -- the failure #133 exists to prevent."""
    import logging
    from roamwise.agents.llm_client import TemplateLLMClient, get_default_llm_client

    monkeypatch.setenv("ROAMWISE_LLM", "nvida")  # the transposition that will happen

    with caplog.at_level(logging.WARNING, logger="roamwise.agents.llm_client"):
        client = get_default_llm_client()

    assert isinstance(client, TemplateLLMClient)
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "nvida" in messages, "the log must quote the value that was not understood"
    assert "nvidia" in messages, "and list the ones that are"


def test_the_nvidia_client_needs_its_key_and_says_which_one(monkeypatch):
    """The opt-in that cannot be honoured has to name the missing piece --
    otherwise the UI reports 'no model is running' with nothing to act on."""
    from roamwise.agents.llm_client import NvidiaLLMClient

    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="NVIDIA_API_KEY"):
        NvidiaLLMClient()


def test_the_rate_limiter_waits_rather_than_exceeding_the_quota(monkeypatch):
    """Issue #7: NVIDIA's free tier allows 40 requests a minute.

    Normal use is nowhere near it -- a plan costs two generations -- but the
    limiter is what turns a burst into a slower run instead of a 429 in the
    middle of a traveler's plan. Driven on a fake clock: the point is that the
    fourth call inside a window of three waits out the window, not that the
    test spends a minute proving it.
    """
    from roamwise.agents import llm_client as lc

    clock, slept = [1000.0], []
    monkeypatch.setattr(lc.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(lc.time, "sleep", lambda s: (slept.append(s), clock.__setitem__(0, clock[0] + s)))

    limiter = lc._RateLimiter(3)
    for _ in range(3):
        limiter.wait()
    assert not slept, "calls inside the quota must not wait"

    limiter.wait()
    assert slept, "the call over the quota must wait"
    assert sum(slept) >= 60.0, f"it must wait out the window, slept {sum(slept)}s"

    # And after waiting, the window has rolled: the next call is free again.
    limiter.wait()
    assert len(slept) == 1


def test_a_disabled_rate_limiter_never_waits(monkeypatch):
    """ROAMWISE_LLM_RPM=0 is the escape hatch for a paid tier with no such
    quota. It has to mean 'no limit', not 'no calls'."""
    from roamwise.agents import llm_client as lc

    monkeypatch.setattr(lc.time, "sleep", lambda s: pytest.fail(f"waited {s}s with the limiter off"))
    limiter = lc._RateLimiter(0)
    for _ in range(50):
        limiter.wait()


class _FakeResponse:
    def __init__(self, status_code, payload=None, headers=None):
        self.status_code = status_code
        self.headers = headers or {}
        self.ok = 200 <= status_code < 300
        self.text = "" if payload is None else str(payload)
        self._payload = payload

    def json(self):
        return self._payload


def _nvidia_client(monkeypatch, responses):
    """An NvidiaLLMClient whose HTTP layer replays `responses` in order."""
    from roamwise.agents import llm_client as lc

    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    monkeypatch.setattr(lc.time, "sleep", lambda s: None)      # no real backoff in tests
    monkeypatch.setattr(lc._limiter, "wait", lambda: None)     # the limiter has its own test

    client = lc.NvidiaLLMClient()
    sent = []

    class _FakeSession:
        def post(self, url, json=None, timeout=None):
            sent.append(json)
            return responses.pop(0)

    client._session = _FakeSession()
    return client, sent


def _ok_payload(text, finish_reason="stop"):
    return {"choices": [{"message": {"content": text}, "finish_reason": finish_reason}]}


def test_a_429_is_retried_rather_than_shown_to_the_traveler(monkeypatch):
    """Issue #7: the quota can still be hit -- another process sharing the key,
    or a burst the limiter did not see. A retried 429 costs a pause; an
    unretried one costs the plan."""
    responses = [_FakeResponse(429, headers={"Retry-After": "1"}),
                 _FakeResponse(200, _ok_payload("Day 1: the Brandenburg Gate."))]
    client, sent = _nvidia_client(monkeypatch, responses)

    answer = client.complete_verbose(system="s", prompt="p")

    assert answer.text.startswith("Day 1"), "the retry's answer is the one that counts"
    assert not answer.truncated
    assert len(sent) == 2, "the failed attempt and the one that worked"


def test_a_rejected_request_is_not_retried_into_the_quota(monkeypatch):
    """A 401 is a bad key and a 400 is a bad request; neither improves on the
    second attempt, and retrying spends a quota that a working call will need."""
    from roamwise.agents.llm_client import LLMRequestFailed

    responses = [_FakeResponse(401, "invalid api key")]
    client, sent = _nvidia_client(monkeypatch, responses)

    with pytest.raises(LLMRequestFailed, match="401"):
        client.complete_verbose(system="s", prompt="p")
    assert len(sent) == 1, "a rejected request must be attempted exactly once"


def test_a_hosted_model_that_never_answers_raises_instead_of_returning_nothing(monkeypatch):
    """The failure has to be loud. An empty Completion is indistinguishable
    from a model with nothing to say, and the UI would render the gap as a
    finished plan -- the silent-failure shape of #125 and #133."""
    from roamwise.agents.llm_client import LLMRequestFailed, MAX_ATTEMPTS

    responses = [_FakeResponse(503) for _ in range(MAX_ATTEMPTS)]
    client, sent = _nvidia_client(monkeypatch, responses)

    with pytest.raises(LLMRequestFailed):
        client.complete_verbose(system="s", prompt="p")
    assert len(sent) == MAX_ATTEMPTS


def test_a_capped_nvidia_generation_reports_that_it_was_cut(monkeypatch):
    """Issue #125's rule, on the new client: a generation that ran out of room
    stops mid-sentence and otherwise reads like a finished answer."""
    responses = [_FakeResponse(200, _ok_payload("Day 1: the Louvre, then lunch at",
                                                finish_reason="length"))]
    client, _ = _nvidia_client(monkeypatch, responses)

    answer = client.complete_verbose(system="s", prompt="p")
    assert answer.truncated, "finish_reason=length is the endpoint saying it was cut"


def test_the_nvidia_request_carries_the_prompt_and_a_budget(monkeypatch):
    """The budget is derived from the prompt (#125), so it has to reach the
    wire -- a client that sends the default cap reintroduces the truncated
    five-day narrative."""
    from roamwise.agents.llm_client import budget_for

    responses = [_FakeResponse(200, _ok_payload("ok"))]
    client, sent = _nvidia_client(monkeypatch, responses)
    prompt = "Day 1: a long itinerary. " * 400

    client.complete_verbose(system="be brief", prompt=prompt)

    body = sent[0]
    assert body["max_tokens"] == budget_for(prompt)
    assert body["messages"][0] == {"role": "system", "content": "be brief"}
    assert body["messages"][1]["content"] == prompt


def test_thinking_is_off_by_default_and_switchable(monkeypatch):
    """Issue #7: the default model (nemotron-3.5-lightning) deliberates before
    answering unless told not to, and that deliberation is billed like any
    other output token while nothing downstream reads it. Off by default, but
    a body field rather than a hardcoded False, because a model that needs it
    should not need a new client."""
    responses = [_FakeResponse(200, _ok_payload("Day 1: the Louvre."))]
    client, sent = _nvidia_client(monkeypatch, responses)

    client.complete_verbose(system="s", prompt="p")
    assert sent[0]["chat_template_kwargs"] == {"enable_thinking": False}


def test_a_model_that_only_reasons_is_a_failure_not_an_empty_plan(monkeypatch):
    """The nastiest shape this endpoint can return: HTTP 200, finish_reason
    'length', an empty `content` and a full `reasoning_content` -- the model
    spent its whole budget thinking. Rendering that empty string would put a
    blank narrative under a "Written by: NVIDIA API" label, which reads as a
    finished plan (#125, #133)."""
    from roamwise.agents.llm_client import LLMRequestFailed

    payload = {"choices": [{"message": {"content": "",
                                        "reasoning_content": "Let me consider the days..."},
                            "finish_reason": "length"}]}
    client, _ = _nvidia_client(monkeypatch, [_FakeResponse(200, payload)])

    with pytest.raises(LLMRequestFailed, match="ROAMWISE_LLM_THINKING"):
        client.complete_verbose(system="s", prompt="p")

