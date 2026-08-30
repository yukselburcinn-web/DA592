"""Test-wide isolation from any live LLM (issue #7).

Most tests here build a `RoamWiseOrchestrator()` without an explicit `llm=`,
so they take whatever `get_default_llm_client()` returns. Before ROAMWISE_LLM
that was AnthropicLLMClient the moment ANTHROPIC_API_KEY was in the
environment -- so a developer who exports the key into their shell, which is
exactly what setting up the real integration involves, turned a plain
`pytest tests/` into 42 live API calls, measured on this suite. At a
40 requests-per-minute quota that is a whole minute of the budget spent on a
test run that never wanted a model.

Five of those calls also broke their tests rather than just costing money:
`_synthesis_prompt` in test_llm.py reads `final_plan` back as the verbatim
prompt, which only TemplateLLMClient does (see the note on it in CLAUDE.md). A
real model returns prose there, and the assertion that the prompt reaches
"Day N" fails for reasons that have nothing to do with the code under test.

ROAMWISE_LLM fixed the default; this fixture makes it structural. All three
variables are cleared for every test, so no future one can reach a paid API by
forgetting to. Tests that need a particular environment set it themselves -- a
fixture runs before the test body, so their own monkeypatch still wins.

`ROAMWISE_LLM` has to be cleared too, and it is the easiest one to miss: it is
the switch that *authorises* spending, so a shell holding it is exactly the
shell a developer runs `pytest` from. The documented setup ends in
`set -a; source .env; set +a`, which exports `ROAMWISE_LLM=nvidia` into that
shell -- and the next `pytest tests/` in it would have gone to the paid
endpoint on every orchestrator that takes the default client. README says to
load it per shell rather than per machine, and that advice stands, but a
convention is not a guard.
"""
import pytest


@pytest.fixture(autouse=True)
def isolate_from_live_llm(monkeypatch):
    """No test may reach a paid API or trigger a model download by accident."""
    monkeypatch.delenv("ROAMWISE_LLM", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ROAMWISE_LOCAL_LLM", raising=False)
