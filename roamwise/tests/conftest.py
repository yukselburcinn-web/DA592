"""Test-wide isolation from any live LLM (issue #7).

`get_default_llm_client()` picks AnthropicLLMClient the moment
ANTHROPIC_API_KEY is in the environment, and 11 of the 15 orchestrators built
in `test_pipeline.py` are constructed without an explicit `llm=`. So a
developer who exports the key into their shell -- which is exactly what
setting up the real integration involves -- turns a plain `pytest tests/` into
42 live API calls, measured on this suite. At a 40 requests-per-minute quota
that is a whole minute of the budget spent on a test run that never wanted a
model.

Five of those calls also break their tests rather than just costing money:
`_synthesis_prompt` reads `final_plan` back as the verbatim prompt, which only
TemplateLLMClient does (see the note on it in CLAUDE.md). A real model returns
prose there, and the assertion that the prompt reaches "Day N" fails for
reasons that have nothing to do with the code under test.

Individual tests already deleted these variables one at a time -- see
`test_a_model_that_was_asked_for_and_did_not_load_is_reported` and its
neighbours, which still do, because they assert on the environment itself.
Doing it once, autouse, makes the rule structural instead of something every
new test has to remember. Tests that need a particular environment set it
themselves; a fixture runs before the test body, so their own monkeypatch
still wins.
"""
import pytest


@pytest.fixture(autouse=True)
def isolate_from_live_llm(monkeypatch):
    """No test may reach a paid API or trigger a model download by accident."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ROAMWISE_LOCAL_LLM", raising=False)
