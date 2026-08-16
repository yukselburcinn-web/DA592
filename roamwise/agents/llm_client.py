"""
Pluggable "reasoning engine" used by every agent for the final natural-language
synthesis step described in the proposal ("GenAI acts... as a central
reasoning engine").

Default: TemplateLLMClient -- deterministic, zero-cost, needs no API key, and
keeps the whole pipeline reproducible for grading/offline demos. It still
performs real synthesis (selecting which retrieved facts to surface, not just
string formatting) so the agent behavior is meaningfully different from a
static template dump.

Optional: AnthropicLLMClient -- if the user sets ANTHROPIC_API_KEY themselves
and has the `anthropic` package installed, agents will produce genuinely
LLM-generated narration instead. This is opt-in only; RoamWise never assumes
or requires network/API access, and never bundles a key.
"""
import os
import textwrap


class LLMClient:
    def complete(self, system: str, prompt: str) -> str:
        raise NotImplementedError


class TemplateLLMClient(LLMClient):
    """A deterministic stand-in 'LLM' that composes grounded context into
    readable prose via templates. Used as the default so the full agentic
    pipeline runs offline with no API key."""

    def complete(self, system: str, prompt: str) -> str:
        return textwrap.dedent(prompt).strip()


class AnthropicLLMClient(LLMClient):
    def __init__(self, model: str = "claude-sonnet-5"):
        import anthropic  # imported lazily so the dependency is optional
        self.client = anthropic.Anthropic()
        self.model = model

    def complete(self, system: str, prompt: str) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in resp.content if hasattr(block, "text"))


def get_default_llm_client() -> LLMClient:
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return AnthropicLLMClient()
        except Exception:
            pass
    return TemplateLLMClient()
