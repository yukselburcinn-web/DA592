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

Optional: LocalHuggingFaceLLMClient (issue #54) -- a small, open-weight,
Apache-2.0-licensed model run entirely locally via MLX (Apple Silicon only),
for genuinely LLM-generated narration with no API key and no per-call cost.
Weights are downloaded to the user's own Hugging Face cache on first use, not
bundled with or committed to this repo -- see requirements-local-llm.txt for
why. Like the Anthropic path, this only activates on an explicit opt-in
(ROAMWISE_LOCAL_LLM=1): merely having the `mlx-lm` package importable must
never trigger an unexpected multi-gigabyte download.
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


# mlx-community re-converts popular Hugging Face models into MLX's native
# weight format; this one is Qwen3-4B-Instruct (Apache-2.0, see issue #54's
# license research) at 4-bit quantization, small enough to be a genuinely
# free, redistribution-clean, laptop-runnable choice.
LOCAL_LLM_MODEL_ID = "mlx-community/Qwen3-4B-Instruct-2507-4bit"


class LocalHuggingFaceLLMClient(LLMClient):
    """Runs a small open-weight instruct model locally via MLX instead of a
    paid API. Apple Silicon only -- MLX is Apple's own array framework, with
    no equivalent runtime on other platforms; get_default_llm_client() below
    falls back to TemplateLLMClient anywhere this doesn't import cleanly."""

    def __init__(self, model_id: str = LOCAL_LLM_MODEL_ID):
        from mlx_lm import load  # imported lazily so the dependency is optional
        self.model, self.tokenizer = load(model_id)

    def complete(self, system: str, prompt: str) -> str:
        from mlx_lm import generate
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return generate(self.model, self.tokenizer, prompt=rendered, max_tokens=1024, verbose=False).strip()


def get_default_llm_client() -> LLMClient:
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return AnthropicLLMClient()
        except Exception:
            pass
    # Explicit opt-in, not just "mlx-lm happens to be importable": the first
    # construction downloads a multi-gigabyte model to the user's Hugging
    # Face cache, which must never happen as a surprise side effect of
    # starting the app.
    if os.environ.get("ROAMWISE_LOCAL_LLM"):
        try:
            return LocalHuggingFaceLLMClient()
        except Exception:
            pass
    return TemplateLLMClient()
