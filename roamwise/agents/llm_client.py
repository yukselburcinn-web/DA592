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
import logging
import os
import textwrap
from typing import NamedTuple

log = logging.getLogger(__name__)


class Completion(NamedTuple):
    """What a client produced, plus whether it was allowed to finish.

    `truncated` is the part callers previously had no way to see. A generation
    that runs into its output cap returns well-formed prose that simply stops
    mid-sentence, which is indistinguishable from a finished answer unless the
    client reports it -- so a half-written itinerary reached the UI looking
    complete (issue #125).
    """

    text: str
    truncated: bool


# How much room a generation gets, derived from the prompt rather than fixed.
#
# The synthesis prompt grows with the trip -- measured on the committed
# catalogue, one day of itinerary facts costs ~520 tokens and five days ~1980 --
# and the narrative has to restate every one of those stops as prose, so the
# output runs longer than the prompt, not shorter. A flat 1024-token cap
# therefore covered a 2-day trip (966 prompt tokens) and cut a 3-day one (1348)
# off mid-sentence: the narrative described two days of a three-day route and
# said nothing about the third (issue #125).
#
# Budgeting from the prompt is what keeps that from returning at some longer
# trip nobody tried. Every call gets room proportional to what it was asked to
# describe, instead of a constant that happened to fit the trip length someone
# tested once.
MAX_TOKENS_FLOOR = 1024      # short prompts still get a usable answer
MAX_TOKENS_CEILING = 8192    # a runaway generation is a bug, not a long trip
PROMPT_TO_OUTPUT_RATIO = 2.0
CHARS_PER_TOKEN = 4  # rough English-prose estimate; only ever used to size a budget


def budget_for(prompt: str) -> int:
    """Output-token budget for `prompt`, clamped to [FLOOR, CEILING].

    Deliberately an estimate from character count rather than a real tokenizer
    count: this has to work for AnthropicLLMClient too, which has no local
    tokenizer, and being generous by a few hundred tokens costs nothing while
    being short by one costs the last day of the itinerary.
    """
    wanted = int(len(prompt) / CHARS_PER_TOKEN * PROMPT_TO_OUTPUT_RATIO)
    return max(MAX_TOKENS_FLOOR, min(wanted, MAX_TOKENS_CEILING))


def _warn_if_truncated(truncated: bool, client: str, budget: int) -> None:
    """Truncation is a silent failure by nature, so it is logged loudly.

    The System logs screen reads this logger, which makes a cut narrative
    visible to an operator instead of something a reader has to notice by
    spotting a sentence that stops in the middle.
    """
    if truncated:
        log.warning(
            "%s hit its %d-token output cap; the answer is cut off mid-sentence",
            client, budget,
            extra={"roamwise_fields": {"client": client, "max_tokens": budget,
                                       "truncated": True}},
        )


class LLMClient:
    def complete(self, system: str, prompt: str, max_tokens: int = None) -> str:
        """The text alone, for the callers that cannot act on truncation."""
        return self.complete_verbose(system, prompt, max_tokens).text

    def complete_verbose(self, system: str, prompt: str,
                         max_tokens: int = None) -> Completion:
        raise NotImplementedError


class TemplateLLMClient(LLMClient):
    """A deterministic stand-in 'LLM' that composes grounded context into
    readable prose via templates. Used as the default so the full agentic
    pipeline runs offline with no API key.

    It applies no output cap, which is exactly why it cannot surface issue
    #125 on its own: offline the prompt *is* the answer, so nothing is ever
    cut. Asserting that `budget_for(prompt)` actually covers the prompt is how
    a test catches that risk without a model -- see
    test_the_synthesis_budget_covers_every_trip_length.
    """

    def complete_verbose(self, system: str, prompt: str,
                         max_tokens: int = None) -> Completion:
        return Completion(textwrap.dedent(prompt).strip(), truncated=False)


class AnthropicLLMClient(LLMClient):
    def __init__(self, model: str = "claude-sonnet-5"):
        import anthropic  # imported lazily so the dependency is optional
        self.client = anthropic.Anthropic()
        self.model = model

    def complete_verbose(self, system: str, prompt: str,
                         max_tokens: int = None) -> Completion:
        budget = max_tokens or budget_for(prompt)
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=budget,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        # The API says outright when it stopped because it ran out of room;
        # this used to be dropped on the floor with the rest of the response.
        truncated = resp.stop_reason == "max_tokens"
        _warn_if_truncated(truncated, "AnthropicLLMClient", budget)
        text = "".join(block.text for block in resp.content if hasattr(block, "text"))
        return Completion(text, truncated)


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
        self.model_id = model_id
        self.model, self.tokenizer = load(model_id)

    def complete_verbose(self, system: str, prompt: str,
                         max_tokens: int = None) -> Completion:
        # stream_generate rather than generate: generate() returns the text and
        # nothing else, so whether it stopped at an end-of-turn token or simply
        # ran out of budget is unrecoverable. The streaming form carries a
        # finish_reason -- "stop" for EOS, "length" for the cap (issue #125).
        from mlx_lm import stream_generate
        budget = max_tokens or budget_for(prompt)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        parts, finish_reason = [], None
        for chunk in stream_generate(self.model, self.tokenizer, prompt=rendered,
                                     max_tokens=budget):
            parts.append(chunk.text)
            finish_reason = chunk.finish_reason
        truncated = finish_reason == "length"
        _warn_if_truncated(truncated, "LocalHuggingFaceLLMClient", budget)
        return Completion("".join(parts).strip(), truncated)


def describe_client(client: LLMClient) -> str:
    """Short human label for whichever client is actually running.

    The UI shows this because "am I really on the local model?" was previously
    answerable only by finding the `llm=` field of one step on the System logs
    screen (issue #133).
    """
    if isinstance(client, LocalHuggingFaceLLMClient):
        return f"local open-weight model ({getattr(client, 'model_id', LOCAL_LLM_MODEL_ID)})"
    if isinstance(client, AnthropicLLMClient):
        return f"Anthropic API ({client.model})"
    return "deterministic template (no model)"


def fallback_reason(client: LLMClient) -> str:
    """The opt-in that was set but did not take effect, or None.

    Recomputed from the environment rather than remembered at construction:
    `views/itinerary.py` caches the orchestrator with `st.cache_resource`, so
    the client is shared across sessions and must not carry per-run state.

    Returns None when the template is running because nobody asked for
    anything else -- that is the deliberate default and must stay silent, or
    the warning that matters becomes background noise.
    """
    if not isinstance(client, TemplateLLMClient):
        return None
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "ANTHROPIC_API_KEY is set"
    if os.environ.get("ROAMWISE_LOCAL_LLM"):
        return "ROAMWISE_LOCAL_LLM is set"
    return None


def _warn_construction_failed(wanted: str, opt_in: str, exc: Exception) -> None:
    log.warning(
        "%s was requested via %s but could not be constructed (%s: %s)",
        wanted, opt_in, type(exc).__name__, exc,
        extra={"roamwise_fields": {"wanted_client": wanted, "opt_in": opt_in,
                                   "error": f"{type(exc).__name__}: {exc}"}},
    )


def get_default_llm_client() -> LLMClient:
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return AnthropicLLMClient()
        except Exception as exc:
            # Swallowing this whole was how a missing package or a broken model
            # cache turned into "the app works fine" (issue #133).
            _warn_construction_failed("AnthropicLLMClient", "ANTHROPIC_API_KEY", exc)
    # Explicit opt-in, not just "mlx-lm happens to be importable": the first
    # construction downloads a multi-gigabyte model to the user's Hugging
    # Face cache, which must never happen as a surprise side effect of
    # starting the app.
    if os.environ.get("ROAMWISE_LOCAL_LLM"):
        try:
            return LocalHuggingFaceLLMClient()
        except Exception as exc:
            _warn_construction_failed("LocalHuggingFaceLLMClient", "ROAMWISE_LOCAL_LLM", exc)

    client = TemplateLLMClient()
    reason = fallback_reason(client)
    if reason:
        # The consequence, not just the fact: the template returns the prompt
        # verbatim, so a run that lands here silently reports a hallucination
        # rate of exactly 0.0 and a narrative that is not model output at all.
        log.warning(
            "Falling back to TemplateLLMClient even though %s -- it echoes the prompt "
            "back instead of generating text, so no narrative or measurement from this "
            "run is model output", reason,
            extra={"roamwise_fields": {"client": "TemplateLLMClient",
                                       "requested_but_unavailable": reason}},
        )
    return client
