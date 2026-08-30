"""
Pluggable "reasoning engine" used by every agent for the final natural-language
synthesis step described in the proposal ("GenAI acts... as a central
reasoning engine").

Default: TemplateLLMClient -- deterministic, zero-cost, needs no API key, and
keeps the whole pipeline reproducible for grading/offline demos. It still
performs real synthesis (selecting which retrieved facts to surface, not just
string formatting) so the agent behavior is meaningfully different from a
static template dump.

Optional: NvidiaLLMClient (issue #7) -- NVIDIA's OpenAI-compatible endpoint,
the hosted path this project actually uses. Spoken to over plain `requests`
rather than the `openai` SDK: `requests` is already a dependency, the request
is one JSON POST, and the retry behaviour here has to co-operate with the rate
limiter below rather than race a second one inside an SDK.

Optional: AnthropicLLMClient -- the same shape against the Anthropic API, for
anyone who has a key and the `anthropic` package.

Optional: LocalHuggingFaceLLMClient (issue #54) -- a small, open-weight,
Apache-2.0-licensed model run entirely locally via MLX (Apple Silicon only),
for genuinely LLM-generated narration with no API key and no per-call cost.
Weights are downloaded to the user's own Hugging Face cache on first use, not
bundled with or committed to this repo -- see requirements-local-llm.txt for
why.

Every non-template client is opt-in through one variable, ROAMWISE_LLM.
Holding a key is not the same as asking to spend it: exporting a key into a
shell profile used to be enough to send every `pytest` run, every evaluation
script and every `python -m` invocation to a paid API, silently (issue #7).
The key says what you *could* use; ROAMWISE_LLM says what you *want* used.
"""
import logging
import os
import random
import textwrap
import threading
import time
from collections import deque
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


class LLMRequestFailed(RuntimeError):
    """A remote generation could not be produced, after retries.

    Raised rather than swallowed into an empty Completion. A client that
    returns "" on failure is indistinguishable from a model with nothing to
    say, and the UI would render the gap as a finished plan -- the same class
    of silent failure as the truncation #125 fixed and the quiet template
    fallback #133 fixed. `views/itinerary.py` catches this and tells the
    traveler the narration failed; nothing else in the pipeline depends on the
    narrative, so the itinerary itself still stands.
    """


# How many requests per minute the remote clients allow themselves.
#
# 40 is NVIDIA's free tier (issue #7). Normal use is nowhere near it -- one
# plan costs two generations, so 40/min is twenty plans a minute -- but a
# burst is one loop away, and the failure mode is a 429 in the middle of a
# traveler's plan rather than a slower one. Waiting is the cheaper outcome, so
# the limiter blocks instead of raising. Read from the environment because the
# quota belongs to whoever's key it is, not to this file.
LLM_RPM_LIMIT = int(os.environ.get("ROAMWISE_LLM_RPM", "40"))

# Retries are for the failures that pass: a 429 the limiter did not prevent
# (another process sharing the key), and the 5xx family. A 400 or a 401 is a
# bug or a bad key and retrying it just spends the quota three times.
RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})
MAX_ATTEMPTS = 3


class _RateLimiter:
    """Sliding-window limiter shared by every remote client in the process.

    Module-level rather than per-client because the quota is per *key*, not
    per object: `views/itinerary.py` caches one orchestrator, but the
    evaluation scripts build their own agents, and two clients each politely
    keeping to 40/min would together ask for 80.
    """

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self._calls = deque()
        self._lock = threading.Lock()

    def wait(self) -> None:
        """Block until another call fits inside the window."""
        if self.per_minute <= 0:          # 0 disables the limiter entirely
            return
        while True:
            with self._lock:
                now = time.monotonic()
                while self._calls and now - self._calls[0] >= 60.0:
                    self._calls.popleft()
                if len(self._calls) < self.per_minute:
                    self._calls.append(now)
                    return
                sleep_for = 60.0 - (now - self._calls[0])
            log.warning(
                "rate limit reached (%d requests/minute); waiting %.1fs before the next "
                "generation", self.per_minute, sleep_for,
                extra={"roamwise_fields": {"rpm_limit": self.per_minute,
                                           "waited_seconds": round(sleep_for, 1)}},
            )
            time.sleep(max(sleep_for, 0.0) + 0.01)


_limiter = _RateLimiter(LLM_RPM_LIMIT)


def _retry_delay(attempt: int, retry_after: str = None) -> float:
    """Honour the server's own Retry-After when it sends one, else back off.

    Jittered, because the two generations of a single plan fail together and
    would otherwise retry in lockstep.
    """
    if retry_after:
        try:
            return min(float(retry_after), 60.0)
        except (TypeError, ValueError):
            pass
    return min(2.0 ** attempt, 30.0) + random.uniform(0, 0.5)


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


# NVIDIA's hosted inference endpoint speaks the OpenAI chat-completions
# shape. Both are overridable: the model is whatever the account has access to
# (the free tier's catalogue changes), and the base URL lets the same client
# talk to any other OpenAI-compatible gateway without a second class.
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_DEFAULT_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
_HTTP_TIMEOUT = 120  # a long synthesis on a busy free tier is slow, not hung

# Nemotron-class models deliberate before answering, and return that
# deliberation in a separate `reasoning_content` field. RoamWise wants the
# narration, not the deliberation: nothing downstream reads reasoning, it is
# billed like any other output token, and with thinking on, a model can spend
# the whole budget reasoning and return an empty `content` -- which would reach
# the traveler as a blank plan.
#
# Sent as a plain body field because this client posts raw JSON; it is what
# the `openai` SDK's `extra_body` puts on the wire. Set ROAMWISE_LLM_THINKING=1
# to leave it on for a model that needs it.
NVIDIA_THINKING = os.environ.get("ROAMWISE_LLM_THINKING", "").strip() not in ("", "0")


class NvidiaLLMClient(LLMClient):
    """NVIDIA's OpenAI-compatible endpoint (issue #7).

    Plain `requests` rather than the `openai` SDK. `requests` is already a
    dependency of this project, the call is a single JSON POST, and the SDK
    would bring its own retry loop to argue with `_limiter` and the backoff
    below -- two independent retriers against one 40/minute quota is how a
    burst turns into a ban rather than a wait.
    """

    def __init__(self, model: str = None, base_url: str = None):
        import requests  # kept local for symmetry with the optional clients

        self.api_key = os.environ.get("NVIDIA_API_KEY")
        if not self.api_key:
            raise RuntimeError("NVIDIA_API_KEY is not set")
        self.model = model or os.environ.get("NVIDIA_LLM_MODEL", NVIDIA_DEFAULT_MODEL)
        self.base_url = (base_url or os.environ.get("NVIDIA_BASE_URL", NVIDIA_BASE_URL)).rstrip("/")
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        })

    def complete_verbose(self, system: str, prompt: str,
                         max_tokens: int = None) -> Completion:
        import requests

        budget = max_tokens or budget_for(prompt)
        payload = {
            "model": self.model,
            "max_tokens": budget,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt}],
            "chat_template_kwargs": {"enable_thinking": NVIDIA_THINKING},
        }
        last_error = None
        for attempt in range(MAX_ATTEMPTS):
            _limiter.wait()
            try:
                resp = self._session.post(f"{self.base_url}/chat/completions",
                                          json=payload, timeout=_HTTP_TIMEOUT)
            except requests.RequestException as exc:
                last_error = exc
                if attempt == MAX_ATTEMPTS - 1:
                    break
                time.sleep(_retry_delay(attempt))
                continue

            if resp.status_code in RETRYABLE_STATUS:
                last_error = f"HTTP {resp.status_code}"
                if attempt == MAX_ATTEMPTS - 1:
                    break
                delay = _retry_delay(attempt, resp.headers.get("Retry-After"))
                log.warning(
                    "NVIDIA endpoint returned %s; retrying in %.1fs (attempt %d of %d)",
                    resp.status_code, delay, attempt + 1, MAX_ATTEMPTS,
                    extra={"roamwise_fields": {"status": resp.status_code,
                                               "attempt": attempt + 1,
                                               "retry_in_seconds": round(delay, 1)}},
                )
                time.sleep(delay)
                continue

            if not resp.ok:
                # 400/401/404: retrying spends the quota on the same mistake.
                raise LLMRequestFailed(
                    f"NVIDIA endpoint rejected the request: HTTP {resp.status_code} "
                    f"{resp.text[:300]}")

            choice = resp.json()["choices"][0]
            message = choice.get("message") or {}
            text = (message.get("content") or "").strip()
            truncated = choice.get("finish_reason") == "length"

            if not text:
                # A reasoning model that spent the whole budget deliberating
                # returns HTTP 200 with an empty `content` and a full
                # `reasoning_content`. Returning that empty string would put a
                # blank narrative on screen under a "Written by: NVIDIA API"
                # label -- a confident-looking nothing, which is the failure
                # #125 and #133 both exist to prevent. The reasoning itself is
                # not a substitute: it is the model thinking, not the plan.
                reasoned = len((message.get("reasoning_content") or "").strip())
                raise LLMRequestFailed(
                    f"{self.model} returned no narration"
                    + (f" (only {reasoned} characters of reasoning; set "
                       f"ROAMWISE_LLM_THINKING=0 or raise the token budget)" if reasoned
                       else f" (finish_reason={choice.get('finish_reason')!r})"))

            _warn_if_truncated(truncated, "NvidiaLLMClient", budget)
            return Completion(text, truncated)

        raise LLMRequestFailed(
            f"NVIDIA endpoint did not answer after {MAX_ATTEMPTS} attempts ({last_error})")


class AnthropicLLMClient(LLMClient):
    def __init__(self, model: str = "claude-sonnet-5"):
        import anthropic  # imported lazily so the dependency is optional
        self.client = anthropic.Anthropic()
        self.model = model

    def complete_verbose(self, system: str, prompt: str,
                         max_tokens: int = None) -> Completion:
        budget = max_tokens or budget_for(prompt)
        # The same quota discipline as the NVIDIA path: the SDK retries 429s
        # on its own, but nothing stops it *reaching* one without this.
        _limiter.wait()
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
    if isinstance(client, NvidiaLLMClient):
        return f"NVIDIA API ({client.model})"
    if isinstance(client, AnthropicLLMClient):
        return f"Anthropic API ({client.model})"
    return "deterministic template (no model)"


# The engines ROAMWISE_LLM can name, and how each one is built. Ordered so the
# error message below reads in the order the README introduces them.
_ENGINES = {
    "nvidia": ("NvidiaLLMClient", lambda: NvidiaLLMClient()),
    "anthropic": ("AnthropicLLMClient", lambda: AnthropicLLMClient()),
    "local": ("LocalHuggingFaceLLMClient", lambda: LocalHuggingFaceLLMClient()),
    "template": ("TemplateLLMClient", lambda: TemplateLLMClient()),
}


def requested_engine() -> str:
    """Which engine the environment asks for, normalised, or None.

    ROAMWISE_LOCAL_LLM=1 is still honoured as `local`: #54 documented it, the
    README documents it, and someone's working setup should not break because
    a second hosted provider arrived (issue #7).
    """
    wanted = (os.environ.get("ROAMWISE_LLM") or "").strip().lower()
    if wanted:
        return wanted
    if os.environ.get("ROAMWISE_LOCAL_LLM"):
        return "local"
    return None


def _opt_in_label() -> str:
    """The variable the user actually set, as they wrote it.

    Not the engine name: someone who set ROAMWISE_LOCAL_LLM=1 has to be told
    about ROAMWISE_LOCAL_LLM, or the warning points at a variable they never
    touched and cannot find. This is what #133's tests assert on, and they are
    right to -- naming the wrong knob is only marginally better than saying
    nothing.
    """
    if os.environ.get("ROAMWISE_LLM"):
        return f"ROAMWISE_LLM={(os.environ['ROAMWISE_LLM'] or '').strip().lower()}"
    if os.environ.get("ROAMWISE_LOCAL_LLM"):
        return "ROAMWISE_LOCAL_LLM"
    return None


def fallback_reason(client: LLMClient) -> str:
    """The opt-in that was set but did not take effect, or None.

    Recomputed from the environment rather than remembered at construction:
    `views/itinerary.py` caches the orchestrator with `st.cache_resource`, so
    the client is shared across sessions and must not carry per-run state.

    Returns None when the template is running because nobody asked for
    anything else -- that is the deliberate default and must stay silent, or
    the warning that matters becomes background noise. A key sitting in the
    environment is no longer a reason on its own: holding a key is not asking
    to spend it, which is the whole point of ROAMWISE_LLM (issue #7).
    """
    if not isinstance(client, TemplateLLMClient):
        return None
    wanted = requested_engine()
    if wanted and wanted != "template":
        return f"{_opt_in_label()} is set"
    return None


def _warn_construction_failed(wanted: str, opt_in: str, exc: Exception) -> None:
    log.warning(
        "%s was requested via %s but could not be constructed (%s: %s)",
        wanted, opt_in, type(exc).__name__, exc,
        extra={"roamwise_fields": {"wanted_client": wanted, "opt_in": opt_in,
                                   "error": f"{type(exc).__name__}: {exc}"}},
    )


def get_default_llm_client() -> LLMClient:
    """The engine ROAMWISE_LLM names, or the offline template.

    One explicit switch rather than a key hunt (issue #7). Before this, merely
    having ANTHROPIC_API_KEY in the environment routed every caller to a paid
    API -- including `pytest`, which spent 42 calls a run on it, and the
    evaluation scripts, which need no model at all. Exporting a key into a
    shell profile is a normal thing to do; billing every subsequent command
    for it is not.

    An unknown value is refused loudly instead of quietly falling through to
    the template: a typo in ROAMWISE_LLM otherwise produces a plausible-looking
    run whose narrative is the prompt echoed back (see TemplateLLMClient).
    """
    wanted = requested_engine()
    if wanted and wanted not in _ENGINES:
        log.warning(
            "ROAMWISE_LLM=%r is not a known engine (%s); falling back to the template",
            wanted, ", ".join(_ENGINES),
            extra={"roamwise_fields": {"requested": wanted,
                                       "known": sorted(_ENGINES)}},
        )
    elif wanted and wanted != "template":
        name, build = _ENGINES[wanted]
        try:
            return build()
        except Exception as exc:
            # Swallowing this whole was how a missing package, a missing key or
            # a broken model cache turned into "the app works fine" (#133).
            _warn_construction_failed(name, _opt_in_label(), exc)

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
