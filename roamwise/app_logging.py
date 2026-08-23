"""In-process log capture that backs the "System logs" screen (issue #41).

The UI used to carry an "Agent trace" tab that dumped a hand-assembled
``st.json`` blob of orchestrator internals next to the itinerary. That put
implementation detail in the middle of the traveler's flow, and it only ever
showed the five fields someone remembered to add to the dict.

Logs are collected the ordinary way instead -- through the stdlib ``logging``
tree, rooted at the ``roamwise`` logger -- so anything any module logs reaches
the screen without the UI knowing that module exists. This file adds only the
sink: a bounded in-memory ring buffer that the Streamlit page reads.

The buffer is process-global rather than per-session, matching how Streamlit
itself is deployed here (one server process) and how the heavy objects are
held (``@st.cache_resource`` is process-wide too, so work done once on behalf
of one session is exactly the work whose logs everyone needs to see).
"""
import json
import logging
import threading
import time
from collections import deque
from contextlib import contextmanager

# Every logger in the codebase is created as `get_logger(__name__)`, and every
# module already lives under the `roamwise.` package, so attaching one handler
# here captures the whole codebase.
LOGGER_NAME = "roamwise"

# Enough to hold several full trip plans' worth of steps. The buffer is
# bounded because this process is long-lived: a demo left open for a day
# should not grow a log list without limit.
MAX_RECORDS = 2000

# LogRecord attribute used to smuggle structured fields past logging's
# printf-style message API, so a step can carry `n_pois=42` as data rather
# than baking it into a string the page would have to parse back out.
_FIELDS_KEY = "roamwise_fields"

LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

# Only ever used for its formatException(); rendering tracebacks is a
# Formatter responsibility, and Handler has no such method of its own.
_EXC_FORMATTER = logging.Formatter()


class _RingBufferHandler(logging.Handler):
    """Keeps the last MAX_RECORDS log records as plain dicts.

    Dicts, not LogRecords: records hold references to their arguments, which
    here would mean the buffer pinning whole itineraries and DataFrames in
    memory long after the request that built them finished.
    """

    def __init__(self, capacity: int = MAX_RECORDS):
        super().__init__()
        self._records: deque = deque(maxlen=capacity)
        self._seq = 0

    def emit(self, record: logging.LogRecord) -> None:
        # logging calls emit() with the handler lock held, so the sequence
        # counter and the deque need no extra synchronisation of their own.
        try:
            message = record.getMessage()
        except Exception:  # a bad format string must not take down the caller
            message = str(record.msg)

        traceback = None
        if record.exc_info:
            traceback = _EXC_FORMATTER.formatException(record.exc_info)

        self._seq += 1
        self._records.append({
            "seq": self._seq,
            "created": record.created,
            "level": record.levelname,
            "levelno": record.levelno,
            "logger": record.name,
            "message": message,
            "fields": dict(getattr(record, _FIELDS_KEY, None) or {}),
            "traceback": traceback,
        })

    def snapshot(self) -> list[dict]:
        """A copy of the buffer, oldest first, safe to iterate while logging
        continues on other threads (Streamlit runs each session in its own)."""
        self.acquire()
        try:
            return list(self._records)
        finally:
            self.release()

    def clear(self) -> None:
        self.acquire()
        try:
            self._records.clear()
        finally:
            self.release()


_handler = _RingBufferHandler()
_install_lock = threading.Lock()
_installed = False


def install() -> None:
    """Attach the buffer to the `roamwise` logger. Safe to call repeatedly.

    Streamlit re-runs page scripts top to bottom on every interaction while
    keeping imported modules alive, so this is called many times per session
    and must add the handler exactly once -- otherwise every record would be
    buffered N times over.
    """
    global _installed
    if _installed:
        return
    with _install_lock:
        if _installed:  # another session's thread won the race
            return
        logger = logging.getLogger(LOGGER_NAME)
        logger.setLevel(logging.INFO)
        logger.addHandler(_handler)
        # propagate stays on: the console/stderr stream is still useful when
        # running the pipeline headless (`python -m roamwise.agents.orchestrator`),
        # where there is no page to read the buffer.
        _installed = True


def get_logger(name: str = LOGGER_NAME) -> logging.Logger:
    """Logger for `name`, with the buffer guaranteed to be attached.

    Call as `get_logger(__name__)`; module names already sit under `roamwise.`
    so they inherit the handler installed on the package logger.
    """
    install()
    return logging.getLogger(name)


def records() -> list[dict]:
    install()
    return _handler.snapshot()


def clear() -> None:
    install()
    _handler.clear()


@contextmanager
def log_step(logger: logging.Logger, step: str, **fields):
    """Time a pipeline stage and log one record describing how it went.

    Yields the mutable field dict, so a stage can attach what it *learned*
    rather than only what was known before it ran::

        with log_step(log, "Fusion RAG retrieval", config=cfg) as detail:
            rag = agent.run(...)
            detail["n_results"] = len(rag["results"])

    One record is emitted per stage, on the way out, because the durations are
    the point: a started/finished pair would double the volume to say the same
    thing. A stage that raises logs at ERROR with its traceback and the
    exception propagates untouched.
    """
    detail = dict(fields)
    started = time.perf_counter()
    try:
        yield detail
    except Exception:
        detail["duration_ms"] = round((time.perf_counter() - started) * 1000, 1)
        logger.exception("%s failed", step, extra={_FIELDS_KEY: detail})
        raise
    else:
        detail["duration_ms"] = round((time.perf_counter() - started) * 1000, 1)
        logger.info("%s", step, extra={_FIELDS_KEY: detail})


def format_fields(fields: dict) -> str:
    """Structured fields -> one compact `k=v` line for the log table."""
    parts = []
    for key, value in fields.items():
        if isinstance(value, float):
            value = round(value, 3)
        if isinstance(value, (dict, list, tuple)):
            value = json.dumps(value, default=str, separators=(",", ":"))
        parts.append(f"{key}={value}")
    return "  ".join(parts)
