"""Shared, process-wide `genai.Client` and Gemini call helpers.

`genai.Client(...)` does credential/transport setup on construction; every
call site that built a fresh one per request paid that cost repeatedly for no
benefit (the client is stateless w.r.t. any single request). One client per
process, lazily built on first use, mirrors the pattern already used
correctly for the Qdrant client in `core/rag_tool.py`.
"""

import os
import random
import threading
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ServerError, APIError

from .utils import logger

load_dotenv()

_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Per-request-call timeout. Applies to every generate_content/embed_content
# call made through get_gemini_client() below (HttpOptions.timeout is in ms).
_REQUEST_TIMEOUT_MS = 30_000

_client: genai.Client | None = None
_client_lock = threading.Lock()


def get_gemini_client() -> genai.Client:
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = genai.Client(
                    api_key=_GEMINI_API_KEY,
                    http_options=types.HttpOptions(timeout=_REQUEST_TIMEOUT_MS),
                )
    return _client


# Transient Gemini failures (502/503/504 overloads, 429 rate limits, 408
# timeouts) are server-side and self-heal on retry. Backoff is capped rather
# than the previous 5-attempt/32s-tier schedule: a request budget of
# gunicorn's --timeout has to cover several of these calls per pipeline run
# (tool loop + extraction + repair rounds), so one call eating a minute-plus
# on retries alone starves the others.
_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
_MAX_API_RETRIES = 3
_RETRY_BASE_DELAY = 2.0  # seconds; doubles each attempt


def generate_with_retry(client: genai.Client, **kwargs):
    """Wrapper around client.models.generate_content that retries on transient
    Gemini errors (5xx overloads like the 502 Bad Gateway, plus 429 rate limits).
    Non-retryable errors (e.g. 400/401/404) and exhausted retries re-raise so the
    caller's error handler can still fail gracefully."""
    last_err = None
    for attempt in range(_MAX_API_RETRIES):
        try:
            return client.models.generate_content(**kwargs)
        except APIError as e:
            code = getattr(e, "code", None)
            if not (isinstance(e, ServerError) or code in _RETRYABLE_STATUS):
                raise
            last_err = e
            if attempt == _MAX_API_RETRIES - 1:
                break
            delay = _RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1)
            logger.warning(
                f"Gemini transient error (code={code}, attempt "
                f"{attempt + 1}/{_MAX_API_RETRIES}); retrying in {delay:.1f}s"
            )
            time.sleep(delay)
    raise last_err
