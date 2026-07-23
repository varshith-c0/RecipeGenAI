"""Startup environment-variable validation.

Import this before any route is registered. Missing required vars fail fast
with a clear message here, instead of `None` silently propagating into
`genai.Client()`/`QdrantClient()` construction and surfacing later as a
confusing, hard-to-trace downstream error on the first real request.
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

# Hard-required: the app cannot serve /process_recipe without these.
_REQUIRED_VARS = ("GEMINI_API_KEY", "QDRANT_URI", "QDRANT_API_KEY")

# Optional: only needed for specific features, which degrade gracefully
# without them (YouTube path returns a clean error; tracing already
# no-ops on setup failure per core/tracing.py).
_OPTIONAL_VARS = ("YOUTUBE_DATA_API_KEY", "OPEN_ROUTER_API_KEY", "PHOENIX_API_KEY")


def validate_environment() -> None:
    missing = [v for v in _REQUIRED_VARS if not os.getenv(v)]
    if missing:
        sys.exit(
            f"Missing required environment variable(s): {', '.join(missing)}. "
            f"Set them in .env before starting the app (see .env.example)."
        )
    for v in _OPTIONAL_VARS:
        if not os.getenv(v):
            print(f"[config] Optional env var {v} not set — its feature will be unavailable.", file=sys.stderr)
