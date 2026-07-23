import os
import time
import random
import logging
from dotenv import load_dotenv
from pydantic import BaseModel
from google import genai
from google.genai import types
from google.genai.errors import ServerError, APIError

load_dotenv()

logger = logging.getLogger(__name__)

_gemini_api_key = os.getenv("GEMINI_API_KEY")
assert _gemini_api_key is not None, "Load GEMINI_API_KEY in .env"

gemini_client = genai.Client(api_key=_gemini_api_key)

# Transient Gemini failures (502/503/504 overloads, 429 rate limits) self-heal on
# retry; back off exponentially with jitter instead of failing on the first blip.
_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
_MAX_API_RETRIES = 5
_RETRY_BASE_DELAY = 2.0  # seconds; doubles each attempt


def _generate_with_retry(**kwargs):
    """gemini_client.models.generate_content with exponential-backoff retries on
    transient server errors (502 Bad Gateway, 503, 504, 429). Re-raises on
    non-retryable errors or once retries are exhausted."""
    last_err = None
    for attempt in range(_MAX_API_RETRIES):
        try:
            return gemini_client.models.generate_content(**kwargs)
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

class GeminiLLM:
    def __init__(self, model_id: str):
        self.model_id = model_id

    def run(
        self,
        input_txt: str,
        system_instruction: str,
        response_model: BaseModel,
        thinking_mode: bool = True
    ):
        output = _generate_with_retry(
            model=self.model_id,
            contents=input_txt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=response_model,
                thinking_config=(
                    types.ThinkingConfig(include_thoughts=True)
                    if thinking_mode
                    else types.ThinkingConfig(thinking_budget=0)
                ),
            ),
        )
        content_parts = output.candidates[0].content.parts
        response = content_parts[-1].text
        thought = [
            part.text for part in content_parts if getattr(part, "thought", False)
        ]
        return {"response": response, "thought": thought}