import os
import json
import time
import random
import isodate
from dotenv import load_dotenv
from urllib.parse import urlparse, parse_qs
from googleapiclient.discovery import build
from google import genai
from google.genai import types
from google.genai.errors import ServerError, ClientError, APIError
from dotenv import load_dotenv
from .utils import logger, func_timing_decorator

load_dotenv()
_gemini_api_key = os.getenv("GEMINI_API_KEY")
assert _gemini_api_key is not None, "Load GEMINI_API_KEY in .env"
_yt_api_key = os.getenv("YOUTUBE_DATA_API_KEY")
assert _yt_api_key is not None, "Load YOUTUBE_DATA_API_KEY in .env"

client = genai.Client(api_key=_gemini_api_key)
youtube = build('youtube', 'v3', developerKey=_yt_api_key)

# Transient Gemini failures (502/503/504 overloads, 429 rate limits) self-heal on
# retry. The video-understanding call here is heavy and just as prone to them, so
# back off exponentially with jitter instead of failing on the first blip.
_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
_MAX_API_RETRIES = 5
_RETRY_BASE_DELAY = 2.0  # seconds; doubles each attempt


def _generate_with_retry(**kwargs):
    """client.models.generate_content with exponential-backoff retries on transient
    server errors (502 Bad Gateway, 503, 504, 429). Re-raises on non-retryable
    errors or once retries are exhausted."""
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

def _extract_yt_videoID(url: str):
    parsed = urlparse(url)
    hostname = parsed.hostname or ''
    path = parsed.path
    if 'youtu.be' in hostname:
        return path.lstrip('/')
    if 'youtube.com' in hostname:
        if path == '/watch':
            return parse_qs(parsed.query).get('v', [None])[0]
        for prefix in ('/embed/', '/v/', '/shorts/'):
            if path.startswith(prefix):
                return path[len(prefix):]
    return None

def _time_iso_to_sec(iso_duration):
    return isodate.parse_duration(iso_duration).total_seconds()

def _get_yt_video_metadata(videoID: str):
    request = youtube.videos().list(part="snippet,statistics,contentDetails", id=videoID)
    response = request.execute()
    if not response['items']:
        logger.debug("Metadata extraction failed: No items found in response.")
        return None
    item = response['items'][0]
    metadata = {
        'title': item['snippet']['title'].strip(),
        'description': item['snippet']['description'].strip() ,
        'duration_s': _time_iso_to_sec(item['contentDetails']['duration'])
    }
    logger.debug(f"YT video metadata:\n{metadata}")
    return metadata


def _yt_link_2_recipe_prompt(metadata: dict):
   title = metadata.get('title')
   description = metadata.get('description').strip()
   if not description or description == '':
      description = ''
   else:
       description = f"\nVideo Description:\n=====\n{description}\n=====\n"

   return f"""
Based on this YouTube recipe video, extract and provide:
1. Recipe Name: Determine the most likely name of the recipe
2. Ingredients: Create a complete list of all ingredients mentioned with quantities
3. Recipe Steps: Provide numbered steps for preparing the recipe

Video Title:
{title}
{description}
Format your response exactly as follows:
Recipe Name: [name]

Ingredients:
- [ingredient 1 with quantity]
- [ingredient 2 with quantity]
...

Instructions:
1. [step 1]
2. [step 2]
...

Important:
- If ingredient quantities aren’t specified, estimate them based on the visual cues provided.
"""

@func_timing_decorator
def extract_recipe_cnt_from_yt_url(url: str, metadata: dict):
    try:
        resp_obj = _generate_with_retry(
            model='models/gemini-3.1-flash-lite',
            contents=types.Content(
                parts=[
                  types.Part(text=_yt_link_2_recipe_prompt(metadata)),
                  types.Part(file_data=types.FileData(file_uri=url), video_metadata=types.VideoMetadata(fps=1.5)),
                ]
            ),
            config=types.GenerateContentConfig(
               seed=1,
               mediaResolution="MEDIA_RESOLUTION_LOW" # https://ai.google.dev/api/generate-content#MediaResolution
            )
        )
    except ClientError as e:
        # The model fetches the YouTube video through its own access path (separate
        # from the Data API that read the metadata). A 403/404 here means Gemini
        # cannot ingest THIS specific video — it's private/unlisted, age- or
        # region-restricted, has embedding disabled, or was removed. This is bad
        # user input, not a server fault, so return a clean message instead of 500.
        code = getattr(e, "code", None)
        if code in (403, 404):
            logger.warning(f"Video not accessible to model (code={code}) for url='{url}': {e}")
            return (
                False,
                None,
                "This video can't be accessed. Make sure it's public (not private "
                "or unlisted) and not age- or region-restricted, then try again.",
            )
        logger.error(f"Recipe extraction failed with client error (code={code}) for url='{url}': {e}")
        return False, None, "System error."
    # prompt_tokens = response.usage_metadata.prompt_token_count
    # output_tokens = response.usage_metadata.candidates_token_count
    # total_tokens = response.usage_metadata.total_token_count
    # logger.info(f"Num prompt tokens: {prompt_tokens} | Num output tokens: {output_tokens} | Num total tokens: {total_tokens}")
    if not resp_obj:
       logger.error(f"Recipe extraction failed: no response from agent. resp_obj='{resp_obj}'")
       return False, None, "System error."
    raw_txt = resp_obj.text
    logger.debug(f"Extracted recipe content from url: {raw_txt}")
    return True, raw_txt, None


if __name__ == '__main__':
    youtube_url = 'https://www.youtube.com/shorts/GhpaBtvRPpw'
    result = _get_yt_video_metadata(youtube_url)
