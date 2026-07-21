import os
import re
from dotenv import load_dotenv
from .utils import logger

load_dotenv()

_QDRANT_URI = os.getenv("QDRANT_URI")
_QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
COLLECTION_NAME = "recipes_v2"
EMBEDDING_MODEL = "gemini-embedding-001"

_qdrant_client = None
_collection_available: bool | None = None


def _get_client():
    global _qdrant_client
    if _qdrant_client is None:
        from qdrant_client import QdrantClient
        _qdrant_client = QdrantClient(url=_QDRANT_URI, api_key=_QDRANT_API_KEY, check_compatibility=False)
    return _qdrant_client


def _collection_exists() -> bool:
    global _collection_available
    if _collection_available is not True:
        # Only cache a successful check. A failure may be transient (cold-start,
        # network blip) so don't lock RAG off for the rest of the process — retry
        # on the next call instead.
        try:
            client = _get_client()
            client.get_collection(COLLECTION_NAME)
            _collection_available = True
            logger.info(f"Qdrant collection '{COLLECTION_NAME}' found.")
        except Exception as e:
            _collection_available = False
            logger.warning(f"Qdrant collection '{COLLECTION_NAME}' check failed ({e}). RAG disabled for this call.")
    return _collection_available


def _embed(text: str) -> list[float]:
    from google import genai
    client = genai.Client(api_key=_GEMINI_API_KEY)
    result = client.models.embed_content(model=EMBEDDING_MODEL, contents=text)
    return result.embeddings[0].values


# Matches each per-serving block in a Qdrant recipe payload, e.g.:
#   Serving 4
#   - Preparation Time: 10.0 mins
#   - Cooking Time: 48.0 mins
#   - Total Time: 58.0 mins
_SERVING_BLOCK_RE = re.compile(
    r"Serving\s+(\d+)\s*\n(?:-[^\n]*\n){0,4}?-\s*Cooking Time:\s*([\d.]+)\s*mins",
    re.IGNORECASE,
)


def _extract_cook_time_by_serving(payload_text: str) -> dict:
    """Deterministically parse {serving_count: cooking_time_minutes} from one payload's raw text."""
    return {
        int(m.group(1)): float(m.group(2))
        for m in _SERVING_BLOCK_RE.finditer(payload_text)
    }


def get_rag_reference(query: str, top_k: int = 3) -> dict:
    """
    Single retrieval call returning both:
      - "context": the existing concatenated top-k text block, for prompt grounding.
      - "cook_time_by_serving" / "dish_name": from the SINGLE closest match only —
        a {serving_count: cooking_time_minutes} map regex-extracted (deterministic,
        no LLM involved) straight from its payload text. Empty dict if no
        collection, no hits, or the closest match has no parseable per-serving
        cooking-time block. Serving count isn't known yet at retrieval time (it's
        resolved later in the pipeline), so callers pick the right entry via
        pick_cook_time_for_serving() once they know it.
    """
    result = {"context": "", "cook_time_by_serving": {}, "dish_name": None}
    if not _collection_exists():
        return result
    try:
        query_vector = _embed(query)
        results = _get_client().query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
        )
        if not results.points:
            return result

        parts = []
        for i, pt in enumerate(results.points):
            payload = pt.payload or {}
            text = payload.get("text", "")
            # extract dish name from the first line: "Dish Name: X"
            name = "Recipe"
            for line in text.splitlines():
                if line.startswith("Dish Name:"):
                    name = line.split(":", 1)[1].strip()
                    break
            parts.append(f"## {name}\n{text}")

            if i == 0:
                result["cook_time_by_serving"] = _extract_cook_time_by_serving(text)
                result["dish_name"] = name

        result["context"] = "\n\n---\n\n".join(parts)
        return result
    except Exception as e:
        logger.warning(f"RAG retrieval failed: {e}")
        return result


def pick_cook_time_for_serving(cook_time_by_serving: dict, serving: int | None) -> tuple:
    """Returns (cook_time_min, matched_serving) for the entry closest to `serving`, or (None, None)."""
    if not cook_time_by_serving:
        return None, None
    matched_serving = min(
        cook_time_by_serving,
        key=lambda s: abs(s - serving) if serving else s,
    )
    return cook_time_by_serving[matched_serving], matched_serving
