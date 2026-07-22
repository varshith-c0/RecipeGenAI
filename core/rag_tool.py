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

# Same idea for the calibrated water volume, which sits two bullets further down the
# block. Some dishes record "Water Quantity: N/A ml" — [\d.]+ simply won't match those,
# leaving them absent from the map rather than parsed as zero.
_WATER_BLOCK_RE = re.compile(
    r"Serving\s+(\d+)\s*\n(?:-[^\n]*\n){0,6}?-\s*Water Quantity:\s*([\d.]+)\s*ml",
    re.IGNORECASE,
)


def _extract_cook_time_by_serving(payload_text: str) -> dict:
    """Deterministically parse {serving_count: cooking_time_minutes} from one payload's raw text."""
    return {
        int(m.group(1)): float(m.group(2))
        for m in _SERVING_BLOCK_RE.finditer(payload_text)
    }


def _extract_water_by_serving(payload_text: str) -> dict:
    """Deterministically parse {serving_count: water_ml} from one payload's raw text."""
    return {
        int(m.group(1)): float(m.group(2))
        for m in _WATER_BLOCK_RE.finditer(payload_text)
    }


# --- timing-reference selection ------------------------------------------------
# The vector search scores whole documents, so a reference dish matching on every
# ingredient EXCEPT the slow-cooking one can outrank the exact match. Observed:
# a cabbage-and-potato recipe retrieved "Spicy Cabbage Masala" (cabbage/peas/tomato/
# onion — four of five ingredients identical) at rank 1 and the real "Cabbage Aloo
# Sabzi" at rank 4. The potato is one line in a 700-character document and barely
# moves the embedding, but it is the densest thing in the pan and dominates cook
# time: the 27-minute target came from a potato-free dish and the potato came out
# raw. Cook time is set by what is physically in the pan, so choose the timing
# reference by ingredient overlap and let the vector rank break ties.
_INGREDIENT_LINE_RE = re.compile(r"^-\s*([A-Za-z][A-Za-z \-&']*?)\s*\(", re.MULTILINE)

# A reference with one or two ingredients scores 1.0 against almost anything that
# mentions them, so require a few before its coverage is allowed to mean something.
_MIN_REF_INGREDIENTS = 2
# Coverage must beat rank 1 by this much to overturn it. The vector score is still
# real evidence; only a clear ingredient-level win should displace it.
_COVERAGE_OVERRIDE_MARGIN = 0.15


def _payload_ingredients(payload_text: str) -> set:
    """Unique lowercased ingredient names from a payload's '- Name (qty Unit)' lines.

    Requiring the '(' means only the structured ingredient lines match — the free
    prose inside parenthetical prep notes (and nested sub-recipes like Bengali baja
    masala, which list 'Cumin seeds - 1 tsp') is skipped.
    """
    return {m.group(1).strip().lower() for m in _INGREDIENT_LINE_RE.finditer(payload_text)}


def _mentions(ingredient: str, text: str) -> bool:
    """True if `ingredient` appears in `text`, tolerating plurals (potato/potatoes)."""
    return re.search(rf"\b{re.escape(ingredient)}(e?s)?\b", text, re.IGNORECASE) is not None


def _pick_timing_reference(candidates: list, recipe_text: str) -> int:
    """Index of the candidate whose ingredients best match the recipe; 0 (vector rank 1)
    unless another candidate wins on coverage by a clear margin."""
    coverages = []
    for _name, text in candidates:
        ings = _payload_ingredients(text)
        if len(ings) < _MIN_REF_INGREDIENTS:
            coverages.append(0.0)  # too thin to judge — can never override
            continue
        matched = sum(1 for i in ings if _mentions(i, recipe_text))
        coverages.append(matched / len(ings))

    # Ties fall back to vector order: -i favours the earlier (closer) candidate.
    best = max(range(len(candidates)), key=lambda i: (coverages[i], -i))
    if best != 0 and coverages[best] - coverages[0] >= _COVERAGE_OVERRIDE_MARGIN:
        logger.info(
            f"RAG timing reference: '{candidates[best][0]}' (rank {best + 1}, "
            f"ingredient coverage {coverages[best]:.2f}) overrides rank 1 "
            f"'{candidates[0][0]}' (coverage {coverages[0]:.2f})."
        )
        return best
    return 0


# Candidates pulled for the ingredient-aware timing pick. Kept larger than the number
# actually pasted into the prompt so the right dish can be *considered* without
# growing the prompt — the exact match sat at rank 4 under the old top_k of 3.
_TIMING_CANDIDATES = 5
_CONTEXT_DOCS = 3


def get_rag_reference(query: str, top_k: int = _TIMING_CANDIDATES) -> dict:
    """
    Single retrieval call returning both:
      - "context": the top _CONTEXT_DOCS payloads concatenated for prompt grounding,
        led by the chosen timing reference.
      - "cook_time_by_serving" / "water_by_serving" / "dish_name": from the SINGLE
        best timing reference — the candidate chosen by _pick_timing_reference(),
        which is vector rank 1 unless another retrieved dish covers this recipe's
        ingredients clearly better. {serving_count: value} maps regex-extracted
        (deterministic, no LLM involved) straight from its payload text. Empty dict
        if no collection, no hits, or that match has no parseable per-serving block.
        Serving count isn't known yet at retrieval time (it's resolved later in the
        pipeline), so callers pick the right entry via pick_value_for_serving() once
        they know it.
    """
    result = {"context": "", "cook_time_by_serving": {}, "water_by_serving": {}, "dish_name": None}
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

        candidates = []
        for pt in results.points:
            text = (pt.payload or {}).get("text", "")
            # extract dish name from the first line: "Dish Name: X"
            name = "Recipe"
            for line in text.splitlines():
                if line.startswith("Dish Name:"):
                    name = line.split(":", 1)[1].strip()
                    break
            candidates.append((name, text))

        best = _pick_timing_reference(candidates, query)
        result["cook_time_by_serving"] = _extract_cook_time_by_serving(candidates[best][1])
        result["water_by_serving"] = _extract_water_by_serving(candidates[best][1])
        result["dish_name"] = candidates[best][0]

        # The chosen reference leads the context as well. Grounding the prompt on one
        # dish while timing the cook against another would put the two in conflict,
        # and the model reads the first block as the primary reference.
        ordered = [candidates[best]] + [c for i, c in enumerate(candidates) if i != best]
        result["context"] = "\n\n---\n\n".join(
            f"## {name}\n{text}" for name, text in ordered[:_CONTEXT_DOCS]
        )
        return result
    except Exception as e:
        logger.warning(f"RAG retrieval failed: {e}")
        return result


def pick_value_for_serving(by_serving: dict, serving: int | None) -> tuple:
    """Returns (value, matched_serving) for the entry closest to `serving`, or (None, None).

    Works for any {serving_count: value} map parsed off a payload — cooking time,
    water volume — since the selection only depends on the serving key.
    """
    if not by_serving:
        return None, None
    matched_serving = min(
        by_serving,
        key=lambda s: abs(s - serving) if serving else s,
    )
    return by_serving[matched_serving], matched_serving


def pick_cook_time_for_serving(cook_time_by_serving: dict, serving: int | None) -> tuple:
    """Returns (cook_time_min, matched_serving) for the entry closest to `serving`."""
    return pick_value_for_serving(cook_time_by_serving, serving)
