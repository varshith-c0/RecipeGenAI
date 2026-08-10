"""
recipe_file_index.py — Deterministic RAG over local .recipe files.

Parses the 700+ .recipe files in the recipes/ directory into an in-memory index
keyed by ingredient name. Used as a second retrieval source alongside the Qdrant
vector DB, providing exact verified Nosh command patterns (tray assignments, spice
sequences, cook/wait timings) that the free-text Qdrant payloads cannot expose precisely.

Public API
----------
find_by_ingredients(ing_list)      → list[RecipeMatch]  (overlap/Jaccard-ranked, top N)
find_by_id(recipe_id, servings)    → list[RecipeMatch]  (lookup exact Qdrant match)
format_as_rag_context(matches)     → str                (ready for prompt injection)
get_reference_signals(matches)     → dict               (spice counts + cook time for validators)
"""

from __future__ import annotations

import os
import re
import glob
import time
from dataclasses import dataclass, field
from typing import Optional

from .utils import logger

# ── Constants ──────────────────────────────────────────────────────────────────

# Root of the recipes folder, relative to THIS file's location.
# core/recipe_file_index.py → ../recipes/
_RECIPES_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "recipes")
)

# How many top matches to return to the caller.
_TOP_K = 3

# Minimum ingredient overlap fraction (matched / recipe_total) to include a result.
# Keeps unrelated matches (e.g. 1/12 ingredients) out of the context window.
_MIN_OVERLAP = 0.15

# Maximum number of commands to include in the RAG context snippet per recipe.
# Keeps the context block tight; the full command list can be very long.
_MAX_COMMANDS_IN_CONTEXT = 30


# ── Cook-time + spice-count parsing (same arithmetic as orchestrator) ──────────
# These regexes mirror _WAIT_RE / _COOK_RE / etc. in orchestrator.py exactly so
# the cook-time computed here is directly comparable to what the validator checks.

_RF_WAIT_RE    = re.compile(r"^wait\s+(\d+)\s+seconds",              re.IGNORECASE)
_RF_COOK_RE    = re.compile(r"^cook\s+(\d+)\s+seconds",              re.IGNORECASE)
_RF_COOK_AI_RE = re.compile(r"^cook\s+(.+?)\s+till\b",              re.IGNORECASE)
_RF_TIMEOUT_RE = re.compile(r"^set_temperature_timeout\s+(\d+)\s+seconds", re.IGNORECASE)
_RF_STIR_RE    = re.compile(r"^stir\s+(\S+)\s+(\d+)\s+times",       re.IGNORECASE)
_RF_SPICE_RE   = re.compile(r"^spice\s+(.+?)\s+dispense\s+(\d+)\s+times\s*$", re.IGNORECASE)

# Stir-type durations in seconds — identical to orchestrator's _STIR_SECONDS.
_RF_STIR_SECONDS: dict[str, float] = {"saute": 16, "mix": 30, "mix_liquid": 45}
_RF_DEFAULT_STIR_SECONDS = 30

# AI-assisted step duration credits (same as orchestrator).
_RF_AI_BROWNING_INGS = frozenset({"onion", "tomato"})
_RF_AI_BROWNING_SECONDS = 240.0
_RF_AI_COOKED_SECONDS   = 600.0

# Spice control pseudo-commands to skip (not real dispenses).
_RF_SPICE_SKIP = frozenset({"position", "rest"})


def _extract_spice_counts(commands: list[str]) -> dict[str, int]:
    """{spice_canonical_name: total_dispense_count} summed from all spice commands.

    Skips the position/rest control pseudo-commands emitted around each spice block.
    Names are stored as-is (e.g. 'corianderPowder') so the caller can normalise
    them with the orchestrator's alias table.
    """
    counts: dict[str, int] = {}
    for cmd in commands:
        m = _RF_SPICE_RE.match(cmd.strip())
        if not m:
            continue
        name = m.group(1).strip().lower()
        if name in _RF_SPICE_SKIP:
            continue
        counts[name] = counts.get(name, 0) + int(m.group(2))
    return counts


def _compute_recipe_cook_time(commands: list[str]) -> float:
    """Total active cook time in seconds, using the same arithmetic as
    orchestrator._compute_total_active_minutes (written + AI-credit).
    """
    total_s = 0.0
    for cmd in commands:
        c = cmd.strip()
        m = _RF_COOK_AI_RE.match(c)
        if m:
            ing = m.group(1).strip().lower()
            total_s += (_RF_AI_BROWNING_SECONDS if ing in _RF_AI_BROWNING_INGS
                        else _RF_AI_COOKED_SECONDS)
            continue
        m = (_RF_WAIT_RE.match(c) or _RF_COOK_RE.match(c) or _RF_TIMEOUT_RE.match(c))
        if m:
            total_s += float(m.group(1))
            continue
        m = _RF_STIR_RE.match(c)
        if m:
            stir_type = m.group(1).lower()
            count     = int(m.group(2))
            total_s  += count * _RF_STIR_SECONDS.get(stir_type, _RF_DEFAULT_STIR_SECONDS)
    return total_s


# ── DSL parsing helpers ────────────────────────────────────────────────────────

_META_RE = re.compile(
    r"^#(?:recipe\s+name|version|creator|course|cuisine|dish_type|servings)"
    r"\s*[-:]\s*(.+)$",
    re.IGNORECASE,
)

def _parse_meta_value(line: str, key: str) -> Optional[str]:
    """Return the value after 'key - ...' or 'key: ...' on a single header line."""
    pattern = re.compile(
        rf"^#{re.escape(key)}\s*[-:]\s*(.+)$", re.IGNORECASE
    )
    m = pattern.match(line.strip())
    return m.group(1).strip() if m else None


def _extract_ingredient_names(ing_block_lines: list[str]) -> list[str]:
    """
    Parse ingredient names from the #ingredients block.

    Each ingredient line looks like:
        #Name,Qty Unit,Prep,TrayNum[;Name2,...]
    or occasionally:
        #Name,QtyUnit,Prep,TrayNum    (no space between qty and unit)

    We only want the names — the first token before the first comma on each
    semicolon-separated segment. Lines starting with '#Additional-Instructions',
    '#end', '#servings', '#version', '#creator', '#course', '#cuisine',
    '#dish_type', '#recipe' are ignored.

    Also ignores the `#steps` / `#steps` header lines.
    """
    names = []
    _SKIP_RE = re.compile(
        r"^#(additional|end|servings|version|creator|course|cuisine|"
        r"dish_type|recipe|steps)\b",
        re.IGNORECASE,
    )
    for raw in ing_block_lines:
        line = raw.strip()
        if not line.startswith("#"):
            continue
        if _SKIP_RE.match(line):
            continue
        content = line.lstrip("#").strip()
        # Split on ';' then grab first comma-token for each segment
        for segment in content.split(";"):
            segment = segment.strip()
            if not segment:
                continue
            name = segment.split(",")[0].strip()
            # Drop pure-digit or empty tokens (e.g. lone tray numbers like "5;")
            if name and not name.isdigit():
                names.append(name.lower())
    return names


def _extract_vegmap(steps_lines: list[str]) -> dict[str, int]:
    """
    Parse the vegmap line: 'vegmap Onion=1,Tomato=2,...'
    Returns {canonical_name_lower: tray_number}.
    """
    mapping: dict[str, int] = {}
    for line in steps_lines:
        stripped = line.strip()
        if stripped.lower().startswith("vegmap "):
            payload = stripped[len("vegmap "):].strip()
            for pair in payload.split(","):
                pair = pair.strip()
                if "=" in pair:
                    name, num = pair.split("=", 1)
                    try:
                        mapping[name.strip().lower()] = int(num.strip())
                    except ValueError:
                        pass
    return mapping


def _extract_commands(steps_lines: list[str]) -> list[str]:
    """
    Return the raw command lines from the #steps block,
    excluding the vegmap line and blank lines.
    """
    cmds = []
    for line in steps_lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith("vegmap "):
            continue
        if stripped.lower().startswith("#steps"):
            continue
        cmds.append(stripped)
    return cmds


# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class RecipeEntry:
    """One parsed .recipe file, stored in the index."""
    recipe_id: str                   # e.g. "1-VT-INDN-MAIN"
    dish_name: str
    course: str
    cuisine: str
    dish_type: str
    servings: int
    ingredient_names: list[str]      # lowercased, all ingredients
    tray_map: dict[str, int]         # canonical_name_lower → tray number
    commands: list[str]              # all step commands (raw strings)
    filepath: str
    # ── Deterministic validator signals (extracted at parse time) ──────────────
    spice_counts: dict[str, int]     # {spice_name: total_dispenses} for this dish
    cook_time_seconds: float         # total active cook time (same arithmetic as orchestrator)


@dataclass
class RecipeMatch:
    """A retrieved recipe with its overlap score."""
    entry: RecipeEntry
    overlap_score: float             # fraction of recipe's ingredients that matched
    matched_ingredients: list[str]   # which query ingredients hit


# ── Index builder ──────────────────────────────────────────────────────────────

class RecipeFileIndex:
    """
    Lazy-loading, in-memory index over all .recipe files.

    Thread safety: fine for the single-process Flask/FastAPI use case.
    Build time on the 736-file corpus: ~1-2 s on first call, then cached forever.
    """

    def __init__(self, recipes_root: str = _RECIPES_ROOT):
        self._root = recipes_root
        self._entries: list[RecipeEntry] = []
        self._loaded = False

    # ── public API ─────────────────────────────────────────────────────────────

    def find_by_ingredients(
        self,
        query_ingredients: list[str],
        top_k: int = _TOP_K,
        min_overlap: float = _MIN_OVERLAP,
    ) -> list[RecipeMatch]:
        """
        Return up to `top_k` recipes ranked by Jaccard similarity and overlap.

        Uses set-based word overlap that ignores common descriptor/adjectives so
        that specific query names (e.g. 'green cabbage') match generic recipe
        ingredients (e.g. 'cabbage').
        """
        self._ensure_loaded()
        if not query_ingredients or not self._entries:
            return []

        query_set = {q.strip().lower() for q in query_ingredients if q.strip()}
        modifiers = {
            'green', 'red', 'fresh', 'dry', 'powder', 'seeds', 'leaves', 'oil',
            'sauce', 'paste', 'chopped', 'minced', 'sliced', 'cubes', 'stick',
            'ground', 'powdered', 'whole', 'split', 'yellow', 'white', 'strip',
            'strips', 'finely', 'thinly', 'peeled', 'dice', 'diced', 'cubed'
        }

        def _ing_match(query_name: str, recipe_name: str) -> bool:
            q_words = set(query_name.split())
            r_words = set(recipe_name.split())
            q_clean = q_words - modifiers
            r_clean = r_words - modifiers
            if not q_clean or not r_clean:
                return bool(q_words & r_words)
            return bool(q_clean & r_clean)

        scored: list[tuple[float, float, int, RecipeMatch]] = []

        for entry in self._entries:
            matched = []
            for q in query_set:
                if any(_ing_match(q, ing_name) for ing_name in entry.ingredient_names):
                    matched.append(q)
            if not matched:
                continue
            
            # 1. Standard overlap: fraction of the recipe's ingredients that matched
            overlap = len(matched) / max(len(entry.ingredient_names), 1)
            if overlap < min_overlap:
                continue

            # 2. Jaccard similarity: how representative the match is of both query & recipe
            union_size = len(entry.ingredient_names) + len(query_set) - len(matched)
            jaccard = len(matched) / max(union_size, 1)

            scored.append((
                jaccard,
                overlap,
                -len(entry.ingredient_names),
                RecipeMatch(
                    entry=entry,
                    overlap_score=overlap,
                    matched_ingredients=matched,
                )
            ))

        # Sort primarily by Jaccard similarity (best overall fit), then by standard overlap
        scored.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
        return [rm for _, _, _, rm in scored[:top_k]]

    def find_by_id(
        self,
        recipe_id: str,
        target_serving: int | None = None,
    ) -> list[RecipeMatch]:
        """Look up all entries matching the specified recipe ID (e.g. from Qdrant).
        
        Sorts the entries so the one closest to target_serving comes first.
        """
        self._ensure_loaded()
        if not recipe_id:
            return []
        
        clean_id = recipe_id.strip().lower()
        matches = []
        for entry in self._entries:
            if entry.recipe_id.lower() == clean_id:
                # 1.0 overlap because it's the exact same recipe category
                matches.append(RecipeMatch(
                    entry=entry,
                    overlap_score=1.0,
                    matched_ingredients=[]
                ))
        
        if not matches:
            return []
            
        # Tie-break servings: prefer exact target serving match, fallback to servings=4 or highest
        if target_serving:
            matches.sort(key=lambda m: abs(m.entry.servings - target_serving))
        else:
            matches.sort(key=lambda m: m.entry.servings, reverse=True)
        return matches

    # ── formatting ─────────────────────────────────────────────────────────────

    @staticmethod
    def format_as_rag_context(matches: list[RecipeMatch]) -> str:
        """
        Render retrieved .recipe matches as a labeled prompt section.

        Includes: dish metadata, ingredient→tray table, and the first
        _MAX_COMMANDS_IN_CONTEXT commands verbatim so the LLM can see
        exact Nosh syntax without needing to infer it.
        """
        if not matches:
            return ""

        blocks = []
        for rm in matches:
            e = rm.entry
            # Build ingredient→tray table (only entries present in vegmap)
            tray_rows = []
            for name, tray_num in sorted(e.tray_map.items(), key=lambda x: x[1]):
                tray_rows.append(f"  Tray {tray_num}: {name}")
            tray_table = "\n".join(tray_rows) if tray_rows else "  (no tray map)"

            # Truncate commands if very long
            cmds = e.commands[:_MAX_COMMANDS_IN_CONTEXT]
            cmd_note = (
                f"  [first {_MAX_COMMANDS_IN_CONTEXT} of {len(e.commands)} commands]"
                if len(e.commands) > _MAX_COMMANDS_IN_CONTEXT
                else ""
            )

            cmd_block = "\n".join(f"  {c}" for c in cmds)

            overlap_pct = int(rm.overlap_score * 100)
            block = (
                f"### {e.dish_name} [{e.recipe_id}]"
                f" | {e.cuisine} {e.course} | {e.dish_type}"
                f" | {e.servings} servings"
                f" | ingredient overlap: {overlap_pct}%\n"
                f"Matched on: {', '.join(rm.matched_ingredients)}\n\n"
                f"Tray assignments:\n{tray_table}\n\n"
                f"Verified Nosh commands:{cmd_note}\n{cmd_block}"
            )
            blocks.append(block)

        return "\n\n---\n\n".join(blocks)

    # ── internal loading ────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        t0 = time.monotonic()
        self._entries = self._build_index()
        elapsed = time.monotonic() - t0
        logger.info(
            f"RecipeFileIndex: loaded {len(self._entries)} recipes "
            f"from '{self._root}' in {elapsed:.2f}s"
        )
        self._loaded = True

    def _build_index(self) -> list[RecipeEntry]:
        pattern = os.path.join(self._root, "**", "*.recipe")
        files = glob.glob(pattern, recursive=True)
        if not files:
            logger.warning(
                f"RecipeFileIndex: no .recipe files found under '{self._root}'. "
                f"Recipe-file RAG will return empty context."
            )
            return []

        entries: list[RecipeEntry] = []
        errors = 0
        for fp in files:
            try:
                entry = self._parse_file(fp)
                if entry:
                    entries.append(entry)
            except Exception as exc:
                errors += 1
                if errors <= 5:  # don't spam the log on a broken corpus
                    logger.warning(f"RecipeFileIndex: parse error in {fp}: {exc}")
        if errors:
            logger.warning(f"RecipeFileIndex: {errors} files failed to parse (skipped).")
        return entries

    def _parse_file(self, filepath: str) -> Optional[RecipeEntry]:
        with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
            raw_lines = fh.readlines()

        # Strip BOM / carriage returns
        lines = [l.rstrip("\r\n") for l in raw_lines]

        recipe_id = os.path.basename(os.path.dirname(filepath))

        # ── parse metadata from header lines ───────────────────────────────────
        dish_name = recipe_id  # fallback
        course = ""
        cuisine = ""
        dish_type = ""
        servings = 0

        for line in lines:
            v = _parse_meta_value(line, "recipe name")
            if v:
                dish_name = v
                continue
            v = _parse_meta_value(line, "course")
            if v:
                course = v
                continue
            v = _parse_meta_value(line, "cuisine")
            if v:
                cuisine = v
                continue
            v = _parse_meta_value(line, "dish_type")
            if v:
                dish_type = v
                continue
            v = _parse_meta_value(line, "servings")
            if v:
                try:
                    servings = int(v.split()[0])
                except (ValueError, IndexError):
                    pass

        # ── split into ingredient / steps sections ─────────────────────────────
        ing_lines: list[str] = []
        steps_lines: list[str] = []
        in_ingredients = False
        in_steps = False

        for line in lines:
            stripped = line.strip()
            lower = stripped.lower()

            if lower.startswith("#ingredients"):
                in_ingredients = True
                in_steps = False
                continue
            if lower.startswith("#end ingredients") or lower == "#end":
                in_ingredients = False
                continue
            if lower.startswith("#steps"):
                in_steps = True
                in_ingredients = False
                continue

            if in_ingredients:
                ing_lines.append(stripped)
            elif in_steps:
                steps_lines.append(stripped)

        ingredient_names = _extract_ingredient_names(ing_lines)
        # Also extract servings from the #ingredients block if not found in header
        if not servings:
            for il in ing_lines:
                v = _parse_meta_value(il, "servings")
                if not v:
                    v = _parse_meta_value("#" + il.lstrip("#"), "servings")
                if v:
                    try:
                        servings = int(v.split()[0])
                    except (ValueError, IndexError):
                        pass

        tray_map = _extract_vegmap(steps_lines)
        commands = _extract_commands(steps_lines)

        if not ingredient_names and not commands:
            return None  # unreadable / empty file

        spice_counts     = _extract_spice_counts(commands)
        cook_time_seconds = _compute_recipe_cook_time(commands)

        return RecipeEntry(
            recipe_id=recipe_id,
            dish_name=dish_name,
            course=course,
            cuisine=cuisine,
            dish_type=dish_type,
            servings=servings,
            ingredient_names=ingredient_names,
            tray_map=tray_map,
            commands=commands,
            filepath=filepath,
            spice_counts=spice_counts,
            cook_time_seconds=cook_time_seconds,
        )


# ── Module-level singleton ─────────────────────────────────────────────────────

_INDEX = RecipeFileIndex()


def find_by_ingredients(
    query_ingredients: list[str],
    top_k: int = _TOP_K,
    min_overlap: float = _MIN_OVERLAP,
) -> list[RecipeMatch]:
    """Retrieve up to `top_k` .recipe entries ranked by ingredient overlap."""
    return _INDEX.find_by_ingredients(query_ingredients, top_k=top_k, min_overlap=min_overlap)


def find_by_id(
    recipe_id: str,
    target_serving: int | None = None,
) -> list[RecipeMatch]:
    """Look up all entries matching the specified recipe ID."""
    return _INDEX.find_by_id(recipe_id, target_serving=target_serving)


def format_as_rag_context(matches: list[RecipeMatch]) -> str:
    """Render a list of RecipeMatch objects as a prompt-ready string."""
    return RecipeFileIndex.format_as_rag_context(matches)


def get_reference_signals(matches: list[RecipeMatch]) -> dict:
    """Extract validator-grade signals from the best-matched .recipe file.

    Returns a dict (or {} if no matches) containing:
      dish_name              — name of the reference dish
      servings               — serving count the reference was written for
      overlap_score          — ingredient overlap fraction (0-1)
      spice_counts           — {spice_name: total_dispenses} (absolute, for ref dish)
      spice_counts_per_serving — {spice_name: dispenses_per_serving}
      cook_time_seconds      — total active cook time of the reference dish

    The orchestrator uses spice_counts_per_serving to scale to the target serving
    count and validate the LLM's dispense counts, and cook_time_seconds as a
    fallback timing reference when Qdrant returns no data.
    """
    if not matches:
        return {}
    best = matches[0]  # already sorted by overlap descending
    e = best.entry
    per_serving: dict[str, float] = {}
    if e.servings and e.servings > 0:
        per_serving = {k: v / e.servings for k, v in e.spice_counts.items()}
    return {
        "dish_name":              e.dish_name,
        "servings":               e.servings,
        "overlap_score":          best.overlap_score,
        "spice_counts":           e.spice_counts,
        "spice_counts_per_serving": per_serving,
        "cook_time_seconds":      e.cook_time_seconds,
    }
