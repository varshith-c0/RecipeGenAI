"""
Optimized single-agent orchestrator.

Replaces the 5-node LangGraph LLM pipeline with one reasoning LLM + tool-calling loop:
  RAG retrieval (code) → LLM + tools (convert + validate) → post-process (code)
"""

import os
import re
import json
import time
import random
from enum import Enum
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ServerError, APIError
from pydantic import BaseModel, Field, ValidationError

from core.rag_tool import (get_rag_reference, pick_cook_time_for_serving,
                           pick_value_for_serving)
from core.tools import (_convert_quantity_to_grams, _estimate_serving_size,
                        _get_fallback_instructions, _ANCHOR_SERVING_GRAMS)
from core.ai_cmd_validator import ai_cmds_validator
from core.distribute_ingredients import Slot, Ingredients, Consistency
from core.cmd_generator import DistributionExtended, RecipeExtended
from core.tracing import trace_function, add_span_attribute
from .utils import logger, func_timing_decorator

load_dotenv()

_MODEL = "gemini-3.1-flash-lite"
_MAX_TOOL_ROUNDS = 8
_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Transient Gemini failures (502/503/504 overloads, 429 rate limits, 408 timeouts)
# are server-side and self-heal on retry. Back off exponentially with jitter
# before giving up, instead of failing the whole request on the first blip.
_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
_MAX_API_RETRIES = 5
_RETRY_BASE_DELAY = 2.0  # seconds; doubles each attempt


def _generate_with_retry(client, **kwargs):
    """Wrapper around client.models.generate_content that retries on transient
    Gemini errors (5xx overloads like the 502 Bad Gateway, plus 429 rate limits).
    Non-retryable errors (e.g. 400/401/404) and exhausted retries re-raise so the
    caller's ServerError handler can still fail gracefully."""
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


#  Output schema 

class TrayClass(str, Enum):
    LARGE_CUT = "large_cut"
    SMALL_CUT = "small_cut"
    LIQUID = "liquid"
    BONELESS_MEAT = "boneless_meat"
    BONE_IN_MEAT = "bone_in_meat"
    GRAIN = "grain"

class _IngredientOut(BaseModel):
    ingredient_name: str = Field(description="root name of the ingredient (e.g. 'tomato', 'onion')")
    quantity: float = Field(description="quantity (use 0 if completely unknown/to-taste)")
    unit: Optional[str] = Field(default=None, description="unit: g, ml, tsp, tbsp, cup, count, pinch, etc.")
    preparation_step: Optional[str] = Field(
        default=None,
        description=(
            "This ingredient's OWN prep only, in canonical short form: lowercase, "
            "past-tense, comma-separated (e.g. 'peeled, diced'). If it is part of a "
            "SHARED prep that also involves other ingredients (a marinade, a ground "
            "paste/purée, or a pressure-cook done together), append ONLY a one-word "
            "tag for it — 'pressure-cooked', 'boiled', 'marinated', or "
            "'ground to paste' — and put the full shared step in prep_instructions. "
            "NEVER list the other co-processed ingredients here (do NOT write "
            "'pressure cooked with rice, dal, carrot…'). null if no prep."
        ),
    )
    tray_class: Optional[TrayClass] = Field(
        default=None,
        description=(
            "ONLY for ingredients placed in a tray (not oil/water/dispensed "
            "spices). Classify by how this recipe cuts/prepares it: large_cut "
            "(quartered/halved/thick-sliced), small_cut (diced/minced/thin-"
            "sliced), liquid (purée/sauce/coconut milk), boneless_meat, "
            "bone_in_meat, or grain (rice/pasta/couscous/etc). Used to compute "
            "each tray's weight limit — do not omit this for tray ingredients."
        ),
    )

class _SlotOut(BaseModel):
    number: int = Field(description="tray number 1-5")
    ingredients: list[_IngredientOut] = Field(default_factory=list)

class OrchestratorOutput(BaseModel):
    is_recipe: bool = Field(description="False if input is not a single cookable recipe")
    nosh_compatible: bool = Field(description="False if Nosh cannot execute this recipe")
    reason: Optional[str] = Field(default=None, description="Why is_recipe or nosh_compatible is False")
    recipe_name: Optional[str] = None
    serving: Optional[int] = None
    course: Optional[str] = Field(default=None, description="e.g. Main, Starter, Dessert, Snack")
    cuisine: Optional[str] = Field(default=None, description="e.g. Indian, Italian, Chinese")
    dish_type: Optional[str] = Field(default=None, description="Vegetarian, Non-Vegetarian, Vegan, etc.")
    consistency: Optional[Consistency] = Field(default=None, description="dry | semi gravy | gravy | whole meal")
    pan_type: Optional[str] = Field(
        default=None,
        description=(
            "Cookware the SOURCE recipe assumes, inferred from the text: e.g. "
            "'tawa'/'griddle', 'kadai'/'wok', 'non-stick pan', 'deep pan', "
            "'frying pan', 'pressure cooker'. null if not stated/inferable. Nosh "
            "always uses ONE non-stick induction pan regardless — this is a signal "
            "for oil/heat calibration, NOT a device selection."
        ),
    )
    covered_cook_seconds: Optional[int] = Field(
        default=None,
        description=(
            "Total seconds the recipe cooks COVERED (lid on) — sum across every "
            "'cover and cook', 'cook covered', 'lid on', 'dhak kar pakaayein' step. "
            "0 or null if the pan is never covered. Nosh has NO lid and cooks open, "
            "so this drives a water/heat compensation (see workflow)."
        ),
    )
    prep_instructions: Optional[list[str]] = Field(
        default=None,
        description=(
            "Ordered, DE-DUPLICATED list of SHARED user-prep steps the user does "
            "before Nosh starts — each stated exactly ONCE. Put here anything that "
            "spans multiple ingredients or is a composite: marinades, ground "
            "pastes/purées, and pressure-cook/boil-together groups. Use fixed "
            "grammar, one sentence per step, naming all members once, e.g. "
            "'Pressure cook rice, tur dal, beans, carrot, green peas and potato "
            "together with water and a little oil until soft.' or 'Marinate paneer "
            "with turmeric, chilli powder and salt for 15 minutes.' Do NOT repeat a "
            "step per ingredient. Purely single-ingredient mechanical prep (just "
            "'diced'/'sliced') stays in that ingredient's preparation_step and does "
            "NOT go here. null/empty if there is no shared prep."
        ),
    )
    post_cooking_step: Optional[str] = Field(default=None, description="User steps after Nosh finishes")
    slots: Optional[list[_SlotOut]] = None
    updated_instructions: Optional[str] = Field(default=None, description="Cooking steps rewritten to reference trays")
    commands: Optional[list[str]] = None


# ── Tool: unit converter ───────────────────────────────────────────────────────

def _exec_convert_ingredients(names: list, quantities: list, units: list) -> str:
    q_list = [float(q) if q else None for q in quantities]
    u_list = [u if u else None for u in units]
    try:
        results = _convert_quantity_to_grams(names, q_list, u_list)
    except Exception as e:
        return f"ERROR: {e}"
    lines = []
    for ing, q, unit, converted, _ in results:
        if q is None:
            q_s = "unknown"
        elif unit == "tsp":
            # Nosh's spice dispenser only supports whole dispenses (each = 1/4 tsp).
            dispense_count = max(1, round(q * 4)) if q > 0 else 0
            q_s = (
                f"{round(q, 2)} tsp -> USE spice dispense count = {dispense_count} "
                f"(this is the exact N for 'spice {ing} dispense N times'; do not "
                f"recompute or use the tsp value directly)"
            )
        else:
            q_s = f"{round(q, 2)} {unit}"
        lines.append(f"{ing} - is_converted: {converted}, new_quant: {q_s}")
    return "\n".join(lines) if lines else "No ingredients provided."


# ── Tool: serving size estimator ───────────────────────────────────────────────

def _exec_estimate_serving(names: list, quantities: list, units: list) -> str:
    q_list = [float(q) if q else None for q in quantities]
    u_list = [u if u else None for u in units]
    try:
        return _estimate_serving_size(names, q_list, u_list)
    except Exception as e:
        return f"ERROR: {e}"


# ── Tool: verified fallback timing lookup ──────────────────────────────────────

def _exec_get_fallback_timing(names: list) -> str:
    try:
        return _get_fallback_instructions(names)
    except Exception as e:
        return f"ERROR: {e}"


# ── Deterministic post-generation validators ───────────────────────────────────
# Each checks the LLM's own generated output with plain code (regex/arithmetic),
# never by asking the LLM to self-judge — self-calibration against fuzzy context
# is what testing showed to be unreliable. Every validator returns a list of
# plain-English issue strings (empty = pass), fed verbatim into one shared
# repair loop (see run_orchestrator, Step 4).

_MAX_REPAIR_ROUNDS = 3

# --- cook time ---------------------------------------------------------------

_WAIT_RE = re.compile(r"^wait\s+(\d+)\s+seconds", re.IGNORECASE)
_COOK_RE = re.compile(r"^cook\s+(\d+)\s+seconds", re.IGNORECASE)
# Any `cook X till <state>` is an AI-assisted (vision-model) step, not just
# `till cooked` — `cook onion till golden_brown` watches the pan exactly the same way.
# Restricting this to the literal word "cooked" made the orchestrator disagree with
# ai_cmd_validator's own _AI_COOK_RE (`^cook (.+?) till\b`), so a browning step was
# invisible here: it skipped the AI-eligibility and cook-before-dispense checks
# entirely, and counted as ZERO seconds of cook time. Group 1 is the ingredient.
_COOK_AI_RE = re.compile(r"^cook\s+(.+?)\s+till\b", re.IGNORECASE)
_TRAY_DISPENSE_RE = re.compile(r"^ingredient_tray\s+(\d+)\s+dispense", re.IGNORECASE)
# add_rice_specific_cmds keys off this exact command to override the water amount.
_COOK_RICE_RE = re.compile(r"^cook\s+rice\s+till\s+cooked\s*$", re.IGNORECASE)
_OIL_WATER_RE = re.compile(r"^(oil|water)\s+dispense\s+(\S+)\s+ml\s*$", re.IGNORECASE)
# Mirrors cmd_processor's own `startswith("oil")/startswith("water")` branch: every
# line it routes there must parse, so this must flag exactly what that branch takes.
_OIL_WATER_ANY_RE = re.compile(r"^(oil|water)\b", re.IGNORECASE)
_TIMEOUT_RE = re.compile(r"^set_temperature_timeout\s+(\d+)\s+seconds", re.IGNORECASE)
_STIR_RE = re.compile(r"^stir\s+(\S+)\s+(\d+)\s+times", re.IGNORECASE)
_STIR_SECONDS = {"saute": 16, "mix": 30, "mix_liquid": 45}  # matches cmd_processor.py's timeMap
_DEFAULT_STIR_SECONDS = 30
_EXPLICIT_TIME_RE = re.compile(r"total\s*(?:cooking|active)?\s*time\s*[:\-]?\s*\d", re.IGNORECASE)
_COOK_TIME_TOLERANCE_RATIO = 0.18  # ~15-20% band


# An AI-assisted step has no written duration — it runs until the vision model says
# the ingredient looks done — so it used to contribute nothing at all, and its mere
# presence switched off under-time detection entirely. That blank cheque is what let a
# 17.5-minute cabbage-and-potato script pass against a 30-minute target. Crediting each
# step a realistic duration keeps the check alive instead.
#
# The credit keys off the INGREDIENT, not the written state. Every AI step reaches us
# as "till cooked" — cmd_processor is what later rewrites onion to "till golden_brown"
# — so trusting the state word here would credit a 4-minute browning step the same 10
# minutes as rice cooking through. That single misestimate is enough to pull a badly
# under-timed dish back inside tolerance and hide the very bug this check exists for.
_AI_BROWNING_INGS = ("onion", "tomato")  # cmd_processor turns these into browning steps
_AI_BROWNING_SECONDS = 240
_AI_COOKED_SECONDS = 600


def _compute_total_active_minutes(commands: list) -> tuple:
    """Returns (written_active_minutes, ai_credit_minutes) parsed from generated commands.

    Kept separate so callers can tell measured time from estimated time — the written
    total is exact, the AI credit is an estimate and shouldn't be reported as fact.
    """
    total_s = 0.0
    ai_credit_s = 0.0
    for cmd in commands or []:
        c = cmd.strip()
        m = _COOK_AI_RE.match(c)
        if m:
            ing = m.group(1).strip().lower()
            ai_credit_s += (_AI_BROWNING_SECONDS if ing in _AI_BROWNING_INGS
                            else _AI_COOKED_SECONDS)
            continue
        m = _WAIT_RE.match(c) or _COOK_RE.match(c) or _TIMEOUT_RE.match(c)
        if m:
            total_s += float(m.group(1))
            continue
        m = _STIR_RE.match(c)
        if m:
            stir_type, count = m.group(1).lower(), int(m.group(2))
            total_s += count * _STIR_SECONDS.get(stir_type, _DEFAULT_STIR_SECONDS)
    return total_s / 60.0, ai_credit_s / 60.0


def _has_explicit_total_time(recipe_text: str) -> bool:
    """True if the input recipe itself states a total/cooking time — that should win over RAG."""
    return bool(_EXPLICIT_TIME_RE.search(recipe_text))


def _time_out_of_tolerance(effective_s: float, target_s: float) -> bool:
    """True if effective cook time (written + credited AI time) deviates from the RAG
    target by more than ~18%."""
    return abs(effective_s - target_s) / target_s > _COOK_TIME_TOLERANCE_RATIO


def _check_cook_time(output, rag_ref: dict, recipe_text: str) -> Optional[str]:
    target_min, matched_serving = pick_cook_time_for_serving(rag_ref["cook_time_by_serving"], output.serving)
    if not target_min or _has_explicit_total_time(recipe_text):
        return None
    target_s = target_min * 60
    written_min, ai_credit_min = _compute_total_active_minutes(output.commands)
    written_s, ai_credit_s = written_min * 60, ai_credit_min * 60
    effective_s = written_s + ai_credit_s
    if not _time_out_of_tolerance(effective_s, target_s):
        return None
    gap_s = target_s - effective_s
    # The AI steps' credit is an estimate, so name it separately — the model can only
    # change the written durations, and telling it the two numbers are one would make
    # the requested adjustment wrong by exactly the credit.
    credit_note = (
        f" (that is {written_s:.0f}s of explicit wait/stir/cook commands plus about "
        f"{ai_credit_s:.0f}s estimated for the AI-assisted 'cook ... till ...' step(s), "
        f"whose duration you cannot set directly)"
        if ai_credit_s else ""
    )
    return (
        f"Total active cook time is {effective_s:.0f} seconds{credit_note}, but the "
        f"hardware-verified target for this dish ('{rag_ref['dish_name']}', "
        f"serving {matched_serving}) is {target_s:.0f} seconds. "
        f"{'Add' if gap_s > 0 else 'Remove'} approximately {abs(gap_s):.0f} "
        f"seconds by adjusting existing wait/cook/stir durations "
        f"proportionally (do not change ingredients, quantities, or step "
        f"structure)."
    )


# --- spice-dispenser compatibility --------------------------------------------
# Nosh's spice dispenser only physically supports these 8 spices (same set
# cmd_processor.py's sanitize_spice_name() force-remaps everything to — which
# means today, an unsupported spice like "hing" or "kasuri methi" silently gets
# mapped to whichever of the 8 happens to look closest, rather than flagged.
# This validator catches that case before it ever reaches that stage.

_SPICE_ALIASES = {
    "salt": ["salt"],
    "turmeric": ["turmeric", "turmeric powder", "haldi"],
    "chilliPowder": ["chilli powder", "chili powder", "red chilli powder", "red chili powder"],
    "garamMasala": ["garam masala"],
    "corianderPowder": ["coriander powder", "dhania powder"],
    "cuminPowder": ["cumin powder", "jeera powder"],
    "cumin": ["cumin seeds", "cumin", "jeera"],
    "mustard": ["mustard seeds", "mustard", "rai"],
}
_SPICE_DISPENSE_RE = re.compile(r"^spice\s+(.+?)\s+dispense\s+(\d+)\s+times\s*$", re.IGNORECASE)
_SPICE_ANY_RE = re.compile(r"^spice\b", re.IGNORECASE)
# cmd_processor inserts these around each spice block itself; tolerated if the model
# happens to emit them so they aren't reported as malformed.
_SPICE_CONTROL_RE = re.compile(r"^spice\s+(?:position|rest)\s+\d+\s+times\s*$", re.IGNORECASE)


def _is_supported_spice(name: str) -> bool:
    # Word-boundary match, not raw substring: "salt" must not match "salted
    # butter", nor an ingredient named "red" match the "red chilli powder" alias.
    # The dict keys count as aliases too — they are the canonical dispenser names
    # the RAG examples use, so the model does emit "chilliPowder" verbatim, and
    # matching only the spaced-out aliases would flag it as unsupported.
    n = name.strip().lower()
    return any(
        re.search(rf"\b{re.escape(alias)}\b", n)
        for key, aliases in _SPICE_ALIASES.items()
        for alias in (*aliases, key.lower())
    )


def _check_spice_commands(commands: list) -> list:
    issues = []
    for cmd in commands or []:
        c = cmd.strip()
        if not _SPICE_ANY_RE.match(c) or _SPICE_CONTROL_RE.match(c):
            continue
        m = _SPICE_DISPENSE_RE.match(c)
        if not m:
            # Never skip a spice command just because it doesn't parse: downstream
            # sanitize_spice_name assumes this exact grammar and slices by token
            # position, so a malformed one is not ignored there — it is silently
            # rewritten into a DIFFERENT spice ('spice turmeric dispense 1', missing
            # 'times', becomes 'spice cumin turmeric dispense 1'). Report it here so
            # the repair loop fixes the wording before it can be mangled.
            issues.append(
                f"'{c}' is not a valid spice command. The exact grammar is "
                f"'spice [name] dispense [N] times' — [N] a whole number and the "
                f"trailing 'times' is REQUIRED, with nothing after it. Rewrite this "
                f"command in exactly that form."
            )
            continue
        if not _is_supported_spice(m.group(1)):
            issues.append(
                f"'{c}' dispenses '{m.group(1)}' via the spice dispenser, but that "
                f"is NOT one of Nosh's 8 supported spices (Salt, Turmeric, Chilli powder, "
                f"Garam masala, Coriander powder, Cumin powder, Cumin seeds, Mustard seeds). "
                f"Move this ingredient into an ingredient tray instead of the spice dispenser."
            )
    return issues


# --- tray weight & distribution ------------------------------------------------

_TRAY_SCALE_FACTORS = {
    TrayClass.LARGE_CUT: 2.0,
    TrayClass.SMALL_CUT: 1.33,
    TrayClass.LIQUID: 1.0,
    TrayClass.BONELESS_MEAT: 1.0,
    TrayClass.BONE_IN_MEAT: 1.33,
    TrayClass.GRAIN: 2.0,
}
_DEFAULT_TRAY_SCALE = 1.33  # conservative assumption when tray_class is omitted
_MAX_TRAY_EFFECTIVE_G = 400
_MAX_TRAYS = 5


def _check_tray_distribution(slots: list) -> list:
    issues = []
    if not slots:
        return issues

    numbers = sorted(s.number for s in slots)
    if len(numbers) > _MAX_TRAYS:
        issues.append(f"{len(numbers)} trays used, but Nosh only has {_MAX_TRAYS}.")
    if numbers and (numbers[0] < 1 or numbers[-1] > _MAX_TRAYS):
        issues.append(f"Tray numbers {numbers} fall outside the valid range 1-{_MAX_TRAYS}.")
    for a, b in zip(numbers, numbers[1:]):
        if b - a > 1:
            issues.append(f"There's an empty gap between tray {a} and tray {b} — trays must be consecutive.")

    for s in slots:
        eff_weight = 0.0
        for ing in s.ingredients:
            name_l = ing.ingredient_name.strip().lower()
            if name_l in ("water", "oil") or _is_supported_spice(name_l):
                issues.append(
                    f"Tray {s.number} contains '{ing.ingredient_name}', which should be "
                    f"dispensed via its dedicated dispenser (water/oil/spice), not a tray."
                )
                continue
            # Trays are metered by weight, so a non-gram quantity is unusable by the
            # hardware AND unusable here: multiplying e.g. "1.5 cup" by a scale factor
            # would score the tray at 3g and wave a ~550g load straight through the
            # limit below. Flag it and leave it out of the total rather than trusting
            # a number whose unit we don't know.
            unit_l = (ing.unit or "").strip().lower()
            if not unit_l.startswith("g"):
                issues.append(
                    f"Tray {s.number}'s '{ing.ingredient_name}' is given as "
                    f"'{ing.quantity} {ing.unit or '(no unit)'}', not grams. Trays dispense "
                    f"by weight, so every tray ingredient must be in grams. Call "
                    f"convert_ingredients_to_grams for it; if the tool cannot convert it, "
                    f"estimate a sensible gram weight yourself and use that."
                )
                continue
            scale = _TRAY_SCALE_FACTORS.get(ing.tray_class, _DEFAULT_TRAY_SCALE)
            eff_weight += (ing.quantity or 0) * scale
        if eff_weight > _MAX_TRAY_EFFECTIVE_G:
            # With all 5 trays in use, "split into another tray" is impossible advice —
            # a repair loop given only impossible options thrashes until its rounds run
            # out (observed: 466g and 598g trays shipped after 3 wasted rounds). Name
            # the only remedies that can actually work.
            if len(slots) >= _MAX_TRAYS:
                remedy = (
                    f"All {_MAX_TRAYS} trays are already in use, so splitting into a new "
                    f"tray is NOT possible. Either move some of its ingredients to an "
                    f"existing tray with spare capacity (only if dispensed at the same "
                    f"moment), or scale the ENTIRE recipe down — every ingredient and "
                    f"the serving count together — until every tray fits, or set "
                    f"nosh_compatible=false and explain in reason."
                )
            else:
                remedy = (
                    f"Split its ingredients across an additional consecutive tray, or "
                    f"move some to a tray with spare capacity."
                )
            issues.append(
                f"Tray {s.number}'s effective weight is {eff_weight:.0f}g, exceeding the "
                f"{_MAX_TRAY_EFFECTIVE_G}g limit. {remedy}"
            )
    return issues


# A same-moment split is the one tray repair that cannot change how the dish cooks:
# the identical contents enter the pan at the identical instant, just from two cups.
# So when the model exhausts its repair rounds with a tray still overweight and a
# tray slot free, do that split deterministically instead of shipping a tray the
# user physically cannot fill. Anything smarter (moving ingredients between cooking
# moments) is NOT semantics-preserving and stays the model's job.

def _slot_effective_weight(s):
    """Effective grams for a slot, or None if any ingredient is non-gram/unweighted
    (those slots already carry their own validator issue and cannot be reasoned about)."""
    eff = 0.0
    for ing in s.ingredients:
        if not (ing.unit or "").strip().lower().startswith("g") or not ing.quantity:
            return None
        eff += ing.quantity * _TRAY_SCALE_FACTORS.get(ing.tray_class, _DEFAULT_TRAY_SCALE)
    return eff


# 'cook rice till cooked' drives Nosh's RAW-rice program: cmd_processor keys a
# large calibrated water dispense (700-1300 ml) off it and runs a full rice cook
# cycle. Aimed at rice that is already cooked (fried-rice recipes), that floods
# the pan and recooks the rice to mush. Catch it deterministically so the repair
# loop makes the model swap it for plain toss/heat-through steps.

_NON_RAW_RICE_NAMES = ("rice flour", "puffed rice", "rice cake", "rice paper")
_PRECOOKED_PREP_WORDS = ("cooked", "boiled", "steamed", "leftover")


def _has_raw_rice_in_trays(output) -> bool:
    for s in (output.slots or []):
        for ing in s.ingredients:
            name = ing.ingredient_name.strip().lower()
            if "rice" not in name or any(n in name for n in _NON_RAW_RICE_NAMES):
                continue
            prep = (ing.preparation_step or "").lower()
            if any(w in name for w in _PRECOOKED_PREP_WORDS) or \
               any(w in prep for w in _PRECOOKED_PREP_WORDS):
                continue  # pre-cooked rice, not raw
            return True
    return False


def _check_cooked_rice_ai_command(output) -> list:
    issues = []
    if _has_raw_rice_in_trays(output):
        return issues
    for c in (output.commands or []):
        m = _COOK_AI_RE.match(c.strip())
        if m and "rice" in m.group(1).lower():
            issues.append(
                f"The command '{c.strip()}' runs Nosh's raw-rice program (a large "
                f"calibrated water dispense plus a full cook cycle), but this "
                f"recipe's rice is already cooked — there is no raw rice in any "
                f"tray. Running it would flood the dish and recook the rice to "
                f"mush. Remove this command; instead heat the cooked rice through "
                f"with 'stir mix ... times' and short 'wait ... seconds' steps "
                f"after its dispense."
            )
    return issues


# A marinade is applied to its target ingredients during USER PREP — once other
# tray ingredients carry a 'marinated' prep step, the marinade's mass is already
# coating them. A separate 'marinade' tray then double-counts every marinade
# component AND pours raw marinade into hot oil ahead of the food, where it
# burns before the actual ingredients arrive (observed on a dry paneer tikka).

_MARINADE_NAMES = ("marinade", "marination")


def _check_marinade_double_dispense(output) -> list:
    issues = []
    slots = output.slots or []
    marinated_items = sorted({
        ing.ingredient_name for s in slots for ing in s.ingredients
        if "marinat" in (ing.preparation_step or "").lower()
        and not any(w in ing.ingredient_name.strip().lower() for w in _MARINADE_NAMES)
    })
    if not marinated_items:
        # No tray ingredient is marked marinated — a standalone marinade/sauce
        # tray may then be a legitimate design choice; leave it alone.
        return issues
    for s in slots:
        for ing in s.ingredients:
            if any(w in ing.ingredient_name.strip().lower() for w in _MARINADE_NAMES):
                issues.append(
                    f"Tray {s.number} dispenses '{ing.ingredient_name}' "
                    f"({ing.quantity} {ing.unit or ''}) as a separate tray ingredient, "
                    f"but {', '.join(marinated_items)} are already marked 'marinated' — "
                    f"the marinade is coated onto them during user prep "
                    f"(prep_instructions), so dispensing it again double-counts its "
                    f"ingredients and pours raw marinade into hot oil before the food, "
                    f"where it will burn. Remove '{ing.ingredient_name}' from the trays "
                    f"entirely (the marinade lives in prep_instructions only), remove "
                    f"its dispense command, and renumber the remaining trays and "
                    f"commands to stay consecutive."
                )
    return issues


def _overweight_trays(output) -> list:
    """(tray_number, effective_grams) for every tray over the limit. Slots with
    non-gram/unweighted ingredients report None from _slot_effective_weight and
    are skipped — they already carry their own validator issue."""
    return [
        (s.number, w) for s in (output.slots or [])
        if (w := _slot_effective_weight(s)) is not None and w > _MAX_TRAY_EFFECTIVE_G
    ]


def _max_feasible_serving(output) -> int | None:
    """Largest serving count whose scaled-down ingredients could plausibly fit
    the trays, from total effective weight vs. total tray capacity. Assumes
    oversized ingredients can be split across trays (the auto-split's same-
    moment split), so total capacity — not the single-tray limit — is the
    binding constraint. An estimate, not a proof: the retry re-runs the full
    pipeline, which re-validates for real. None if even 1 serving can't fit
    or the current serving count is unknown."""
    serving = output.serving
    if not serving:
        return None
    total_eff = 0.0
    for s in (output.slots or []):
        w = _slot_effective_weight(s)
        if w is not None:
            total_eff += w
    if total_eff <= 0:
        return None
    # Raw capacity is unreachable in practice: splitting an oversized
    # ingredient across trays quantizes space (750g of chicken occupies two
    # trays but fills neither), and small ingredients can't share a tray with
    # a different dispense moment. 0.85 keeps the suggestion at a serving
    # count the packing can actually realize — for the reference 4-serving
    # biryani (≈2600g effective) it suggests 2, which the model verifiably
    # packs, where the unmargined floor suggests 3, which it cannot.
    packing_efficiency = 0.85
    capacity = _MAX_TRAYS * _MAX_TRAY_EFFECTIVE_G * packing_efficiency
    feasible = int(serving * capacity / total_eff)  # floor
    feasible = min(feasible, _MAX_SERVING, serving - 1)
    return feasible if feasible >= 1 else None


def _auto_split_overweight_trays(output) -> bool:
    """Split overweight trays into same-moment consecutive trays. Returns True if
    anything changed. Stops when nothing is overweight or all trays are in use."""
    changed = False
    while True:
        slots = sorted(output.slots or [], key=lambda s: s.number)
        if len(slots) >= _MAX_TRAYS:
            return changed
        target = next(
            (s for s in slots
             if (w := _slot_effective_weight(s)) is not None and w > _MAX_TRAY_EFFECTIVE_G),
            None,
        )
        if target is None:
            return changed

        def eff(ing):
            return (ing.quantity or 0) * _TRAY_SCALE_FACTORS.get(ing.tray_class, _DEFAULT_TRAY_SCALE)

        ings = sorted(target.ingredients, key=eff, reverse=True)
        if len(ings) == 1:
            # A single ingredient too heavy for one cup: split its MASS in half —
            # same convention the prompt uses for oversized composites.
            whole = ings[0]
            half = round((whole.quantity or 0) / 2.0, 1)
            keep = [whole.model_copy(update={"quantity": half})]
            move = [whole.model_copy(update={"quantity": (whole.quantity or 0) - half})]
        else:
            # Greedy balance into two halves, heaviest first.
            keep, move, keep_eff, move_eff = [], [], 0.0, 0.0
            for ing in ings:
                if keep_eff <= move_eff:
                    keep.append(ing); keep_eff += eff(ing)
                else:
                    move.append(ing); move_eff += eff(ing)

        k = target.number
        for s in slots:
            if s.number > k:
                s.number += 1
        target.ingredients = keep
        new_slot = target.model_copy(update={"number": k + 1, "ingredients": move})
        output.slots = sorted(slots + [new_slot], key=lambda s: s.number)

        # Keep the commands in step: renumber every dispense above k, then dispense
        # the new tray immediately after the original so both cups empty at the
        # same cooking moment.
        new_cmds = []
        for cmd in (output.commands or []):
            m = _TRAY_DISPENSE_RE.match(cmd.strip())
            if m:
                n = int(m.group(1))
                if n > k:
                    cmd = re.sub(r"\d+", str(n + 1), cmd, count=1)
                new_cmds.append(cmd)
                if n == k:
                    new_cmds.append(f"ingredient_tray {k + 1} dispense")
                continue
            new_cmds.append(cmd)
        output.commands = new_cmds

        logger.warning(
            f"Auto-split overweight tray {k} into trays {k} and {k + 1} "
            f"(same dispense moment): "
            f"{[i.ingredient_name for i in keep]} | {[i.ingredient_name for i in move]}."
        )
        changed = True


# --- AI-assisted cooking eligibility --------------------------------------------
# Reuses the same ai_cmds_validator the LLM can call via the validate_nosh_commands
# tool, but enforces it deterministically in the repair loop so correctness no
# longer depends on the model choosing to call the tool and fixing it itself.

def _check_ai_commands(output) -> list:
    commands = output.commands or []
    if not any(_COOK_AI_RE.match(c.strip()) for c in commands):
        return []
    try:
        dist = to_distribution_extended(output)
        bad_ings = ai_cmds_validator(commands, dist)
    except Exception as e:
        logger.warning(f"AI-command validation error: {e}")
        return []
    if not bad_ings:
        return []
    return [
        f"These ingredient(s) cannot use 'cook <ingredient> till cooked' "
        f"(AI-assisted cooking) in this recipe — because of their weight share in a "
        f"shared tray, a colour-altering spice dispensed before them, or another "
        f"substantial ingredient already cooking in the pan from an earlier step "
        f"(the single-ingredient vision model only works when that ingredient is "
        f"essentially alone in the pan; rice is the only exception): "
        f"{', '.join(sorted(set(bad_ings)))}. Replace each such "
        f"'cook ... till cooked' command with explicit manual stir/wait commands "
        f"derived from the recipe's cooking steps for that ingredient."
    ]


def _pan_contains(ing: str, in_pan: set) -> bool:
    """Whether `ing` (as named in a cook command) is already in the pan. The command
    usually names the head noun ('cook rice till cooked') while the tray carries the
    full name ('seeraga samba rice'), so accept a whole-token match too."""
    return any(ing == name or ing in name.split() for name in in_pan)


def _check_cook_before_dispense(output) -> list:
    """`cook X till cooked` points a vision model at X in the pan, so X must already
    have been dispensed when it runs — otherwise the robot watches an empty pan for
    the whole cycle. The model addresses trays by number (`ingredient_tray N
    dispense`) and only cmd_processor later resolves those to names, so map slots to
    their ingredients here to know what is actually in the pan at each point."""
    slot_ings = {
        s.number: {i.ingredient_name.strip().lower() for i in s.ingredients}
        for s in (output.slots or [])
    }
    in_pan: set = set()
    issues = []
    for cmd in (output.commands or []):
        c = cmd.strip()
        m = _TRAY_DISPENSE_RE.match(c)
        if m:
            in_pan |= slot_ings.get(int(m.group(1)), set())
            continue
        m = _COOK_AI_RE.match(c)
        if m:
            ing = m.group(1).strip().lower()
            if not _pan_contains(ing, in_pan):
                issues.append(
                    f"'{c}' runs before '{ing}' is in the pan — no ingredient_tray "
                    f"dispense for it appears earlier in the command list. AI-assisted "
                    f"cooking watches the ingredient while it cooks, so dispense the tray "
                    f"holding '{ing}' BEFORE this command, not after."
                )
    return issues


# --- serving consistency -------------------------------------------------------
# A recipe too big for the trays tempts the model to quietly shrink one ingredient
# rather than report the limit — which ships a dish cooking half the food it claims
# to serve. Anchor ingredients have a stable per-serving weight, so the tray
# contents imply a headcount that can be checked against the stated one.

_SERVING_MISMATCH_RATIO = 1.6  # tolerate rounding/garnish variance, catch ~2x fudges

# _ANCHOR_SERVING_GRAMS backs _estimate_serving_size, which only ever offers a soft,
# 1-4 clamped guess the model may override. Cross-examining a STATED serving count is
# a stricter use, and only holds for anchors that genuinely DEFINE their dish. Potato
# does not: it is a co-star as often as a centrepiece (aloo gobi, aloo matar), where
# it splits the bulk with another vegetable and 100 g/serving of potato alone is
# simply the wrong yardstick — demanding more of it drives the serving count down
# chasing a target the dish never had to meet.
_VALIDATOR_ANCHORS = [(a, g) for a, g in _ANCHOR_SERVING_GRAMS if a != "potato"]

# A marinade or ground paste is collapsed into ONE slot ingredient whose weight is
# the WHOLE slurry — e.g. "marinated chicken" 750 g is ~500 g chicken PLUS paste and
# yogurt. Counting that combined mass against a per-serving PROTEIN weight overstates
# the headcount (750/150 = 5 vs a true ~3-4), which would silently inflate the serving
# count or false-fire this check. So a composite mass is not pure anchor weight — skip
# it. Pressure-cooked / boiled groups are NOT collapsed (they keep individual
# ingredients at true weights), so their tags are absent here and stay countable.
_COLLAPSED_PREP_TAGS = ("marinated", "ground to paste")


# Nosh's salt-dispense counts and rice water volumes are hardware-calibrated tables
# keyed 1-4; there is no 5th or 6th entry to extrapolate from, and a pan that holds 4
# servings does not hold 6. A larger count is not a preference to honour but a recipe
# that must be scaled down before it can be cooked at all.
_MAX_SERVING = 4


def _check_serving_range(output) -> list:
    serving = output.serving
    if not serving or serving <= _MAX_SERVING:
        return []
    return [
        f"The recipe claims {serving} servings, but Nosh's pan and its calibrated salt "
        f"and water tables only go up to {_MAX_SERVING}. Scale the ENTIRE recipe down to "
        f"{_MAX_SERVING} servings or fewer: multiply every ingredient quantity by "
        f"{_MAX_SERVING}/{serving} and set serving to {_MAX_SERVING}. Here the whole "
        f"recipe must move together — unlike a serving/quantity mismatch, nothing is "
        f"wrong with the recipe's proportions, only its total size. Do not simply lower "
        f"the serving number while leaving the quantities alone: that fills the trays "
        f"for {serving} people and seasons the food for {_MAX_SERVING}."
    ]


def _check_serving_consistency(output) -> list:
    serving = output.serving
    if not serving or serving <= 0:
        return []
    # _ANCHOR_SERVING_GRAMS is ordered protein-first: the first anchor present is the
    # most reliable headcount signal, so judge on it alone rather than averaging.
    # An anchor is routinely SPLIT across trays — a 400 g tray cap forces it, and the
    # RAG corpus itself lists rice as two 200 g slot entries at 4 servings — so judge
    # the anchor's TOTAL across every tray. Reading one entry alone would score a
    # correctly-split recipe as half-sized and demand the model inflate it.
    for anchor, per_serving in _VALIDATOR_ANCHORS:
        total = 0.0
        label = anchor
        for s in (output.slots or []):
            for ing in s.ingredients:
                name = ing.ingredient_name.strip().lower()
                if anchor != name and anchor not in name.split():
                    continue
                prep = (ing.preparation_step or "").strip().lower()
                if any(tag in prep for tag in _COLLAPSED_PREP_TAGS):
                    continue  # combined marinade/paste mass, not pure anchor weight
                if not (ing.unit or "").strip().lower().startswith("g") or not ing.quantity:
                    continue  # non-gram quantities are already reported by the tray check
                total += ing.quantity
                label = ing.ingredient_name
        if total <= 0:
            continue  # anchor absent (or only in non-gram units): try the next one
        implied = total / per_serving
        if max(implied / serving, serving / implied) < _SERVING_MISMATCH_RATIO:
            return []
        return [
            f"The recipe claims {serving} serving(s), but its {label} "
            f"({total:.0f}g in total across all trays) feeds about {implied:.1f} at "
            f"~{per_serving}g per serving. Exactly one of the two numbers is wrong, so "
            f"change ONE of them: either (a) the {label} was shrunk to fit a tray — "
            f"restore its real quantity from the recipe, or (b) the quantity is right "
            f"and the serving count is not — set serving to "
            f"{min(_MAX_SERVING, max(1, round(implied)))}. "
            f"Do NOT rescale every ingredient and the serving count together: that "
            f"keeps the two in the same proportion and cannot resolve this. If the "
            f"recipe's real quantity cannot fit Nosh's 5 trays at any sensible serving "
            f"size, set nosh_compatible=false and say so in reason."
        ]
    return []


# --- tray dispense coverage ----------------------------------------------------
# A tray is a physical cup the user fills before pressing start. Declaring a tray
# but never dispensing it means the user preps an ingredient that never enters the
# dish — the tray check only sees `slots` and cannot catch this, and the
# cook-before-dispense check only looks at trays named by a cook command. The
# reverse (dispensing a tray that was never declared) reaches cmd_processor's
# slot->name map with no entry for that number.

def _check_dispense_coverage(output) -> list:
    slot_ings = {
        s.number: sorted(i.ingredient_name.strip().lower() for i in s.ingredients)
        for s in (output.slots or [])
    }
    if not slot_ings:
        return []
    dispensed = {
        int(m.group(1))
        for m in (_TRAY_DISPENSE_RE.match(c.strip()) for c in (output.commands or []))
        if m
    }
    issues = []
    for num in sorted(set(slot_ings) - dispensed):
        issues.append(
            f"Tray {num} ({', '.join(slot_ings[num]) or 'empty'}) is never dispensed — "
            f"no 'ingredient_tray {num} dispense' appears in the commands. The user "
            f"would prep and load it for nothing and the dish would cook without it. "
            f"Fix it whichever way is actually right for this ingredient: dispense the "
            f"tray at the point the recipe adds it, OR — if it is only added after "
            f"cooking ends (a garnish, a squeeze of lemon, a final herb) — remove the "
            f"tray entirely and put that step in post_cooking_step instead."
        )
    for num in sorted(dispensed - set(slot_ings)):
        issues.append(
            f"'ingredient_tray {num} dispense' refers to tray {num}, which does not "
            f"exist — the declared trays are {sorted(slot_ings)}. Dispense an existing "
            f"tray, or add the tray holding the ingredient this step needs."
        )
    return issues


# --- oil / water dispense sanity -----------------------------------------------
# cmd_processor parses these positionally (`float(parts[2])`) and asserts the
# amount is > 0, so a malformed or zero dispense is an uncaught exception, not a
# bad command. Oil is decided entirely by the model and never overridden, so its
# amount is worth policing; a deep-fry quantity copied from the source recipe
# would otherwise reach the pan verbatim. Water is only checked as a pan-overflow
# guard, and only for non-rice dishes — for rice, add_rice_specific_cmds replaces
# the amount with Nosh's calibrated per-serving volume, so the model's figure is
# discarded and must NOT be flagged.

_MAX_TOTAL_OIL_ML = 50    # single non-stick pan; well above the ~20 ml the prompt asks for
_MAX_WATER_ML = 1500      # calibrated rice max is 1300 ml at 4 servings — beyond this overflows

def _check_oil_water_commands(output) -> list:
    commands = output.commands or []
    is_rice = any(_COOK_RICE_RE.match(c.strip()) for c in commands)
    issues = []
    total_oil = 0.0
    for cmd in commands:
        c = cmd.strip()
        if not _OIL_WATER_ANY_RE.match(c):
            continue
        m = _OIL_WATER_RE.match(c)
        if not m:
            issues.append(
                f"'{c}' is not a valid dispense command. The exact form is "
                f"'oil dispense [N] ml' or 'water dispense [N] ml', with N a positive number."
            )
            continue
        liquid, raw = m.group(1).lower(), m.group(2)
        try:
            amount = float(raw)
        except ValueError:
            issues.append(f"'{c}' has a non-numeric amount ('{raw}'). Use a positive number of ml.")
            continue
        if amount <= 0:
            issues.append(
                f"'{c}' dispenses {raw} ml. A dispense must be a positive amount — "
                f"remove the command entirely if nothing should be dispensed."
            )
            continue
        if liquid == "oil":
            total_oil += amount
        elif not is_rice and amount > _MAX_WATER_ML:
            issues.append(
                f"'{c}' dispenses {amount:.0f} ml of water, which overflows Nosh's single "
                f"pan (max ~{_MAX_WATER_ML} ml). Reduce it to what the pan can hold, or set "
                f"nosh_compatible=false if the dish genuinely needs more."
            )
    if total_oil > _MAX_TOTAL_OIL_ML:
        issues.append(
            f"The recipe dispenses {total_oil:.0f} ml of oil in total, far more than Nosh's "
            f"single non-stick pan needs (~10-15 ml typical, {_MAX_TOTAL_OIL_ML} ml ceiling). "
            f"This is usually a deep-fry quantity copied from the source recipe — Nosh cannot "
            f"deep-fry, so convert the step to shallow-frying with a thin film of oil."
        )
    return issues


# --- water volume vs the RAG reference -----------------------------------------
# The check above is only an overflow guard: any amount under 1500 ml passes, so a
# dish can be dispensed 150 ml where the hardware-verified figure is 250 ml and
# nothing objects. For a covered dish, moisture matters as much as time — a dense
# ingredient steams tender on the water, and too little leaves it firm no matter how
# long the pan runs. Same source and same authority rules as the cook-time check:
# skipped for rice, where add_rice_specific_cmds overwrites the amount downstream.

# Looser than cook time — absorbency genuinely varies with cut size and how much steam
# the lid holds — but not so loose it only catches absurdities. Both directions hurt:
# short leaves dense ingredients firm, long makes a dry sabzi soupy.
_WATER_TOLERANCE_RATIO = 0.25
_MIN_WATER_TARGET_ML = 50      # below this the reference is a splash, not a steaming volume

# cmd_processor appends its own "water dispense 50 ml" after every AI-assisted onion or
# tomato step (deglazing the browned base). That water is code-decided and invisible in
# the commands we validate here, so it must be counted — otherwise this check demands
# the model add water that the pipeline is already going to add for it.
_POST_AI_WATER_ML = 50
_POST_AI_WATER_INGS = ("onion", "tomato")


def _check_water_volume(output, rag_ref: dict) -> list:
    target_ml, matched_serving = pick_value_for_serving(
        rag_ref.get("water_by_serving") or {}, output.serving
    )
    if not target_ml or target_ml < _MIN_WATER_TARGET_ML:
        return []
    commands = output.commands or []
    if any(_COOK_RICE_RE.match(c.strip()) for c in commands):
        return []  # rice water is code-decided downstream; never police it here
    total_ml = 0.0
    for c in commands:
        c = c.strip()
        m = _OIL_WATER_RE.match(c)
        if m and m.group(1).lower() == "water":
            try:
                total_ml += float(m.group(2))
            except ValueError:
                return []  # malformed dispense is already reported by the check above
            continue
        m = _COOK_AI_RE.match(c)
        if m and m.group(1).strip().lower() in _POST_AI_WATER_INGS:
            total_ml += _POST_AI_WATER_ML  # cmd_processor will add this downstream
    if abs(total_ml - target_ml) / target_ml <= _WATER_TOLERANCE_RATIO:
        return []
    gap = target_ml - total_ml
    return [
        f"The recipe dispenses {total_ml:.0f} ml of water in total, but the "
        f"hardware-verified figure for this dish ('{rag_ref['dish_name']}', serving "
        f"{matched_serving}) is {target_ml:.0f} ml. "
        f"{'Add' if gap > 0 else 'Remove'} about {abs(gap):.0f} ml by adjusting the "
        f"existing 'water dispense' amounts — too little water leaves dense "
        f"ingredients underdone however long they cook, too much makes a dry dish "
        f"soupy. Do not change ingredients, quantities, or step structure."
    ]


# --- spice presence ------------------------------------------------------------
# An LLM round was observed dropping the ENTIRE seasoning block — salt, turmeric,
# chilli powder, cumin powder, garam masala — while every other validator passed,
# because _check_spice_commands polices the spice commands that exist, never the
# ones that don't. A dish with correct timing and water but zero masala ships
# looking perfect and tasting of nothing. So: every supported spice the recipe
# text calls for must have a dispense command. Only fires for spices the recipe
# itself mentions, so a dish that genuinely uses two spices isn't badgered into
# eight. (cmd_processor's _insert_salt_dispense stays as the last-resort layer for
# salt specifically, whose count is code-decided; the other spices' counts are the
# model's call, so the repair loop is the right layer for them.)

# Longest alias first, and each match masks its span before shorter aliases are
# tried — otherwise "cumin powder" in the text would ALSO match the bare "cumin"
# alias and falsely demand cumin seeds. Keys count as aliases (the model emits
# "chilliPowder" verbatim), mirroring _is_supported_spice.
_SPICE_ALIAS_KEY_PAIRS = sorted(
    ((alias.lower(), key)
     for key, aliases in _SPICE_ALIASES.items()
     for alias in (*aliases, key)),
    key=lambda p: -len(p[0]),
)
# Phrases that contain a spice word but are not the dispenser spice: "mustard oil"
# must not demand a mustard-seed dispense. Masked out before alias matching.
_SPICE_FALSE_FRIENDS = ("mustard oil",)


def _spices_in_text(text: str) -> set:
    t = (text or "").lower()
    for phrase in _SPICE_FALSE_FRIENDS:
        t = t.replace(phrase, " ")
    found = set()
    for alias, key in _SPICE_ALIAS_KEY_PAIRS:
        pattern = re.compile(rf"\b{re.escape(alias)}\b")
        if pattern.search(t):
            found.add(key)
            t = pattern.sub(" ", t)  # claim the span so shorter aliases can't re-match it
    return found


def _dispensed_spices(commands: list) -> set:
    found = set()
    for cmd in commands or []:
        m = _SPICE_DISPENSE_RE.match(cmd.strip())
        if not m:
            continue
        name = m.group(1).strip().lower()
        for alias, key in _SPICE_ALIAS_KEY_PAIRS:
            if re.search(rf"\b{re.escape(alias)}\b", name):
                found.add(key)
                break
    return found


def _check_spice_presence(output, recipe_text: str) -> list:
    wanted = _spices_in_text(recipe_text)
    if not wanted:
        return []
    missing = sorted(wanted - _dispensed_spices(output.commands))
    if not missing:
        return []
    return [
        f"The recipe calls for {', '.join(missing)} but the commands never dispense "
        f"{'it' if len(missing) == 1 else 'them'} — no matching 'spice [name] dispense "
        f"[N] times' exists. Add the missing dispense command(s) at the point the "
        f"recipe adds each spice, with the food it seasons already in the pan, using "
        f"the exact dispenser name(s): "
        + "; ".join(f"'spice {m} dispense [N] times'" for m in missing) + "."
    ]


# --- staged ingredients sharing a tray ------------------------------------------
# The prompt already says "same tray only if dispensed at the same moment AND same
# cooking instruction", but nothing enforced it, and violating it for a STAGED
# ingredient is physically self-defeating rather than merely untidy: a tray is
# dispensed as a unit, so an ingredient the recipe cooks to a state as its own step
# (onion browned golden, tomato cooked down soft) cannot reach that state with the
# next step's bulk already in the pan. Onion sharing with tomato steams instead of
# browning; tomato sharing with 200 g of cabbage never cooks down before the cabbage
# hits. Deliberately narrow: a rule only fires when the recipe ITSELF stages that
# ingredient, and co-tray items are allowed if the staging sentence names them
# (added together by the recipe) or they are minor riders (whole spices, chillies).

_STAGE_RULES = (
    # (ingredient names, state words that mark a staging sentence)
    (("onion", "shallot"), ("golden", "brown", "translucent", "caramel")),
    (("tomato",), ("soft", "mushy", "pulpy")),
)
_MINOR_CO_TRAY_G = 15  # tempering-scale items riding along: bay leaf, chilli, cinnamon


def _staging_sentences(recipe_text: str) -> list:
    """(staged_names, sentence) for each sentence that cooks an ingredient to a state."""
    out = []
    for sentence in re.split(r"[.;\n]", (recipe_text or "").lower()):
        for ings, words in _STAGE_RULES:
            staged = [a for a in ings if a in sentence]
            if staged and any(w in sentence for w in words):
                out.append((staged, sentence))
    return out


def _check_staged_ingredient_trays(output, recipe_text: str) -> list:
    stages = _staging_sentences(recipe_text)
    if not stages:
        return []
    issues = []
    for s in (output.slots or []):
        names = [(i.ingredient_name.strip().lower(), i.quantity or 0) for i in s.ingredients]
        for staged, sentence in stages:
            anchor = next((n for n, _q in names if any(a in n for a in staged)), None)
            if anchor is None:
                continue
            offenders = [
                n for n, q in names
                if n != anchor
                and q > _MINOR_CO_TRAY_G                      # tempering riders are fine
                and not any(a in n for a in staged)
                # named in the staging sentence = the recipe adds them together
                and not any(tok in sentence for tok in n.split())
            ]
            if offenders:
                issues.append(
                    f"Tray {s.number} holds {anchor} together with {', '.join(offenders)}, "
                    f"but the recipe cooks the {anchor} to a stage on its own first "
                    f"(\"{sentence.strip()}\"). A tray dispenses as a single unit, so "
                    f"{', '.join(offenders)} would enter the pan at the same instant and "
                    f"the {anchor} can never reach that stage. Move "
                    f"{', '.join(offenders)} to a separate tray dispensed at the step "
                    f"where the recipe actually adds {'it' if len(offenders) == 1 else 'them'}."
                )
    return issues


def _run_deterministic_validators(output, rag_ref: dict, recipe_text: str) -> list:
    issues = []
    cook_time_issue = _check_cook_time(output, rag_ref, recipe_text)
    if cook_time_issue:
        issues.append(cook_time_issue)
    issues.extend(_check_spice_commands(output.commands))
    issues.extend(_check_tray_distribution(output.slots))
    issues.extend(_check_ai_commands(output))
    issues.extend(_check_cooked_rice_ai_command(output))
    issues.extend(_check_marinade_double_dispense(output))
    issues.extend(_check_cook_before_dispense(output))
    # An oversized recipe usually trips the anchor check too, but with the opposite
    # remedy: the range check says "shrink the food to fit the pan", the consistency
    # check says "raise the serving count to match the food". Emitting both in one
    # repair round hands the model contradictory instructions, so the range check
    # wins — resize first, then re-examine proportions on the next round.
    serving_range_issues = _check_serving_range(output)
    issues.extend(serving_range_issues)
    if not serving_range_issues:
        issues.extend(_check_serving_consistency(output))
    issues.extend(_check_dispense_coverage(output))
    issues.extend(_check_oil_water_commands(output))
    issues.extend(_check_water_volume(output, rag_ref))
    issues.extend(_check_spice_presence(output, recipe_text))
    issues.extend(_check_staged_ingredient_trays(output, recipe_text))
    return issues


# ── Tool: command validator ────────────────────────────────────────────────────

def _exec_validate_commands(commands: list, slots_json: str) -> str:
    try:
        slots_data = json.loads(slots_json) if isinstance(slots_json, str) else slots_json
        slot_objs = []
        for sd in slots_data:
            ings = [
                Ingredients(
                    ingredient_name=i.get("ingredient_name", ""),
                    quantity=float(i.get("quantity", 0)),
                    unit=i.get("unit"),
                    preparation_step=i.get("preparation_step"),
                )
                for i in sd.get("ingredients", [])
            ]
            slot_objs.append(Slot(number=sd["number"], ingredients=ings))

        mock_recipe = RecipeExtended(
            recipe_name="tmp", serving=1,
            consistency=Consistency.DRY, slots=slot_objs,
            updated_instructions="", commands=commands,
        )
        mock_dist = DistributionExtended(
            is_recipe=True, nosh_compatible=True, reason=None, recipe=mock_recipe
        )
        issues = ai_cmds_validator(commands, mock_dist)
        if not issues:
            return "VALID: No AI-command issues found."
        return (
            f"ISSUES: These ingredients cannot use 'cook X till cooked' due to "
            f"weight/spice constraints — replace with manual stir/wait commands: "
            f"{', '.join(issues)}"
        )
    except Exception as e:
        logger.warning(f"validate_commands tool error: {e}")
        return f"SKIP: validation error ({e})"


def _dispatch(name: str, args: dict) -> str:
    if name == "convert_ingredients_to_grams":
        return _exec_convert_ingredients(
            args.get("ingredient_names", []),
            args.get("quantities", []),
            args.get("units", []),
        )
    if name == "estimate_serving_size":
        return _exec_estimate_serving(
            args.get("ingredient_names", []),
            args.get("quantities", []),
            args.get("units", []),
        )
    if name == "get_fallback_timing":
        return _exec_get_fallback_timing(args.get("ingredient_names", []))
    if name == "validate_nosh_commands":
        return _exec_validate_commands(
            args.get("commands", []),
            args.get("slots_json", "[]"),
        )
    return f"Unknown tool: {name}"


#  Gemini tool declarations 

def _make_tools() -> types.Tool:
    return types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="convert_ingredients_to_grams",
            description=(
                "Convert all recipe ingredient quantities to Nosh-compatible units: "
                "grams for solid ingredients, ml for water/oil, tsp for Nosh-supported "
                "spices (Salt, Turmeric, Chilli powder, Garam masala, Coriander powder, "
                "Cumin powder, Cumin seeds, Mustard seeds). "
                "Call this FIRST, before distributing into trays."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "ingredient_names": types.Schema(
                        type=types.Type.ARRAY,
                        items=types.Schema(type=types.Type.STRING),
                        description="Root names of ingredients (lowercase, singular)"
                    ),
                    "quantities": types.Schema(
                        type=types.Type.ARRAY,
                        items=types.Schema(type=types.Type.NUMBER),
                        description="Quantities (use 0 if unknown or 'to taste')"
                    ),
                    "units": types.Schema(
                        type=types.Type.ARRAY,
                        items=types.Schema(type=types.Type.STRING),
                        description="Units (g, ml, tsp, tbsp, cup, count, pinch, clove, etc.; empty string if none)"
                    ),
                },
                required=["ingredient_names", "quantities", "units"],
            ),
        ),
        types.FunctionDeclaration(
            name="estimate_serving_size",
            description=(
                "Estimate serving size deterministically from a recognized anchor "
                "ingredient's quantity (paneer, chicken, mutton, fish, rice, dal, "
                "potato, pasta, etc.), using fixed per-serving reference weights. "
                "Call this ONLY when the recipe text does NOT explicitly state a "
                "serving count. If the text explicitly says e.g. 'serves 2' or "
                "'servings: 4', use that instead and do not call this tool."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "ingredient_names": types.Schema(
                        type=types.Type.ARRAY,
                        items=types.Schema(type=types.Type.STRING),
                        description="Root names of ingredients (lowercase, singular)"
                    ),
                    "quantities": types.Schema(
                        type=types.Type.ARRAY,
                        items=types.Schema(type=types.Type.NUMBER),
                        description="Quantities (use 0 if unknown or 'to taste')"
                    ),
                    "units": types.Schema(
                        type=types.Type.ARRAY,
                        items=types.Schema(type=types.Type.STRING),
                        description="Units (g, ml, tsp, tbsp, cup, count, pinch, clove, etc.; empty string if none)"
                    ),
                },
                required=["ingredient_names", "quantities", "units"],
            ),
        ),
        types.FunctionDeclaration(
            name="get_fallback_timing",
            description=(
                "Look up hardware-verified, pre-tested cook timing for known ingredients "
                "(e.g. paneer, chicken, mutton, fish, cauliflower, chana, broccoli, "
                "cabbage, and ~30 others). Call this ONCE with ALL extracted ingredient "
                "base names, right alongside convert_ingredients_to_grams — not just for "
                "ones you think are vague. If a match is returned, its DURATIONS AND "
                "FREQUENCIES are the exact numbers to use for that ingredient's cook "
                "step, overriding any timing you'd otherwise infer from the recipe text "
                "or RAG. The match is plain English, NOT Nosh command syntax — you must "
                "still translate it into valid commands: 'Cook for N seconds while "
                "Xing every M seconds' -> 'cook N seconds stir X every M seconds'; a "
                "bare 'Cook for N seconds' (no stirring mentioned) -> 'wait N seconds'; "
                "'Stir once'/'Stir K times' -> 'stir mix K times'. Never emit a bare "
                "'cook N seconds' line — that is not valid syntax. Exception: ingredients "
                "using Nosh's AI-assisted cooking (onion, tomato, rice, potato, okra, "
                "suji, atta, pasta, millet) always use 'cook [ingredient] till cooked' "
                "instead — ignore any fallback match for those."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "ingredient_names": types.Schema(
                        type=types.Type.ARRAY,
                        items=types.Schema(type=types.Type.STRING),
                        description="Root names of all extracted ingredients (lowercase, singular)"
                    ),
                },
                required=["ingredient_names"],
            ),
        ),
        types.FunctionDeclaration(
            name="validate_nosh_commands",
            description=(
                "Validate whether the generated commands use AI-assisted cooking correctly. "
                "Returns VALID or lists ingredients that need manual stir/wait commands instead of "
                "'cook X till cooked'. Call this AFTER generating commands."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "commands": types.Schema(
                        type=types.Type.ARRAY,
                        items=types.Schema(type=types.Type.STRING),
                        description="Generated list of Nosh commands"
                    ),
                    "slots_json": types.Schema(
                        type=types.Type.STRING,
                        description=(
                            'JSON string: [{"number": 1, "ingredients": '
                            '[{"ingredient_name": "onion", "quantity": 150, "unit": "g"}]}]'
                        ),
                    ),
                },
                required=["commands", "slots_json"],
            ),
        ),
    ])


# ── System instruction ─────────────────────────────────────────────────────────

_SYSTEM_INSTRUCTION = r"""
You are an expert Nosh chef-programmer. Given a recipe text and optional RAG context (similar verified Nosh recipes), you will produce a complete, validated Nosh command script in a single reasoning session using two tools.

═══════════════════════════════════════════════════════════════════
NOSH OVERVIEW
═══════════════════════════════════════════════════════════════════

Nosh is an autonomous single-pot cooking robot with:
• Induction heating + stirring/sautéing arm
• 5 ingredient trays (each dispensed once, all contents together) # they dispense one by one try not at once 
• Spice dispenser for: Salt, Turmeric, Chilli powder, Garam masala,
  Coriander powder, Cumin powder, Cumin seeds, Mustard seeds
• Auto water & oil dispensers
• AI cooking models for: Onion, Tomato, Rice, Potato, Okra, Suji, Atta, Pasta, Millet

Nosh CANNOT: knead/roll/ferment, bake/fry/grill/steam, use a pressure cooker,
move/transfer/strain ingredients, or cook multi-pot dishes.

═══════════════════════════════════════════════════════════════════
TRAY WEIGHT RULES
═══════════════════════════════════════════════════════════════════

Max tray effective weight: 400g (enforced via type-specific scale factors)

Type            | Raw limit | Scale factor
Large-cut       |   200g    |   ×2.0
Small-cut       |   300g    |   ×1.33
Liquid/pourable |   400g    |   ×1.0
Boneless meat   |   400g    |   ×1.0
Bone-in meat    |   300g    |   ×1.33
Grains          |   200g    |   ×2.0

Mixed tray: sum each ingredient's (weight × scale factor) — must stay ≤ 400.
Overflow fix: move ingredient to next tray OR split into portions across two consecutive trays.

Distribution rules:
• Same tray only if dispensed at the same moment AND same cooking instruction.
• Oil, water, and supported spices go in their dedicated dispensers — never in trays.
• Trays must be consecutive (no empty tray between two filled ones).
• Every ingredient must appear — none omitted.
• Max 5 trays total.
• NEVER shrink an ingredient's quantity to make it fit a tray limit. Cutting the
  rice in half while still claiming 4 servings cooks the wrong amount of food and
  silently changes the user's recipe. When the recipe genuinely will not fit in 5
  trays, you have exactly two honest options:
    1. Scale the WHOLE recipe down to a serving size that does fit — every
       ingredient scaled by the same factor, and `serving` updated to match.
    2. If no sensible serving size fits, set nosh_compatible=false and explain
       the limit in `reason`.
  Report the limitation; never disguise it by quietly editing one quantity.

═══════════════════════════════════════════════════════════════════
NOSH COMMAND FORMAT
═══════════════════════════════════════════════════════════════════

Stove:
  stove start                                 – always first command
  stove stop                                  – always last command
  stove level [low|mid|high|very_high]        – heat level
  heat_till [°C]                              – wait until temperature reached

Dispensing:
  ingredient_tray [N] dispense                – dispense tray N
  oil dispense [N] ml                         – ONLY pourable liquid cooking oil
                                                 (sunflower/vegetable/mustard/olive/etc).
                                                 NEVER ghee, butter, coconut oil, lard, or
                                                 any fat that solidifies at room temperature
                                                 — those go in an ingredient tray instead.
     QUANTITY: If the recipe states an oil amount, use exactly that. Otherwise do
     NOT guess high — Nosh cooks in a single non-stick pan and needs far less oil
     than the amount printed in most recipes. Pick a sensible default by method:
       • tempering / sautéing aromatics / base gravy → 10 ml (~1 tbsp)
       • shallow / pan-frying (paneer, tikki, fish, etc.) → 15 ml (~1 tbsp)
     Scale up only slightly for very large batches; never exceed ~20 ml unless the
     recipe explicitly calls for more. Oil already mixed into a marinade/paste in a
     tray is separate — do NOT re-add it as a pan dispense.
  water dispense [N] ml
     → For a recipe using "cook rice till cooked", the system overrides this amount
       with Nosh's calibrated per-serving volume for open-pan rice, so do not agonise
       over it — state the recipe's own figure and move on.
  spice [name] dispense [N] times             – each = ¼ tsp

AI-Assisted Cooking (for supported ingredients):
  cook [ingredient] till cooked
  → For onion/tomato, the system automatically appends a rinse (water dispense 50 ml
    + stir mix 1 times) right after this command. Do NOT add those two lines yourself
    — just emit "cook onion till cooked" / "cook tomato till cooked" and move on to
    the next step.
  → PAN MUST BE ESSENTIALLY EMPTY OF OTHER INGREDIENTS: "cook X till cooked" uses a
    vision model trained on that ONE ingredient alone in the pan. It is valid only
    when no other substantial ingredient from an earlier step is already cooking in
    the pan. So if onion was cooked first and potato is added afterwards, potato
    canNOT use "cook potato till cooked" — write manual stir/wait commands for it
    instead. RICE IS THE ONLY EXCEPTION — "cook rice till cooked" works even with
    other ingredients already in the pan. (Ingredients dispensed together in the
    SAME tray as X, and small aromatics/tempering like cumin seeds, do not count as
    "other ingredients" here.)

Combined cook+stir:
  cook [total_sec] seconds stir [saute|mix|mix_liquid] every [x] [seconds]

Standalone:
  wait [N] seconds
  stir [saute|mix|mix_liquid] [N] times

IMPORTANT:
• First two commands: stove start → heat_till 75
• Last command: stove stop
• Use ingredient_tray N dispense (not ingredient names) for dispensing
• Follow original recipe order strictly
• No zero-quantity commands

═══════════════════════════════════════════════════════════════════
UPDATED INSTRUCTIONS FORMAT
═══════════════════════════════════════════════════════════════════

Rewrite cooking steps so they reference trays:
  "Add tray 2 (onion)." instead of "Add the onions."
  "Dispense tray 1 (curry leaves), then tray 2 (onion)." if sequential.
  "Dispense oil [N] ml." for oil.
  "Add salt ½ tsp, turmeric ¼ tsp." for spices.
Keep all timing, temperature, and stirring details unchanged.
Exclude preparation steps (chopping, marinating) and post-cooking steps.

═══════════════════════════════════════════════════════════════════
INGREDIENT EXTRACTION RULES
═══════════════════════════════════════════════════════════════════

For each ingredient:
• Use the root/base name (e.g., "tomato" for "chopped tomatoes")
• Extract quantity (numeric float) and unit
• Non-numeric quantities (to-taste, a few) → quantity = 0
• Preparation step = mechanical/cleaning/marinating work the user must do before Nosh starts
• Post-cooking step = what the user does after Nosh finishes (serving, garnishing shape)
• Supported units: g, ml, tsp, tbsp, cup, clove, pinch, count
• Do NOT treat size words (small, medium, large) as units
• If an ingredient is pre-cooked using equipment Nosh doesn't have — pressure-cooked,
  boiled separately, deep-fried, baked, steamed, etc. — before it joins the single pot,
  that cooking is a preparation_step for that ingredient (done by the user beforehand).
  The ingredient enters Nosh's pot already in its cooked form (e.g. "cooked toor dal",
  "boiled potato") — do NOT generate any Nosh commands trying to cook it from raw, and
  do NOT give it a short simmer expecting it to fully cook in that time.

═══════════════════════════════════════════════════════════════════
YOUR WORKFLOW (follow in order)
═══════════════════════════════════════════════════════════════════

1. VALIDATE
   • Is it a single recipe Nosh can cook? If not → is_recipe=false or nosh_compatible=false + reason.

2. EXTRACT
   • Extract recipe_name, post_cooking_step.
   • Extract course (e.g. Main, Starter, Dessert, Snack), cuisine (e.g. Indian,
     Italian, Chinese), and dish_type (Vegetarian, Non-Vegetarian, Vegan, etc.)
     from the recipe content — infer if not explicitly stated.
   • Serving size: if the recipe text explicitly states one (e.g. "serves 2",
     "servings: 4"), use that number directly. If it does NOT state one, do NOT
     guess it yourself — call the estimate_serving_size tool (step 3) and use its
     result.
   • Nosh cooks 1 to 4 servings — `serving` must never exceed 4. If the recipe
     states more (e.g. "serves 6"), scale the WHOLE recipe down: multiply EVERY
     ingredient quantity by 4/stated_serving and set serving=4. Scaling the count
     without the quantities fills the trays for 6 people and seasons for 4.
   • List ALL ingredients with base_name, quantity, unit, preparation_step.
   • NORMALIZE PREPARATION (this makes prep output stable — follow exactly):
       - Each ingredient's preparation_step holds ONLY its own prep, canonical form:
         lowercase, past-tense, comma-separated ("peeled, diced"). null if none.
       - When several ingredients share ONE prep — a marinade, a ground paste/purée,
         or a pressure-cook/boil done together — describe that shared step ONCE in
         the recipe-level prep_instructions list (fixed grammar, naming every member
         a single time), and give each member ONLY a short tag in its own
         preparation_step: "pressure-cooked" / "boiled" / "marinated" /
         "ground to paste". NEVER re-list the sibling ingredients inside a single
         ingredient's preparation_step. (Bad: potato prep = "diced and pressure
         cooked with rice, dal, carrot, peas, oil". Good: potato prep = "peeled,
         diced, pressure-cooked" AND one prep_instructions line names the whole
         group.)
       - The tag word must stay ("pressure-cooked"/"boiled") so the pre-cooked
         rule below still recognises the ingredient enters already cooked.
   • Extract pan_type: the cookware the recipe assumes (tawa/griddle, kadai/wok,
     non-stick pan, deep pan, frying pan, pressure cooker). Infer from verbs too —
     "shallow fry"/"roast on tawa" ⇒ tawa; "deep fry"/"bhuna in kadai" ⇒ kadai;
     null if genuinely unclear. Nosh still uses its one non-stick pan; this is only
     a calibration signal (see step 4).
   • Extract covered_cook_seconds: add up the duration of EVERY step that cooks
     with a lid on ("cover and cook 10 min", "cook covered", "put the lid",
     "dum"/"dhak kar"). Convert to seconds and sum them. Use 0 if the pan is never
     covered. Nosh cooks open, so this is compensated in step 4.
   • Write the single-pot cooking steps (no prep/post steps).

3. CONVERT  [TOOL CALL: convert_ingredients_to_grams]
   • Pass every extracted ingredient (name, quantity, unit).
   • Use 0 for unknown quantities; empty string for no unit.
   • Use the returned converted quantities for all subsequent steps.
   • If serving size was not explicitly stated in the recipe text, ALSO call
     estimate_serving_size with the same ingredients and use its returned value
     as the final serving count.
   • ALSO call get_fallback_timing with ALL extracted ingredient names, in the
     same round — not only ones you suspect are vague. This is mandatory, not
     conditional.

4. RESOLVE VAGUENESS
   • For any ingredient where get_fallback_timing returned a VERIFIED sequence,
     translate its exact durations/frequencies into valid Nosh command syntax
     (see get_fallback_timing's description for the translation rules) — this
     overrides RAG, the recipe text's own wording, and your own judgment, EXCEPT
     for ingredients using Nosh's AI-assisted cooking (onion, tomato, rice,
     potato, okra, suji, atta, pasta, millet), which always use
     "cook [ingredient] till cooked". Never copy the English sentence as a
     command — it must become a real "cook...stir...every", "wait", or "stir"
     command.
   • For everything else: if a quantity is still 0/unknown, estimate from RAG
     examples or recipe context. If a cooking step is vague, fill in
     timing/temperature from RAG examples.
   • OIL IS A SPECIAL CASE — DO NOT COPY OIL AMOUNTS FROM RAG EXAMPLES. The oil
     quantities shown in RAG reference recipes are unreliable and usually too
     high for Nosh's single non-stick pan. Ignore them. Decide oil from the
     recipe's own stated amount if it gives one; otherwise use the method-based
     default from the OIL QUANTITY note in the command reference (≈10 ml for
     tempering/sautéing, ≈15 ml for shallow/pan-frying), and do NOT exceed ~20 ml
     unless the input recipe explicitly calls for more. This overrides RAG.
   • CALIBRATE TOTAL COOK TIME AGAINST RAG: the input recipe's own stated cook
     durations (e.g. "cook 5 minutes", "simmer 3 minutes") describe a
     conventional stovetop and are usually NOT reliable for Nosh — Nosh's
     single-pot induction cooking is slower and typically needs much longer
     active cook time for the same dish. If a RAG reference recipe is a close
     match (same or similar dish), find its "Cooking Time" field for the
     serving count closest to this recipe's — that is a hardware-validated
     total active cook time for this kind of dish, and it takes priority over
     the input recipe's stated durations when the two disagree substantially.
     Distribute your wait/stir/simmer commands (after dispensing, after
     spices, after water, etc.) so their total duration approximates that
     RAG-verified figure, scaled proportionally if this recipe's serving count
     differs from the matched RAG entry. Do not silently default to the
     input text's short stovetop timings just because they're explicitly
     stated — explicit but wrong is still wrong.
   • PAN-TYPE CALIBRATION (use the extracted pan_type to set oil + heat):
       - tawa / griddle / non-stick  → thin oil film: oil at the LOW end (≈10–15 ml),
         heat "mid" for shallow-frying/roasting. Do not pool oil.
       - kadai / wok / deep pan / "frying pan" with "deep fry" → Nosh CANNOT deep
         fry (no lid, single shallow pan). Treat it as pan-/shallow-frying: cap oil
         at ~20 ml and use "high" heat only for the brief searing/bhuna phase, then
         drop to mid. Never emit a large deep-fry oil quantity.
       - pressure cooker → Nosh has none; the pressure-cooked step is a user
         preparation_step (ingredient enters already cooked) — do NOT try to
         reproduce it with heat/time. (Consistent with the pre-cooked rule above.)
       - This adjusts oil/heat only; it never changes which pan is used.
   • LID / COVERED-COOKING COMPENSATION (use covered_cook_seconds):
       Nosh has no lid and cooks OPEN, so any step the recipe cooks covered will
       lose the moisture a lid would have trapped. When covered_cook_seconds > 0,
       compensate on the corresponding simmer/cook step(s):
         (a) Add extra water to offset evaporation — about +10 ml of water per full
             minute of covered cooking (i.e. covered_cook_seconds/60 * 10 ml),
             capped at +60 ml total, ADDED to whatever water the recipe already
             calls for. Skip this only for dishes that must stay dry (dry roasts/
             stir-fries where the recipe adds no water at all).
         (b) Keep that step's heat at "low" or "mid" (never high) so the added water
             simmers off slowly rather than flashing away, mimicking a covered simmer.
         (c) Keep the covered step's full duration in the wait/stir timing — do NOT
             shorten it just because the pan is open.
       Reflect the added water as a "water dispense [N] ml" command at the start of
       that step, and mention it in updated_instructions.
   • If strict-mode flags are set (see user message), flag instead of estimating.

5. DISTRIBUTE
   • Group ingredients by dispense moment and cooking instruction.
   • For each ingredient placed in a tray, set its tray_class (large_cut,
     small_cut, liquid, boneless_meat, bone_in_meat, or grain) based on how
     THIS recipe cuts/prepares it — this is required, not optional, for every
     tray ingredient (never set it for oil/water/dispensed spices).
   • MARINADES & GROUND PASTES ARE ONE MASS — DO NOT SPLIT THEM BY COMPONENT.
     When several ingredients were combined into a single marinade or a single
     ground paste/purée (the members you tagged 'marinated' or 'ground to paste'
     under ONE prep_instructions line), they are now one homogeneous slurry that
     cannot be separated back into its parts. Emit them as ONE slot ingredient,
     never as separate lines spread across trays:
       - ingredient_name: name the combined mass, e.g. "marinated chicken",
         "onion-tomato paste", "green masala paste".
       - quantity: the SUM of every member's grams. (Dispenser spices such as
         salt/turmeric/chilli powder are NOT slot members and are excluded — they
         stay in the spice dispenser even when the recipe also marinates with
         them.)
       - tray_class: ONE class for the whole mass — boneless_meat / bone_in_meat
         if a meat anchors it, otherwise liquid for a pure paste/purée.
       - Keep the component breakdown ONLY in the prep_instructions line; do NOT
         re-list the parts as individual slot ingredients.
     If the combined mass exceeds the tray weight limit, split THIS SINGLE
     ingredient across consecutive trays BY MASS (equal representative portions,
     dispensed back-to-back) — never put some components in one tray and the rest
     in another. Group strictly by shared prep: if the recipe has TWO different
     marinades/pastes (two separate prep_instructions lines), keep them as two
     separate combined ingredients, not one merged blob. This rule does NOT apply
     to 'pressure-cooked' or 'boiled' groups — those keep their individual
     ingredients so grain/rice tray and water handling still work.
   • Apply tray weight rules using each ingredient's scale factor; split
     overweight groups.
   • Write updated_instructions referencing tray numbers.

6. GENERATE COMMANDS
   • Translate updated_instructions into the Nosh command format.
   • Use ingredient_tray [N] dispense for each group.
   • Use cook X till cooked for eligible AI ingredients.

7. VALIDATE COMMANDS  [TOOL CALL: validate_nosh_commands]
   • Pass the commands list and slots as JSON.
   • If ISSUES returned: replace the flagged 'cook X till cooked' commands with
     manual stir/wait commands derived from the updated_instructions.

8. OUTPUT
   • Return the complete JSON with all fields filled.
"""


# ── Prompt builder ─────────────────────────────────────────────────────────────

def _build_prompt(recipe_text: str, rag_context: str,
                  is_ing_check: bool, is_instr_check: bool,
                  target_serving: int | None = None) -> str:
    strictness = []
    if is_ing_check:
        strictness.append(
            "STRICT (ingredients): if any ingredient quantity is 0 and cannot be "
            "estimated from the recipe or RAG context, set is_recipe=false with a reason."
        )
    if is_instr_check:
        strictness.append(
            "STRICT (instructions): if cooking steps remain too vague to produce valid "
            "Nosh commands even after consulting RAG context, set nosh_compatible=false."
        )
    if not strictness:
        strictness = [
            "LENIENT: estimate missing quantities and resolve vague instructions "
            "using the RAG reference recipes and general cooking knowledge."
        ]

    rag_block = (
        f"\n# Similar Nosh-Tested Recipes (use as reference)\n{rag_context}\n"
        if rag_context else ""
    )

    serving_block = ""
    if target_serving:
        serving_block = (
            f"\n# Required Serving Count\n"
            f"The user requires EXACTLY {target_serving} serving(s). Scale EVERY "
            f"ingredient quantity proportionally from the recipe's own serving "
            f"count to {target_serving} serving(s), set serving={target_serving}, "
            f"and size all commands (water, oil, spices, cook times) for "
            f"{target_serving} serving(s). Do NOT call estimate_serving_size — "
            f"the serving count is already decided.\n"
        )

    return (
        f"# Recipe to Process\n\n{recipe_text.strip()}\n"
        f"{rag_block}"
        f"{serving_block}\n"
        f"# Validation Mode\n"
        + "\n".join(f"• {s}" for s in strictness)
        + "\n\nNow follow the workflow:\n"
          "1. Call convert_ingredients_to_grams with all ingredients.\n"
          "2. Also call get_fallback_timing with all ingredient names (mandatory, "
          "same round as step 1) and use any VERIFIED sequence it returns verbatim "
          "for that ingredient's cook step.\n"
          "3. If the recipe text does not explicitly state a serving count, also "
          "call estimate_serving_size with the same ingredients and use its result.\n"
          "4. Distribute into slots and write updated_instructions.\n"
          "5. Generate commands.\n"
          "6. Call validate_nosh_commands; fix any flagged commands.\n"
          "7. Return complete JSON."
    )


# ── Core orchestrator ──────────────────────────────────────────────────────────

@func_timing_decorator
@trace_function("run_orchestrator")
def run_orchestrator(
    recipe_text: str,
    is_ing_check: bool,
    is_instr_check: bool,
    target_serving: int | None = None,
) -> tuple:
    """
    Single LLM + tool-calling loop.
    Returns (OrchestratorOutput | None, reason | None, error | None).
    `reason` is a plain string for most failures; recoverable failures (recipe
    too large for the trays) return a dict {"reason", "error_code",
    "max_feasible_serving"?} so the API can offer the user a retry-with-fewer-
    servings flow. `target_serving`, when set, forces the model to scale the
    whole recipe to exactly that many servings.
    """
    # ── Step 1: RAG retrieval (pure code) ────────────────────────────────────
    rag_ref = get_rag_reference(recipe_text)
    rag_context = rag_ref["context"]
    # Log exactly what RAG retrieved so it's verifiable, not just a char count:
    # the matched dish, its per-serving cook times (used for cook-time repair),
    # and the full grounding context text passed to the model.
    logger.info(
        f"RAG retrieved | dish_name={rag_ref.get('dish_name')!r} | "
        f"cook_time_by_serving={rag_ref.get('cook_time_by_serving')} | "
        f"context={len(rag_context)} chars"
    )
    logger.info(f"RAG context text:\n{rag_context if rag_context else '(empty — no match)'}")
    add_span_attribute("rag.context_chars", len(rag_context))
    add_span_attribute("rag.dish_name", str(rag_ref.get("dish_name")))
    add_span_attribute("rag.cook_time_by_serving", str(rag_ref.get("cook_time_by_serving")))

    prompt = _build_prompt(recipe_text, rag_context, is_ing_check, is_instr_check,
                           target_serving=target_serving)
    client = genai.Client(api_key=_GEMINI_API_KEY)
    tools = _make_tools()

    config_tools = types.GenerateContentConfig(
        system_instruction=_SYSTEM_INSTRUCTION,
        tools=[tools],
        temperature=0,
    )
    config_extract = types.GenerateContentConfig(
        system_instruction=_SYSTEM_INSTRUCTION,
        response_mime_type="application/json",
        response_schema=OrchestratorOutput,
        temperature=0,
    )

    contents = [types.Content(role="user", parts=[types.Part(text=prompt)])]
    
    try:
        # ── Step 2: Tool-calling phase ────────────────────────────────────────
        tool_rounds = 0
        for _ in range(_MAX_TOOL_ROUNDS):
            response = _generate_with_retry(
                client, model=_MODEL, contents=contents, config=config_tools,
            )
            if not response.candidates:
                return None, None, Exception("Empty response from model")

            candidate = response.candidates[0]
            # A candidate can come back with no content/parts (e.g. finish_reason
            # MAX_TOKENS, SAFETY, RECITATION). Guard against the NoneType iteration
            # instead of assuming .content.parts is always a populated list.
            parts = getattr(candidate.content, "parts", None) if candidate.content else None
            if not parts:
                finish = getattr(candidate, "finish_reason", None)
                logger.warning(
                    f"Candidate had no content parts (finish_reason={finish}); "
                    f"ending tool phase after {tool_rounds} tool rounds"
                )
                break

            fn_calls = [
                p for p in parts
                if getattr(p, "function_call", None)
            ]
            if not fn_calls:
                logger.info(f"Tool phase done after {tool_rounds} tool rounds")
                break

            # Execute all tool calls in this round
            contents.append(candidate.content)
            fn_responses = []
            for part in parts:
                if getattr(part, "function_call", None):
                    fc = part.function_call
                    logger.info(f"Tool call [{tool_rounds}]: {fc.name}")
                    result = _dispatch(fc.name, dict(fc.args))
                    fn_responses.append(
                        types.Part.from_function_response(
                            name=fc.name, response={"output": result}
                        )
                    )
            contents.append(types.Content(role="user", parts=fn_responses))
            tool_rounds += 1

        add_span_attribute("orchestrator.tool_rounds", tool_rounds)

        # ── Step 3: Structured JSON extraction ───────────────────────────────
        contents.append(types.Content(
            role="user",
            parts=[types.Part(text="Output the complete JSON result now.")],
        ))
        final = _generate_with_retry(
            client, model=_MODEL, contents=contents, config=config_extract,
        )

        if not final.text:
            finish = None
            if final.candidates:
                finish = getattr(final.candidates[0], "finish_reason", None)
            return None, None, Exception(
                f"Model returned no JSON text in extraction phase (finish_reason={finish})"
            )

        output = OrchestratorOutput.model_validate_json(final.text)
        logger.info(
            f"Orchestrator done: is_recipe={output.is_recipe}, "
            f"nosh_compatible={output.nosh_compatible}, "
            f"commands={len(output.commands or [])}"
        )
        add_span_attribute("orchestrator.is_recipe", output.is_recipe)
        add_span_attribute("orchestrator.nosh_compatible", output.nosh_compatible)

        if not output.is_recipe or not output.nosh_compatible:
            return None, output.reason, None

        # ── Step 4: Deterministic verify → repair → re-verify loop ────────────
        # Runs ALL validators (cook time, spice-dispenser compatibility, tray
        # weight/distribution) every round and batches every issue found into
        # ONE repair message, rather than repairing one check at a time — fixing
        # a spice issue can shift a tray's weight or a cook step's duration, so
        # checking everything together avoids a fix-one-break-another ping-pong.
        #
        # Stopping conditions (checked in this order each round):
        #   1. No issues found -> done, success.
        #   2. Repair round produced unparseable output -> bail out, keep last
        #      good output (never ship a broken JSON over a merely-imperfect one).
        #   3. Issue count didn't strictly improve vs. the previous round -> stop
        #      early ("stuck"); burning through remaining rounds against a
        #      non-improving fix wastes calls for no benefit.
        #   4. Hard cap of _MAX_REPAIR_ROUNDS reached -> stop regardless.
        # Whatever remains unresolved is logged, not hidden — output still
        # ships (best-effort), but the mismatch is visible in logs/telemetry
        # rather than silently passed off as fully validated.
        last_model_content = final.candidates[0].content
        prev_issue_count = None
        # Most recent output that still had populated slots. When the model
        # refuses mid-repair it usually empties the slots, but the serving
        # suggestion for the retry flow needs the pre-refusal tray weights.
        last_slotted = output if output.slots else None

        for round_num in range(1, _MAX_REPAIR_ROUNDS + 1):
            issues = _run_deterministic_validators(output, rag_ref, recipe_text)
            add_span_attribute(f"repair.round{round_num}.issue_count", len(issues))

            if not issues:
                logger.info(f"All validators passed after {round_num - 1} repair round(s).")
                break
            if prev_issue_count is not None and len(issues) >= prev_issue_count:
                logger.warning(
                    f"Repair made no improvement ({prev_issue_count} -> {len(issues)} "
                    f"issues); stopping early instead of burning remaining rounds."
                )
                break
            prev_issue_count = len(issues)

            logger.info(f"Repair round {round_num}/{_MAX_REPAIR_ROUNDS}: {len(issues)} issue(s) found.")
            repair_msg = (
                "Deterministic validation found the following issue(s) in your "
                "generated output:\n" + "\n".join(f"- {i}" for i in issues) + "\n"
                "Fix ALL of these issues. Do not change anything else unless "
                "required to fix them. Output the complete corrected JSON now."
            )
            contents.append(last_model_content)
            contents.append(types.Content(role="user", parts=[types.Part(text=repair_msg)]))
            repaired = _generate_with_retry(
                client, model=_MODEL, contents=contents, config=config_extract,
            )
            if not repaired.text:
                logger.warning(
                    f"Repair round {round_num} returned no JSON text, stopping; "
                    f"keeping last good output."
                )
                break
            try:
                output = OrchestratorOutput.model_validate_json(repaired.text)
                last_model_content = repaired.candidates[0].content
                if output.slots:
                    last_slotted = output
            except (ValidationError, json.JSONDecodeError) as e:
                logger.warning(f"Repair round {round_num} produced invalid output, stopping: {e}")
                break
        else:
            logger.warning(f"Exhausted {_MAX_REPAIR_ROUNDS} repair rounds; issues may remain.")

        # A repair round may legitimately flip the verdict — e.g. the model
        # realizes mid-repair that the recipe cannot fit the trays. The
        # pre-loop compatibility check cannot see that, so re-check here;
        # otherwise an output self-declared incompatible (often with emptied
        # slots and zero commands) ships to the caller as a success.
        if not output.is_recipe or not output.nosh_compatible:
            logger.info(
                f"Model declared output not shippable during repair: "
                f"is_recipe={output.is_recipe}, nosh_compatible={output.nosh_compatible}."
            )
            # If the refusal is size-related (the last slotted output had
            # overweight trays), attach the retry-with-fewer-servings hint —
            # same shape as the overweight guard below.
            src = output if output.slots else last_slotted
            if src is not None and _overweight_trays(src):
                payload = {
                    "reason": output.reason or "Recipe could not be processed.",
                    "error_code": "too_large",
                }
                suggested = _max_feasible_serving(src)
                if suggested:
                    payload["max_feasible_serving"] = suggested
                return None, payload, None
            return None, output.reason, None

        # Deterministic last resort: if the model left a tray overweight and a tray
        # slot is free, do the same-moment split ourselves rather than shipping a
        # cup the user physically cannot fill. Runs after the loop so the model
        # always gets first crack at a smarter fix (moving items between moments).
        _auto_split_overweight_trays(output)

        # Unconditional final re-check for logging/telemetry — the loop above
        # only measures issues BEFORE each repair, so the last repair's own
        # result is never otherwise confirmed. Don't assume; measure it.
        final_issues = _run_deterministic_validators(output, rag_ref, recipe_text)
        add_span_attribute("repair.final_issue_count", len(final_issues))
        if final_issues:
            logger.warning(f"Shipping output with {len(final_issues)} unresolved issue(s): {final_issues}")
        else:
            logger.info("Final output passes all deterministic validators.")

        # Deterministic guard: an overweight tray is not a quality nit — the
        # hardware cup physically cannot hold it, so the cook would fail
        # mid-run. If the model couldn't fix it across the repair rounds and
        # the auto-split couldn't either (all trays in use), refuse honestly
        # instead of shipping a recipe the machine cannot execute. Other
        # unresolved issues (cook time, water) still ship best-effort above.
        overweight = _overweight_trays(output)
        if overweight:
            detail = ", ".join(f"tray {n} holds {w:.0f}g effective" for n, w in overweight)
            logger.warning(f"Refusing to ship physically impossible output: {detail}.")
            add_span_attribute("orchestrator.refused_overweight_trays", str([n for n, _ in overweight]))
            msg = (
                f"This recipe's quantities are too large to fit Nosh's "
                f"{_MAX_TRAYS} ingredient trays ({detail}; the limit is "
                f"{_MAX_TRAY_EFFECTIVE_G}g effective per tray)."
            )
            payload = {"reason": msg, "error_code": "too_large"}
            suggested = _max_feasible_serving(output)
            if suggested:
                payload["max_feasible_serving"] = suggested
                payload["reason"] = (
                    f"{msg} Try again with {suggested} serving(s) or fewer."
                )
            else:
                payload["reason"] = (
                    f"{msg} Try reducing the quantities or the number of "
                    f"servings and submitting again."
                )
            return None, payload, None

        return output, None, None

    except ServerError as e:
        logger.error(f"Gemini API server error: {e}")
        return None, None, e
    except (ValidationError, json.JSONDecodeError) as e:
        logger.error(f"Output parsing error: {e}")
        return None, None, e
    except Exception as e:
        logger.exception(f"Unexpected orchestrator error: {e}")
        return None, None, e


# ── Convert output → DistributionExtended (for cmd_processor) ─────────────────

def to_distribution_extended(output: OrchestratorOutput) -> DistributionExtended:
    slots = [
        Slot(
            number=s.number,
            ingredients=[
                Ingredients(
                    ingredient_name=i.ingredient_name,
                    quantity=i.quantity,
                    unit=i.unit,
                    preparation_step=i.preparation_step,
                )
                for i in s.ingredients
            ],
        )
        for s in (output.slots or [])
    ]

    recipe_ext = RecipeExtended(
        recipe_name=output.recipe_name or "Recipe",
        serving=output.serving or 1,
        course=output.course,
        cuisine=output.cuisine,
        dish_type=output.dish_type,
        consistency=output.consistency or Consistency.DRY,
        slots=slots,
        updated_instructions=output.updated_instructions or "",
        commands=output.commands or [],
        prep_instructions=output.prep_instructions or [],
    )

    return DistributionExtended(
        is_recipe=True,
        nosh_compatible=True,
        reason=None,
        recipe=recipe_ext,
        post_cooking_step=output.post_cooking_step,
    )
