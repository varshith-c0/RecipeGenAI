import re
from nltk.stem import WordNetLemmatizer

from core.tracing import trace_function, add_span_attribute
from .utils import func_timing_decorator

lemmatizer = WordNetLemmatizer()

# --- Cooking-order (pan-state) rule for AI-assisted cooking --------------------
# The AI-assisted `cook X till cooked` command relies on a vision model trained on
# ONE ingredient alone in the pan. If a *different*, substantial ingredient from an
# earlier cooking step is already in the pan when that command runs, the model
# can't recognise doneness and the command is invalid. Rice is the sole exception —
# its model tolerates a mixed pan (e.g. rice cooked with peas/veg).
_MULTI_INGREDIENT_OK = {"rice"}
# Ignore aromatics/tempering below this weight (bay leaf, cinnamon, cloves, a
# clove of garlic, a green chilli) so "temper cumin seeds, then AI-cook onion"
# stays valid — only a genuine prior main ingredient (veg/protein/legume) counts.
_MIN_CONFUSING_INGREDIENT_G = 20.0
_TRAY_DISPENSE_RE = re.compile(r"^ingredient_tray\s+(\d+)\s+dispense", re.IGNORECASE)
_AI_COOK_RE = re.compile(r"^cook\s+(.+?)\s+till\b", re.IGNORECASE)


def _norm_ing(name: str) -> str:
    n = lemmatizer.lemmatize(name.strip().lower())
    return "okra" if n == "lady finger" else n


def _co_tray_ings(x: str, slots_j: dict) -> set:
    """All ingredients that share a tray with x (dispensed together with it).
    These are the 'same tray' case handled by the weight-ratio rules, so the
    cooking-order check must not double-flag them."""
    co = set()
    for s in slots_j.values():
        if x in s["ings"]:
            co.update(s["ings"])
    return co


def prior_cooked_ai_conflicts(cmd_l: list[str], slots_j: dict) -> list[str]:
    """Ingredients whose `cook X till cooked` fires while a different, substantial
    ingredient from an EARLIER tray is already in the pan. Walks the command list
    in order, tracking what each `ingredient_tray N dispense` has added to the pan.
    Rice is exempt (mixed pan is fine for it)."""
    flagged = []
    pan = {}  # ingredient name -> cumulative weight (g) currently in the pan
    for cmd in cmd_l:
        c = cmd.strip()
        m = _TRAY_DISPENSE_RE.match(c)
        if m:
            n = int(m.group(1))
            for ing, w in slots_j.get(n, {}).get("ing_w", {}).items():
                pan[ing] = pan.get(ing, 0.0) + w
            continue
        m = _AI_COOK_RE.match(c)
        if m:
            x = _norm_ing(m.group(1))
            if x in _MULTI_INGREDIENT_OK:
                continue
            co_tray = _co_tray_ings(x, slots_j)
            others = [
                ing for ing, w in pan.items()
                if ing != x and ing not in co_tray and w >= _MIN_CONFUSING_INGREDIENT_G
            ]
            if others:
                flagged.append(x)
    return flagged


def slot_has_color_altering_spice(ai_ing: str, slots: dict, curr_slot_idx: int, cmds: list[str]) -> bool:
    ai_cmd_pattern = re.compile(rf"^cook\s+{re.escape(ai_ing)}\s+till\b", re.IGNORECASE)
    ai_cmd_pos = next(i for i, cmd in enumerate(cmds) if ai_cmd_pattern.search(cmd))
    spice_cmd_pattern = re.compile(r"^spice\s+.+\s+dispense\b", re.IGNORECASE)
    spice_cmd_present = any(spice_cmd_pattern.search(cmd) for cmd in cmds[: ai_cmd_pos])
    color_altering_spice_l = []
    color_altering_spice_dispensed = any(ing in color_altering_spice_l for i in range(1, curr_slot_idx+1) for ing in slots[i]["ings"])
    return spice_cmd_present or color_altering_spice_dispensed


def is_similar_ratio(w1: float, w2: float, tolerance: float = 0.10) -> bool:
    ratio = w1 / w2
    return (1 - tolerance) <= ratio <= (1 + tolerance)


@func_timing_decorator
@trace_function()
def ai_cmds_validator(cmd_l: list[str], distribution) -> list[str]:
    """Returns list of ingredient names that require manual (non-AI) cooking commands."""
    pattern = re.compile(r"cook\s+(.+?)\s+till\s+(.+)", re.IGNORECASE)
    ai_ings_l = [match.group(1).lower() for cmd in cmd_l if (match := pattern.search(cmd))]
    add_span_attribute("ings_with_ai_cmd", ai_ings_l)

    if len(ai_ings_l) == 0:
        return []

    recipe_j = distribution.recipe

    slots_j = {}
    for slot in recipe_j.slots:
        n = slot.number
        slot_ings = slot.ingredients
        slots_j[n] = {"ings": [], "total_w": 0.0, "ai_info": [], "ing_w": {}}

        if len(slot_ings) == 0:
            continue

        slot_w_g = 0
        for ing in slot_ings:
            name, w_g = lemmatizer.lemmatize(ing.ingredient_name.lower()), ing.quantity
            if name == "lady finger":
                name = "okra"
            slots_j[n]["ings"].append(name)
            slots_j[n]["ing_w"][name] = slots_j[n]["ing_w"].get(name, 0.0) + (w_g or 0.0)
            if name in ai_ings_l:
                slots_j[n]["ai_info"].append({"name": name, "w": w_g})
            slot_w_g += w_g
        slots_j[n]["total_w"] = slot_w_g

        for info in slots_j[n]["ai_info"]:
            info["w_pct"] = info["w"] / slot_w_g
    add_span_attribute("slots_j", slots_j)

    ai_ings_req_manual_inst = []
    for i, slot in slots_j.items():
        for ai_ing in slot["ai_info"]:
            if ai_ing["name"] in ["okra", "potato"]:
                if (ai_ing["w_pct"] < 0.80) or slot_has_color_altering_spice(ai_ing["name"], slots_j, i, cmd_l):
                    ai_ings_req_manual_inst.append(ai_ing["name"])
            elif ai_ing["name"] in ["onion"]:
                if (ai_ing["w_pct"] < 0.80):
                    ai_ings_req_manual_inst.append(ai_ing["name"])
            elif ai_ing["name"] == "tomato":
                tomato_w, tomato_w_pct = ai_ing["w"], ai_ing["w_pct"]
                onion_info = next((info for info in slot["ai_info"] if info["name"] == "onion"), None)
                if onion_info:
                    onion_w = onion_info["w"]
                    if not is_similar_ratio(tomato_w, onion_w):
                        ai_ings_req_manual_inst.append("tomato")
                else:
                    if tomato_w_pct < 0.80:
                        ai_ings_req_manual_inst.append("tomato")
    # Add the cooking-order rule: an AI ingredient is also invalid if a different
    # substantial ingredient from an earlier tray is already in the pan when its
    # `cook X till cooked` runs (the single-ingredient vision model can't cope).
    prior_conflicts = prior_cooked_ai_conflicts(cmd_l, slots_j)
    # Union, preserving order and de-duplicating.
    ai_ings_req_manual_inst = list(dict.fromkeys(ai_ings_req_manual_inst + prior_conflicts))
    add_span_attribute("output.prior_cooked_conflicts", prior_conflicts)
    add_span_attribute("output.ai_ings_req_manual_inst", ai_ings_req_manual_inst)
    return ai_ings_req_manual_inst
