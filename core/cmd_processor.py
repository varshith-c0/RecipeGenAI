import re
import json
from typing import Union, Tuple
from pathlib import Path
from difflib import SequenceMatcher
from .utils import logger, func_timing_decorator
from .tracing import trace_function, add_span_attribute


# Every calibrated table below (salt counts, rice water) is keyed 1-4 because that is
# the range Nosh's pan was tuned over. A serving count outside it is a bug upstream,
# not a value to extrapolate, so clamp it here rather than let a raw dict lookup raise.
_MAX_SERV_SIZE = 4


def _clamp_serv_size(serv_size, caller):
    """Force serv_size into the 1-4 range these tables are calibrated for.

    Clamping is damage limitation, not a fix: at serving 6 the trays hold 6 portions
    of food but the rice gets 4 servings' worth of water. The orchestrator's serving
    range check is what should stop this reaching us, so log loudly when it doesn't.
    """
    clamped = max(1, min(_MAX_SERV_SIZE, serv_size))
    if clamped != serv_size:
        logger.warning(
            f"{caller}: serving size {serv_size} is outside the calibrated 1-{_MAX_SERV_SIZE} "
            f"range; clamping to {clamped}. The dish will be cooked for {clamped} servings "
            f"even though the trays are filled for {serv_size}."
        )
    return clamped


_VEG_DISPENSE_RE = re.compile(r"^vegetable\s+\S+\s+dispense", re.IGNORECASE)
_STOVE_STOP_RE = re.compile(r"^stove\s+stop", re.IGNORECASE)


def _insert_salt_dispense(line_l, reqd_count):
    """Add the missing salt dispense rather than shipping an unsalted dish.

    This function's job is the salt COUNT, and it used to return untouched when there
    were no salt commands to count — so an LLM round that simply forgot salt produced
    food nobody would eat, silently. Salt quantity is already entirely code-decided
    here, so the safe correction is to add it, not to preserve the omission.

    Placement: right after the last ingredient dispense, so the salt lands on food
    that is actually in the pan and before the stove stops. This is a fallback for a
    generation fault the orchestrator's salt validator should have caught, so it is
    logged rather than applied quietly.
    """
    anchor = next((i for i in range(len(line_l) - 1, -1, -1)
                   if _VEG_DISPENSE_RE.match(line_l[i].strip())), None)
    if anchor is None:
        # Nothing dispensed from a tray — fall back to just before the stove stops.
        anchor = next((i for i in range(len(line_l) - 1, -1, -1)
                       if _STOVE_STOP_RE.match(line_l[i].strip())), len(line_l)) - 1
    logger.warning(
        f"No salt dispense in generated commands; inserting "
        f"'spice salt dispense {reqd_count} times' after line {anchor} "
        f"('{line_l[anchor].strip() if 0 <= anchor < len(line_l) else ''}')."
    )
    line_l[anchor + 1:anchor + 1] = [
        "spice position 1 times",
        f"spice salt dispense {reqd_count} times",
        "spice rest 1 times",
    ]
    return line_l


def fix_salt_dispense(line_l, consistency, serv_size):
    from .distribute_ingredients import Consistency # to prevent circular import
    serv_size = _clamp_serv_size(serv_size, "fix_salt_dispense")

    mapping = {
        Consistency.DRY       : {1: 1, 2: 2, 3: 3, 4: 4},
        Consistency.WHOLE_MEAL: {1: 2, 2: 4, 3: 5, 4: 5},
        Consistency.GRAVY     : {1: 2, 2: 3, 3: 4, 4: 5},
        Consistency.SEMI_GRAVY: {1: 2, 2: 3, 3: 4, 4: 5},
    }
    pattern = re.compile(r"(spice salt dispense (\d+) times)")

    disp_count_l = []
    disp_cmd_idx_l = []
    reqd_count = mapping[consistency][serv_size]

    for idx, line in enumerate(line_l):
        match = pattern.search(line)
        if match:
            disp_cmd_idx_l.append(idx)
            disp_count_l.append(int(match.group(2)))

    if not disp_cmd_idx_l:  # no salt dispense command at all
        return _insert_salt_dispense(line_l, reqd_count)

    total_count = sum(disp_count_l)
    if total_count == reqd_count:
        return line_l

    if total_count < reqd_count:
        rem_count = reqd_count - total_count
        for i, (cmd_idx, count) in enumerate(zip(disp_cmd_idx_l, disp_count_l)):
            # Add 1 extra salt dispense
            line_l[cmd_idx] = pattern.sub(f"spice salt dispense {count+1} times", line_l[cmd_idx])
            disp_count_l[i] += 1
            rem_count -= 1
            if rem_count == 0:
                return line_l
        # Add remaining count to last dispense command
        last_disp_cmd_idx = disp_cmd_idx_l[-1]
        new_count = disp_count_l[-1] + rem_count
        line_l[last_disp_cmd_idx] = pattern.sub(f"spice salt dispense {new_count} times", line_l[last_disp_cmd_idx])
        return line_l
    return line_l

def add_onion_specific_cmds(line_l):
    ai_cmd_idx = next(i for i, line in enumerate(line_l) if "cook onion till" in line)
    pattern = re.compile(r"(spice salt dispense (\d+) times)")
    disp_count = 0
    for line in line_l[:ai_cmd_idx]:
        match = pattern.search(line)
        if match:
            disp_count += int(match.group(2))
    if disp_count == 0:
        line_l.insert(ai_cmd_idx, "spice salt dispense 1 times")
    return line_l

# The water table below was calibrated against 100 g of raw rice per serving (the
# RAG corpus's own portions: 400 g at 4 servings, 200 g at 2). Water physically
# follows the rice mass in the pan, not the serving label — if the two ever disagree
# (e.g. an inflated serving count with the true rice amount), trusting the label
# waterlogs or starves the rice.
_RICE_G_PER_SERVING = 100


def add_rice_specific_cmds(line_l, serv_size, rice_quant=None):
    ai_cmd_idx = line_l.index("cook rice till cooked")
    serv_size = _clamp_serv_size(serv_size, "add_rice_specific_cmds")
    water_key = serv_size
    if rice_quant and rice_quant > 0:
        # round half up, clamp to the calibrated 1-4 range
        rice_servings = max(1, min(_MAX_SERV_SIZE, int(rice_quant / _RICE_G_PER_SERVING + 0.5)))
        if rice_servings != serv_size:
            logger.warning(
                f"add_rice_specific_cmds: recipe says {serv_size} serving(s) but the trays "
                f"hold {rice_quant:.0f}g of rice (~{rice_servings} serving(s) at "
                f"{_RICE_G_PER_SERVING}g each); keying water off the rice mass."
            )
        water_key = rice_servings
    new_water_quant = {1: 700, 2: 900, 3: 1100, 4: 1300}[water_key]
    pattern = re.compile(r"(water dispense (\d+) ml)")
    for i in range(ai_cmd_idx, -1, -1):
        if pattern.search(line_l[i]):
            line_l[i] = f"water dispense {new_water_quant} ml"
            break

    for idx, line in enumerate(line_l[ai_cmd_idx:], ai_cmd_idx):
        if line.startswith('vegetable') and line.endswith('dispense'):
            line_l[idx] = line + " no_mix"
    line_l.extend(["exhaust off", "wait 300 seconds"])
    return line_l

######################################################################################
def sanitize_ing_name(name):
    ret_name = ""
    for c in name:
        if c.isalnum():
            ret_name += c
    return ret_name

######################################################################################
def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def sanitize_spice_name(line):
    remap_name_l = ["cumin", "corianderPowder", "garamMasala", "salt",
                    "turmeric", "chilliPowder", "cuminPowder", "mustard"]

    # Parse by grammar, not by token position. Slicing parts[1:-3] assumes the line
    # ends in exactly "dispense N times"; when it doesn't, the name and the tail land
    # in the wrong variables and the closest-match below happily resolves the empty
    # or truncated name to an unrelated spice — e.g. "spice turmeric dispense 1"
    # (no "times") came back out as "spice cumin turmeric dispense 1", silently
    # dispensing cumin instead of turmeric. Anything that doesn't fit the grammar is
    # left untouched for the caller to deal with rather than quietly rewritten.
    m = re.match(r"^(spice)\s+(.+?)\s+(dispense\s+\S+\s+times)\s*$", line.strip(), re.IGNORECASE)
    if not m:
        logger.warning(f"Malformed spice command left unmodified by sanitize_spice_name: {line.strip()!r}")
        return line.strip()
    start, spice_name, end = m.group(1), m.group(2), m.group(3)
    best_match, best_score = None, -1
    for remap_name in remap_name_l:
        score = similarity(spice_name.lower(), remap_name.lower())
        if score > best_score:
            best_match, best_score = remap_name, score

    # Dispense counts must be whole numbers (each dispense = 1/4 tsp). If a
    # fractional count slipped through (e.g. the model used the raw tsp
    # quantity instead of the dispense count), treat it as leftover tsp and
    # convert: N tsp -> round(N * 4) dispenses, minimum 1.
    end_parts = end.split()
    try:
        count = float(end_parts[1])
        if count != int(count):
            end_parts[1] = str(max(1, round(count * 4)))
            end = ' '.join(end_parts)
    except (ValueError, IndexError):
        pass

    return ' '.join([start, best_match, end]).strip()
######################################################################################

def enclose_and_sanitize_spice_blocks(line_l):
    result = []
    i = 0
    while i < len(line_l):
        line = line_l[i].strip()
        if line.startswith('#'):
            result.append(line)
            i += 1
            continue
        if "spice" in line and "dispense" in line:
            result.append("spice position 1 times")
            # Loop through all consecutive spice dispense lines
            while i < len(line_l):
                line = line_l[i].strip()
                if "spice" in line and "dispense" in line:
                    line = sanitize_spice_name(line)
                    result.append(line)
                    i += 1
                else:
                    break
            result.append("spice rest 1 times")
        else: # if regular lines
            if line:
                result.append(line)
            i += 1
    return result
######################################################################################

def generate_mix_wait_block(n_stir_wait, stir_type, inter_t, extra_t, min_wait_time=None):
    block = []
    stir_wait_cmd_block = [f"stir {stir_type} 1 times", f"wait {inter_t} seconds"]
    for _ in range(n_stir_wait):
        block.extend(stir_wait_cmd_block)
    if extra_t > 0:
        last_l = block[-1]
        last_l_wait_t = int(last_l.split()[1])
        if last_l_wait_t + extra_t > 10:
            block[-1] = f"wait {last_l_wait_t + extra_t} seconds"
    return block

_BARE_COOK_PATTERN = re.compile(r"^cook\s+(\d+)\s+seconds$", re.IGNORECASE)
# Grammar for the expandable cook command:
#   cook [total] [seconds|minutes] stir [stir_type] every [x] [seconds|minutes]
# The canonical form uses seconds for the total, but the LLM sometimes emits
# minutes (e.g. "cook 25 minutes stir mix every 2 minutes"). Accept either unit
# in BOTH time slots and normalise to seconds below, instead of rejecting the
# line. Parsed by regex (not positional split) so a slightly-malformed line —
# e.g. a stir-type word landing where a number was expected — degrades
# gracefully instead of throwing `int('saute')` and 500-ing the whole request.
_COOK_STIR_PATTERN = re.compile(
    r"^cook\s+(\d+)\s+(seconds|minutes)\s+stir\s+(\w+)\s+every\s+(\d+)\s+(seconds|minutes)\b",
    re.IGNORECASE,
)

def process_cook_cmd(line_l):
    timeMap = {'saute': 16, 'mix': 30, 'mix_liquid': 45}
    result = []
    for line in line_l:
        bare_match = _BARE_COOK_PATTERN.match(line.strip())
        if bare_match:
            # "cook N seconds" with no stir type/frequency isn't valid Nosh syntax
            # (that requires "stir ... every ..."). Treat it as a plain wait.
            line = f"wait {bare_match.group(1)} seconds"
        if "cook" in line and "stir" in line and "every" in line: # cook [total_seconds] seconds stir [stir_type] every [x] [seconds|minutes]
            m = _COOK_STIR_PATTERN.match(line.strip())
            if not m:
                # Doesn't fit the expandable grammar — don't crash on positional
                # int() parsing. Keep the line as-is so downstream/hardware can
                # reject it, but log it loudly for debugging.
                logger.warning(f"Malformed cook-stir command, passing through unchanged: {line!r}")
                if line:
                    result.append(line)
                continue
            stir_type = m.group(3).lower()
            if stir_type not in timeMap:
                # Unknown stir type would KeyError on timeMap below.
                logger.warning(f"Unknown stir type {stir_type!r} in cook command, passing through: {line!r}")
                if line:
                    result.append(line)
                continue
            # Normalise both time slots to seconds (either may be given in minutes).
            cook_t = int(m.group(1)) * (60 if m.group(2).lower() == "minutes" else 1)  # total
            inter_t = int(m.group(4)) * (60 if m.group(5).lower() == "minutes" else 1)  # [x]
            stir_t = timeMap[stir_type]
            stir_wait_t = stir_t + inter_t
            n_stir_wait = cook_t // stir_wait_t
            extra_t = cook_t - (n_stir_wait * stir_wait_t)
            if n_stir_wait:
                cmd_block = generate_mix_wait_block(n_stir_wait, stir_type, inter_t, extra_t)
            else:
                # Handles cases like: `cook 30 seconds stir mix every 30 seconds`
                # If stir-wait can't be done within the given cook time, do 1 stir and wait
                # for the remaining time (if it is >= 10s).
                rem_t = cook_t - stir_t
                if rem_t >= 10:
                    cmd_block = [f"stir {stir_type} 1 times", f"wait {cook_t - stir_t} seconds"]
                else:
                    cmd_block = [f"stir {stir_type} 1 times"]

            if result:
                if "stir" in result[-1]:
                    # Merge consecutive `stir` commands, if any
                    line_a_p, line_b_p = result[-1].split(), cmd_block[0].split()
                    if line_a_p[1] == line_b_p[1]: # if stir type is same
                        stir_type = line_a_p[1]
                        stir_count = int(line_a_p[2]) + int(line_b_p[2])
                        del cmd_block[0]
                        result[-1] = f"stir {stir_type} {stir_count} times"
                elif "wait" in line:
                    # Merge consecutive `wait` commands, if any
                    line_a_p, line_b_p = result[-1].split(), line.split()
                    if line_a_p[0] == line_b_p[0]:
                        wait_t = int(line_a_p[1]) + int(line_b_p[1])
                        result[-1] = f"wait {wait_t} seconds"
                        continue
            result.extend(cmd_block)
        else:
            if line:
                result.append(line)
    return result
######################################################################################

def get_int_stove_heat(stove_heat):
    stove_heat = stove_heat.lower()
    if stove_heat.startswith('low'):
        return 'heat 1'
    elif stove_heat.startswith('med') or stove_heat.startswith('mid'):
        return 'heat 3'
    elif stove_heat.startswith('high'):
        return 'heat 4'
    elif stove_heat == 'very_high':
        return 'heat 6'
    elif stove_heat == 'off' or stove_heat == "stop":
        return 'stop'
    else:
        raise ValueError(f"Incorrect stove heat {stove_heat}")

def process_oil_water_disp_cmd(x, liquid=None):
    assert x > 0 and liquid in ['oil', 'water']
    base = 5 if liquid == "oil" else 10
    divisor, rem = x // base, x % base
    if rem:
        if rem >= base/2:
            return divisor*base + base
        return divisor*base
    return x
######################################################################################


def _fmt_qty(quant):
    """Render a tray quantity as a whole number for the user filling the trays.

    Display-only: the exact float is still used upstream for tray-weight limits and
    marinade mass-splitting, so we never round there. A positive sub-gram amount
    (e.g. 0.39 g of cardamom) must show as 1, never 0 — rounding a real ingredient
    down to 0 would silently drop it from the tray. Non-numeric quantities pass
    through untouched.
    """
    try:
        q = float(quant)
    except (TypeError, ValueError):
        return quant
    if q <= 0:
        return "0"
    return str(max(1, int(q + 0.5)))  # round half up; floor positive values to 1


@func_timing_decorator
@trace_function("run_cmd_processor")
def run_cmd_processor(info_obj) -> str:
    logger.info("Post-processing generated recipe commands.")

    # Add processing details to span
    recipe_j = info_obj.recipe.model_dump()
    assert isinstance(recipe_j, dict), f"recipe_j is of type: {type(recipe_j)}"

    serv_size, consistency = recipe_j['serving'], recipe_j['consistency']

    # vegmap
    vegmap_ing_d = {}
    temp_name_l = []
    slot_cmd_l_map = {}
    slot_wise_ings_d = {}
    rice_quant = None
    for slot in recipe_j["slots"]:
        assert isinstance(slot, dict)
        slot_id = slot['number']
        assert slot_id not in vegmap_ing_d, f"{slot_id} is alread there in {vegmap_ing_d}"
        ing_l = slot['ingredients']
        if len(ing_l) == 0:
            continue
        name = sanitize_ing_name(ing_l[0]['ingredient_name'])
        if name in temp_name_l:
            for i in range(5):
                new_name = name + str(i)
                if new_name not in temp_name_l:
                    name = new_name
                    break
            else:
                assert True, f"Slot id exceeds 5. Check it {slot_id} {vegmap_ing_d}"
        vegmap_ing_d[slot_id] = name
        slot_cmd_l_map[slot_id] = []
        slot_ing_l = []
        for ing_d in ing_l:
            ing_name, quant, unit, prepStep = ing_d['ingredient_name'], ing_d['quantity'], ing_d['unit'], ing_d['preparation_step']
            ing_name = ing_name.strip().lower()
            if "rice" in ing_name and ing_name not in ['rice flour', 'puffed rice', 'cooked rice']: # temporary fix TODO
                # SUM, not assign: rice is routinely split across trays by the 400g
                # effective-weight cap (two 150g trays), and keeping only the last
                # entry would halve the water calculation downstream.
                rice_quant = (rice_quant or 0.0) + float(quant)
            # Render empty/"None" prep as blank so the slot line stays clean.
            prep_disp = "" if prepStep is None or str(prepStep).strip().lower() in ("", "none") else str(prepStep).strip()
            slot_cmd_l_map[slot_id].append(f"# \t{ing_name} | {_fmt_qty(quant)} {unit} | {prep_disp}")
            slot_ing_l.append((ing_name, quant))
        slot_wise_ings_d[slot_id] = slot_ing_l

    # order
    vegmap_order_l = []
    line_l = recipe_j['commands']
    for line_s in line_l:
        line_s.strip()
        if line_s.startswith("ingredient_tray "):
            assert line_s.endswith(" dispense")
            slot_id = int(line_s.split(" ")[1])
            if (slot_id not in vegmap_ing_d) and any(slot['number'] == slot_id and len(slot['ingredients']) == 0 for slot in recipe_j['slots']):
                continue
            else:
                assert True, f"slot {slot_id} not found in {vegmap_ing_d}"
            vegmap_order_l.append(slot_id)


    ret_line_l = [
        f"#recipe name - {recipe_j['recipe_name']}",
        f"#course - {recipe_j.get('course') or 'Main'}",
        f"#consistency - {getattr(consistency, 'value', consistency)}",
        f"#servings - {recipe_j['serving']}",
    ]
    # Shared user-prep block (marinades, pastes/purées, pressure-cook groups).
    # De-duplicate case-insensitively while preserving order — a deterministic
    # backstop so the same step never prints twice even if the model repeats it.
    prep_instructions = recipe_j.get('prep_instructions') or []
    seen_prep = set()
    deduped_prep = []
    for step in prep_instructions:
        step_s = str(step).strip()
        key = step_s.lower()
        if step_s and key not in seen_prep:
            seen_prep.add(key)
            deduped_prep.append(step_s)
    if deduped_prep:
        ret_line_l.append("# Preparation")
        for idx, step_s in enumerate(deduped_prep, 1):
            ret_line_l.append(f"# \t{idx}. {step_s}")
    # Handles cases when trays are not dispensed in order. Eg: vegmap_order_l == [1, 2, 5, 3, 4]
    for i, slot_id in enumerate(vegmap_order_l):
        ret_line_l.extend([f"# Slot {i+1}"] + slot_cmd_l_map[slot_id])

    # Deduplicate tray ids
    ids = {}
    for id_, name in vegmap_ing_d.items():
        if name in ids:
            vegmap_ing_d[id_] = name + str(ids[name]+1)
        ids[name] = ids.get(name, 0) + 1
    vegmap_s = "vegmap " + ",".join(f"{vegmap_ing_d[vegmap_order_l[idx]]}={idx+1}" for idx in range(len(vegmap_order_l)) )
    ret_line_l.append(vegmap_s)

    rice_ai_used, onion_ai_used = False, False
    first_heat = True
    current_heat = 3
    dispensed_ings = []
    # post process commands
    for idx, line_s in enumerate(line_l):
        line_s = line_s.strip()
        if line_s == '```' or line_s == "":
            continue
        elif line_s.startswith("ingredient_tray ") and line_s.endswith(" dispense"):
            slot_id = int(line_s.split(" ")[1])
            if slot_id in vegmap_ing_d:
                ret_line_l.append("vegetable " + vegmap_ing_d[slot_id] + " dispense")
            dispensed_ings.extend(slot_wise_ings_d[slot_id])
            continue
        elif line_s.startswith("heat_till "):
            temp = int(line_s.split(" ")[1])
            timeout = 120 if temp > 75 else 60
            extend = [f"set_temperature_timeout {timeout} seconds", line_s]
            if first_heat:
                extend.append("stove heat 3")
                current_heat = 3
                first_heat = False
            ret_line_l.extend(extend)
            continue
        elif line_s.startswith("stove level"):
            parts = line_s.split()
            heat_cmd = get_int_stove_heat(parts[-1].lower())
            current_heat = int(heat_cmd.split()[-1]) if heat_cmd.startswith("heat") else current_heat
            ret_line_l.append('stove ' + heat_cmd)
            continue
        elif line_s.startswith("oil") or line_s.startswith("water"):
            parts = line_s.split()
            quant = int(process_oil_water_disp_cmd(float(parts[2]), parts[0]))
            if line_s.startswith("water") and quant > 250:
                # Large water dispense: high heat to bring to boil, mix_liquid to avoid spillover,
                # then return to a simmer once at temperature.
                ret_line_l.append("stove heat 8")
                ret_line_l.append(f'{parts[0]} dispense {quant} ml')
                ret_line_l.append("stir mix_liquid 1 times")
                ret_line_l.extend(["set_temperature_timeout 120 seconds", "heat_till 65", f"stove heat {current_heat}"])
            else:
                ret_line_l.append(f'{parts[0]} dispense {quant} ml')
            continue
        elif line_s.endswith("till cooked"): # if ai command
            parts = line_s.split()
            start, ai_ing, end = parts[0], parts[1].lower(), ' '.join(parts[2:])
            if ai_ing == 'rice':
                rice_ai_used = True
            elif ai_ing == 'onion':
                onion_ai_used = True
            if ai_ing == 'onion':
                end = 'till golden_brown'
            inst = ' '.join([start, ai_ing, end]).strip()
            ret_line_l.append(inst)
            if ai_ing in ['onion', 'tomato']:
                ret_line_l.extend(["water dispense 50 ml", "stir mix 1 times"]) # post ai commands for 'onion' and 'tomato'
            continue
        ret_line_l.append(line_s)

    if onion_ai_used:
        ret_line_l = add_onion_specific_cmds(ret_line_l)
    if rice_ai_used:
        ret_line_l = add_rice_specific_cmds(ret_line_l, serv_size, rice_quant)
    ret_line_l = enclose_and_sanitize_spice_blocks(ret_line_l)
    ret_line_l = fix_salt_dispense(ret_line_l, consistency, serv_size)
    ret_line_l = process_cook_cmd(ret_line_l)
    ret_str = "\n".join(ret_line_l)
    ret_str = f"{ret_str.strip()}\n\n[{pc_s.strip()}]" if (pc_s := info_obj.post_cooking_step) else ret_str

    add_span_attribute("output", ret_str)
    logger.info("Post-processing generated recipe commands done successfully.")
    logger.debug(f"nosh_recipe_cmds: \n{ret_str}")
    return ret_str

if __name__ == "__main__":
    # recipe_name = "Something"
    # info_f = "../eval_dataset/old_output/101-VT-INDN-MAIN/101-VT-INDN-MAIN-1-1.json"
    # cmd_f = "../eval_dataset/old_output/101-VT-INDN-MAIN/101-VT-INDN-MAIN-1-1.recipe"
    # with open(info_f) as f:
    #     info_d = json.load(f)
    # with open(cmd_f) as f:
    #     cmd_s = f.read()
    # print(run_cmd_processor(recipe_name, info_d, cmd_s))

    e_j_l = Path("../eval_dataset/output").glob("*/*_extracted.json")
    for e_j_fp in e_j_l:
        print("Running on", e_j_fp)
        name = e_j_fp.name.split("_")[0]
        i_fp = e_j_fp.parent / f"{name}_info.json"
        g_c_fp = e_j_fp.parent / f"{name}_gen.recipe"
        p_c_fp = e_j_fp.parent / f"{name}_prc.recipe"
        with open(e_j_fp, 'r') as f:
            e_j = json.load(f)
        recipe_name = e_j['DishName']
        with open(i_fp, 'r') as f:
            i_j = json.load(f)
        with open(g_c_fp, 'r') as f:
            g_c_s = f.read()
        p_c_s = run_cmd_processor(recipe_name, i_j, g_c_s)
        with open(p_c_fp, 'w') as f:
            f.write(p_c_s)