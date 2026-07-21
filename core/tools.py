import re
import json
from langsmith import unit
import nltk
import pandas as pd
from typing import List
from difflib import get_close_matches
from nltk.stem import WordNetLemmatizer
from .utils import logger

# Load ingredients file
ing_df = pd.read_excel(r"./resources/ingredients_cleaned.xlsx")

# Load ingredient-to-excel-row mapping
ing2row_j_fp = r"./resources/mapping.json"
with open(ing2row_j_fp, 'r') as map_f:
    mapping = json.load(map_f)

# Load wordnet
nltk.data.path.append(r"./resources/nltk_data")

# NOTE:
# This function serves a dual purpose — it can be invoked as a standalone Python function
# OR registered as an external "tool" for an LLM.
# The docstring and type hints are written specifically for the tool interface so the LLM
# can infer correct argument types and usage.
# When used as a normal Python function, the runtime behavior differs slightly, but the
# docstring and type annotations are left unchanged to ensure compatibility with the tool spec.
def _convert_quantity_to_grams(
        ing_name_l: List[str],
        quant_l: List[float | None],
        unit_l: List[str | None],
    ) -> str:
    """
    Convert ingredient quantities into grams and return a conversion report.

    Args:
        ing_name_l (List[str]): List of ingredient names.
        quant_l (List[float | None]): List of quantity of the ingredients in absolute terms.
        unit_l (List[str | None]): List of unit of the ingredients' quantity.
        

    Returns:
        str: A newline-separated report. Each line states whether a quantity was
        converted to grams and shows the resulting value (in grams if converted,
        otherwise in the original unit).
    """
    assert len(ing_name_l) == len(quant_l) == len(unit_l)

    results = []
    lemmatizer = WordNetLemmatizer()
    spice_l = ['cumin', 'mustard', 'cumin powder', 'chili powder', 'salt', 'turmeric', 'garam masala', 'coriander powder']

    for ing_name, quant, unit in zip(ing_name_l, quant_l, unit_l):
        ing_type = None
        is_converted = False
        ing_name = lemmatizer.lemmatize(ing_name.strip().lower())
        unit = unit.strip().lower() if unit else None

        if quant is not None:
            if unit and unit.startswith('g'):
                results.append((ing_name, quant, unit, False, None))
                continue

            # Get the closest matching ingredient name. If a close match is found, use it;
            # otherwise, fall back to the original name.
            possibilities = list(mapping.keys())
            match = get_close_matches(ing_name, possibilities, n=1, cutoff=0.90)
            lookup_name = match[0] if match else None
            # A whole-name hit is the same ingredient spelled differently, so adopt the
            # canonical name ('green chili' -> 'green chilli'); downstream spice matching
            # relies on that. A suffix hit is only a weight donor (see below), so the
            # caller's name must survive it.
            borrowed_weights = False
            if not match:
                # Regional and qualified names ("seeraga samba rice", "kashmiri red
                # chilli") name the ingredient in the trailing head noun, and are far
                # enough from it as whole strings that the strict cutoff rejects them —
                # leaving the quantity unconverted, which the caller cannot dispense.
                # Retry on progressively shorter suffixes to borrow a plausible
                # weight-per-unit, but keep the original name: the donor row is close
                # enough to weigh by, not necessarily the same ingredient.
                tokens = ing_name.split()
                for i in range(1, len(tokens)):
                    m = get_close_matches(" ".join(tokens[i:]), possibilities, n=1, cutoff=0.90)
                    if m:
                        lookup_name = m[0]
                        borrowed_weights = True
                        logger.debug(f"{ing_name} ~> {lookup_name} (borrowed weights only)")
                        break
            if lookup_name and not borrowed_weights:
                logger.debug(f"{ing_name} -> {lookup_name}")
                ing_name = lookup_name

            # Get ingredient's data
            row_id = mapping.get(lookup_name or ing_name, -1)
            row = ing_df[ing_df['id'] == row_id]

            if not row.empty:
                if not borrowed_weights:
                    ing_name = row['name'].iloc[0].strip().lower()
                get_value = lambda row, col: float(row[col].fillna(-1).iloc[0])
                g_per_scale = -1

                if unit is None or unit.startswith(('count', 'no', 'number', 'unit')):
                    g_per_scale = get_value(row, 'unit')
                elif unit.startswith('cup'):
                    g_per_scale = get_value(row, 'cup')
                elif unit.startswith('cube'):
                    g_per_scale = get_value(row, 'cube')
                elif unit.startswith('ml'):
                    g_per_scale = get_value(row, 'ml')
                elif unit.startswith(('tbsp', 'tablespoon')):
                    g_per_scale = get_value(row, 'tbsp')
                elif unit.startswith(('tsp', 'teaspoon')):
                    g_per_scale = get_value(row, 'tsp')
                elif unit.startswith('clove'):
                    g_per_scale = get_value(row, 'clove')
                elif unit.startswith('pinch'):
                    g_per_scale = get_value(row, 'pinch')

                if g_per_scale > 0:
                    quant, unit, is_converted = quant*g_per_scale, 'g', True

                if ing_name in spice_l and is_converted: # if ingredient is a Nosh's dispense mechanism supported spice, get quant it 'tsp'
                    g_per_tsp = get_value(row, 'tsp')
                    if g_per_tsp > 0:
                        quant, unit = quant/g_per_tsp, 'tsp'
                    else:
                        is_converted = False
                elif (ing_name=='water' or re.search(r'\boil\b', ing_name)) and is_converted: # for water and oil, get quant in 'ml'
                    g_per_ml = get_value(row, 'ml')
                    if g_per_ml > 0:
                        quant, unit = quant/g_per_ml, 'ml'
                    else:
                        is_converted = False

                # Get ingredient type
                val = row['ingredientType'].iloc[0]
                ing_type = val.strip().lower() if pd.notna(val) else None

        logger.info(f"ing_name={ing_name}, quant={quant}, unit={unit}, is_converted={is_converted}, ing_type={ing_type}")
        results.append((ing_name, quant, unit, is_converted, ing_type))

    return results


# Reference per-serving weight (grams) for common anchor ingredients — used to
# deterministically estimate serving size when a recipe doesn't state one, instead
# of leaving it to free-form LLM guessing. Checked in priority order: protein
# anchors first, then staple carbs, since protein quantity is usually the more
# reliable signal of headcount in Indian home cooking.
_ANCHOR_SERVING_GRAMS = [
    ('paneer', 75),
    ('chicken', 150),
    ('mutton', 150),
    ('fish', 150),
    ('prawns', 120),
    ('egg', 60),
    ('rice', 75),
    ('toor dal', 40),
    ('moong dal', 40),
    ('chana dal', 40),
    ('masoor dal', 40),
    ('urad dal', 40),
    ('potato', 100),
    ('pasta', 85),
]

def _estimate_serving_size(
        ing_name_l: List[str],
        quant_l: List[float | None],
        unit_l: List[str | None],
    ) -> str:
    """
    Estimate serving size from a recognized anchor ingredient's quantity (e.g.
    paneer, chicken, rice, dal), converted to grams. Intended for use only when
    the recipe text does not explicitly state a serving count.

    Args:
        ing_name_l (List[str]): List of ingredient names.
        quant_l (List[float | None]): List of quantity of the ingredients in absolute terms.
        unit_l (List[str | None]): List of unit of the ingredients' quantity.

    Returns:
        str: Either the estimated serving size derived from a recognized anchor
        ingredient, or a note that no anchor ingredient was found.
    """
    results = _convert_quantity_to_grams(ing_name_l, quant_l, unit_l)
    converted = {
        ing_name: quant for ing_name, quant, unit, is_converted, _ in results
        if unit == 'g' and quant is not None
    }

    for anchor_name, per_serving_g in _ANCHOR_SERVING_GRAMS:
        for ing_name, quant in converted.items():
            if anchor_name in ing_name and not any(x in ing_name for x in ('flour', 'starch')):
                est = max(1, min(4, round(quant / per_serving_g)))
                return (
                    f"Anchor ingredient found: '{ing_name}' ({round(quant, 1)}g). "
                    f"Reference: ~{per_serving_g}g per serving. "
                    f"Estimated serving size = {est}. Use this value unless the "
                    f"recipe text explicitly states a different serving count."
                )

    return (
        "No recognized anchor ingredient (paneer/chicken/mutton/fish/rice/dal/"
        "potato/pasta) found — estimate serving size from overall recipe context "
        "and typical portion sizes instead."
    )


def _get_fallback_instructions(ing_name_l: List[str]) -> str:
    """
    Look up hardware-verified, pre-tested Nosh cooking instructions for known
    ingredients (from fallback_instr.py). These sequences were validated on real
    Nosh hardware and should override free-form timing estimates for the
    ingredients they cover.

    Args:
        ing_name_l (List[str]): List of ingredient names to check.

    Returns:
        str: For each ingredient, either its verified instruction sequence or a
        note that no verified sequence exists (fall back to RAG/general estimate).
    """
    from fallback_instr import fallback_instr_mapping

    lemmatizer = WordNetLemmatizer()
    lines = []
    for ing_name in ing_name_l:
        key = lemmatizer.lemmatize(ing_name.strip().lower())
        if key in fallback_instr_mapping:
            instr_s = ' '.join(fallback_instr_mapping[key]).strip()
            lines.append(f"{ing_name} - VERIFIED sequence (use verbatim): {instr_s}")
        else:
            lines.append(f"{ing_name} - no verified sequence found; estimate from RAG/context.")
    return "\n".join(lines) if lines else "No ingredients provided."

