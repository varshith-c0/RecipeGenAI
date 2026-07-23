from core.recipe_generator import run_recipe_generator
from core.cmd_processor import run_cmd_processor
from core.tracing import trace_function, add_span_attribute

@trace_function("generate_recipe")
def generate_recipe(recipe_cnt, is_ing_check_enabled, is_instr_check_enabled, target_serving=None):
    is_valid, info_obj = run_recipe_generator(
        recipe_cnt, is_ing_check_enabled, is_instr_check_enabled, target_serving=target_serving
    )

    if not is_valid:
        add_span_attribute("failure.reason", info_obj['reason'])
        # Return the whole dict — recoverable failures carry error_code /
        # max_feasible_serving so the API can offer a retry-with-servings flow.
        return False, info_obj

    p_cmd_str = run_cmd_processor(info_obj)

    if p_cmd_str is None:
        add_span_attribute("failure.reason", "Command processing failed.")
        return False, "Command processing failed"

    return True, p_cmd_str

if __name__ == "__main__":
    example_input = """
    **Aloo Payaz Ki Sabji**
    **Ingredients:**
    * 375 grams potatoes, cubed
    * 8 curry leaves
    * 225 grams onions, cubed
    * Mustard seeds
    * Cumin seeds
    * Salt
    * Red chili powder
    * Coriander powder
    * Turmeric powder
    * Oil (45 ml)
    * Water (250 ml)
    **Instructions:**
    1. Start the stove and heat the oil to 75°C for 30 seconds. Then continue heating the oil until it reaches 105°C for another 30 seconds.
    2. Add 2 pinches of mustard seeds, followed by 3 pinches of cumin seeds. Let them splutter for a short while.
    3. Increase heat to medium. Add the first group of potatoes (375 grams of potatoes, cubed).
    4. Add 4 pinches of salt. Stir well and cook for 5 minutes, stirring occasionally.
    5. Reduce the heat to medium-low. Add the second group of ingredients: 225 grams of onions, cubed.
    6. Stir occasionally and cook for another 5 minutes.
    7. Increase heat to medium. Add 1 pinch of salt, 4 pinches of red chili powder, 3 pinches of coriander powder, and 1 pinch of turmeric powder. Stir well.
    8. Cook for 30 seconds and then add 250 ml of water.
    9. Stir occasionally and cook for another 2.5 minutes.
    10. Turn off the stove. Stir and serve.
    """

    out = generate_recipe(example_input)
    print(out)