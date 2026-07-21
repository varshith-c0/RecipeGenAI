"""Extended recipe/distribution models consumed by the orchestrator.

The command-generation LLM call that used to live here was folded into
core.orchestrator; only these schema extensions remain.
"""

from core.distribute_ingredients import Recipe, Distribution


class RecipeExtended(Recipe):
    commands: list | None = None
    course: str | None = None
    cuisine: str | None = None
    dish_type: str | None = None
    prep_instructions: list | None = None

class DistributionExtended(Distribution):
    recipe: RecipeExtended | None = None
    post_cooking_step: str | None = None
