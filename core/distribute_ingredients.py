"""Core recipe/distribution schema shared by the orchestrator and cmd_processor.

The ingredient-distribution LLM call that used to live here was folded into
core.orchestrator, which now owns the authoritative Nosh prompt (tray limits,
distribution rules, composite handling). Only the schema remains.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class Consistency(Enum):
   DRY = "dry"
   SEMI_GRAVY = "semi gravy"
   GRAVY = "gravy"
   WHOLE_MEAL = "whole meal"

class Ingredients(BaseModel):
    ingredient_name: str = Field(description="ingredients's root name.")
    quantity: float = Field(description="quantity of the ingredient.")
    unit: Optional[str] = Field(description="quantity's unit of measurement.")
    preparation_step: str | None = Field(description="preparation or precooking steps for the ingredient before it can be used in the recipe.")

class Slot(BaseModel):
    number: int = Field(description="tray number.")
    ingredients: list[Ingredients] = Field(description="list of ingredients assigned to this tray, with their details.")

class Recipe(BaseModel):
    recipe_name: str = Field(description="name of the recipe.")
    serving: int = Field(description="serving size specified in the recipe, or an approximation based on ingredient quantities if not provided.")
    consistency: Consistency = Field(description="final dish consistency (dry, semi gravy, gravy, or whole meal).")
    slots: list[Slot] = Field(description="ordered list of ingredient groupings, tray-wise.")
    updated_instructions: str = Field(description="updated cooking instructions incorporating tray dispenses while preserving all other steps and recipe flow.")

class Distribution(BaseModel):
   is_recipe: bool = Field(description="false if input is not a singular recipe.")
   nosh_compatible: bool = Field(description="false if the recipe cook steps cannot be executed within Nosh's capabilities and limitations.")
   reason: str | None = Field(description="reason as to why input is not a singular recipe or not compatible with Nosh.")
   recipe: Recipe | None = Field(description="structured details of the recipe.")
