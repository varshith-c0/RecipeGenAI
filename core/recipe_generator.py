"""
Recipe generator — optimized agentic architecture.

Public interface unchanged:
    run_recipe_generator(recipe_cnt, is_ing_check_enabled, is_instr_check_enabled)
        -> (bool, DistributionExtended | dict)
"""

from google.genai.errors import ServerError

from core.orchestrator import run_orchestrator, to_distribution_extended
from core.tracing import trace_function, add_span_attribute
from .utils import logger, func_timing_decorator


@func_timing_decorator
@trace_function("run_recipe_generator")
def run_recipe_generator(
    recipe_cnt: str,
    is_ing_check_enabled: bool,
    is_instr_check_enabled: bool,
    target_serving: int | None = None,
):
    logger.info(
        f"Starting recipe generation (optimized agentic). "
        f"is_ing_check_enabled={is_ing_check_enabled}, "
        f"is_instr_check_enabled={is_instr_check_enabled}, "
        f"target_serving={target_serving}"
    )
    add_span_attribute("inputs.is_ing_check_enabled", is_ing_check_enabled)
    add_span_attribute("inputs.is_instr_check_enabled", is_instr_check_enabled)
    add_span_attribute("inputs.target_serving", str(target_serving))

    output, reason, error = run_orchestrator(
        recipe_cnt, is_ing_check_enabled, is_instr_check_enabled,
        target_serving=target_serving,
    )

    if error is not None:
        add_span_attribute("failure.has_error", True)
        add_span_attribute("failure.error_message", str(error))
        msg = (
            "Model overloaded. Please try again after some time."
            if isinstance(error, ServerError)
            else "System error."
        )
        return False, {"reason": msg}

    if output is None:
        # Recoverable failures (recipe too large for the trays) arrive as a
        # dict carrying error_code / max_feasible_serving for the retry flow;
        # everything else is a plain string.
        if isinstance(reason, dict):
            add_span_attribute("failure.failure_reason", str(reason.get("reason")))
            return False, reason
        msg = reason or "Recipe could not be processed."
        add_span_attribute("failure.failure_reason", msg)
        return False, {"reason": msg}

    result_obj = to_distribution_extended(output)
    return True, result_obj
