"""
OpenTelemetry tracing utilities for recipe processing.
"""
import os
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from openinference.semconv.trace import SpanAttributes
from openinference.instrumentation import using_session
from opentelemetry.instrumentation.flask import FlaskInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from phoenix.otel import register
from functools import wraps
from typing import Optional, Dict, Any
import logging
import traceback
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Initialize tracer
tracer = trace.get_tracer(__name__)

def setup_tracing():
    """Setup OpenTelemetry tracing with Arize Phoenix."""
    try:
        # Configure the Phoenix tracer
        tracer_provider = register(
            project_name="nosh-recipe-gen", auto_instrument=True,
        )

        # Initialize Flask, requests, GoogleGenAISDK and Agno instrumentation
        FlaskInstrumentor().instrument()
        RequestsInstrumentor().instrument()

        logger.info("OpenTelemetry tracing setup completed successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to setup OpenTelemetry tracing: {e}")
        return False

def trace_function(
    operation_name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
    capture_exceptions: bool = True,
    session_id: Optional[str] = "01" # TODO: Make it env. variable
):
    """
    Decorator to trace function execution with OpenTelemetry spans.

    Args:
        operation_name: Name of the operation for the span
        attributes: Additional attributes to add to the span
        capture_exceptions: Whether to capture exceptions in the span
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            span_name = operation_name or func.__name__
            with tracer.start_as_current_span(span_name) as span:
                try:
                    # Add custom attributes
                    if attributes:
                        for key, value in attributes.items():
                            span.set_attribute(key, value)

                    if kwargs:
                        span.set_attribute("function.kwargs_count", len(kwargs))

                    # Handle session propagation if session_id provided
                    if session_id:
                        span.set_attribute(SpanAttributes.SESSION_ID, session_id)
                        with using_session(session_id):
                            result = func(*args, **kwargs)
                    else:
                        result = func(*args, **kwargs)

                    span.set_status(Status(StatusCode.OK))
                    return result

                except Exception as e:
                    if capture_exceptions:
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        logger.error(f"Exception in traced function {func.__name__}: {e}")
                    raise

        return wrapper
    return decorator

def add_span_attribute(key: str, value: Any):
    """Add an attribute to the current active span."""
    allowed_types = [bool, str, bytes, int, float]

    current_span = trace.get_current_span()
    if current_span.is_recording():
        if isinstance(value, dict):
            value = {
                k: str(v) if (v is None) or (type(v) not in allowed_types) else v
                for k, v in value.items()
            }
            for k, v in value.items():
                current_span.set_attribute(f'{key}.{k}', v)
        elif isinstance(value, list) or isinstance(list, tuple):
            current_span.set_attribute(key, str(value))
        elif isinstance(value, BaseException):
            current_span.set_attribute(f"{key}.type", type(value).__name__)
            current_span.set_attribute(f"{key}.message", str(value))
            current_span.set_attribute(f"{key}.traceback", ''.join(traceback.format_exception(type(value), value, value.__traceback__)))
        else:
            current_span.set_attribute(key, value)
