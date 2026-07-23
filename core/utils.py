import os
import time
from functools import wraps
import logging
from logging.handlers import RotatingFileHandler
import io
import sys

from dotenv import load_dotenv


load_dotenv()

# Ensure UTF-8 for console output
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')

DEBUG_MODE = os.getenv("DEBUG", "False").lower() == "true"
logger = logging.getLogger("my_logger")
logger.setLevel(logging.DEBUG if DEBUG_MODE else logging.INFO)
log_format = logging.Formatter("%(asctime)s - %(levelname)s - [%(request_id)s] - %(message)s")
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_format)
log_file = "app.log"
max_file_size = 10_000 * 100  # Approximate size for 10,000 lines (assuming ~100 bytes per line)
file_handler = RotatingFileHandler(log_file, maxBytes=max_file_size, backupCount=3, encoding='utf-8', delay=True)
file_handler.setFormatter(log_format)


class _RequestIDFilter(logging.Filter):
    """Injects the current Flask request's ID into every log record, so one
    request's full call chain can be grepped out of app.log. Falls back to
    '-' outside a request context (startup logs, background/CLI scripts)."""

    def filter(self, record):
        from flask import g, has_request_context
        record.request_id = g.request_id if has_request_context() and hasattr(g, "request_id") else "-"
        return True


_request_id_filter = _RequestIDFilter()
console_handler.addFilter(_request_id_filter)
file_handler.addFilter(_request_id_filter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)


def func_timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        logger.info(f"{func.__name__} executed in {end_time - start_time:.6f} seconds")
        return result
    return wrapper
