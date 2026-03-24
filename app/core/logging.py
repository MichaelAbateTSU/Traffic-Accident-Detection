"""
app/core/logging.py — Centralised logging configuration.

Call configure_logging() once at application startup (in main.py lifespan).
All other modules should obtain loggers via:

    import logging
    logger = logging.getLogger(__name__)

The root logger is configured to emit JSON-friendly structured records when
running in production, and colourised human-readable output in development.
"""

from __future__ import annotations

import logging
import sys
from typing import Any


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def configure_logging(level: str = "INFO") -> None:
    """
    Configure the root logger.

    Parameters
    ----------
    level : str
        A standard Python logging level name, e.g. "DEBUG", "INFO", "WARNING".
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_ServiceFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(numeric_level)

    # Quiet noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("ultralytics").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Convenience wrapper — equivalent to logging.getLogger(name)."""
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Log context helpers (attach per-request / per-job metadata)
# ---------------------------------------------------------------------------

class _LogContext:
    """
    Thread-local context dict that gets merged into every log record emitted
    within the current request/job scope.

    Usage (in route handlers):
        log_context.set(job_id="abc123", stream_url="https://...")
        logger.info("Job started")   # → includes job_id & stream_url
        log_context.clear()
    """

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    def set(self, **kwargs: Any) -> None:
        self._data.update(kwargs)

    def clear(self) -> None:
        self._data.clear()

    def as_dict(self) -> dict[str, Any]:
        return dict(self._data)


log_context = _LogContext()


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------

class _ServiceFormatter(logging.Formatter):
    """
    Simple formatter that prepends level, logger name, and any log_context
    fields to each log line.

    Output example:
        INFO     app.services.detection_service | job_id=abc123 | Job started
    """

    _LEVEL_COLOURS = {
        "DEBUG":    "\033[36m",    # cyan
        "INFO":     "\033[32m",    # green
        "WARNING":  "\033[33m",    # yellow
        "ERROR":    "\033[31m",    # red
        "CRITICAL": "\033[35m",    # magenta
    }
    _RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        colour = self._LEVEL_COLOURS.get(record.levelname, "")
        reset  = self._RESET

        ctx = log_context.as_dict()
        ctx_str = "  ".join(f"{k}={v}" for k, v in ctx.items())
        separator = " | " if ctx_str else ""

        msg = record.getMessage()
        if record.exc_info:
            msg = f"{msg}\n{self.formatException(record.exc_info)}"

        return (
            f"{colour}{record.levelname:<8}{reset} "
            f"{record.name} | "
            f"{ctx_str}{separator}{msg}"
        )
