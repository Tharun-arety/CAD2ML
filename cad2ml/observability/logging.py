"""Structured JSON logging with correlation context.

Every record carries: timestamp, level, event, correlation_id, job_id, sample_id,
pipeline_stage, duration_ms, result, error_code (null when not applicable).
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import structlog
from structlog.contextvars import bind_contextvars, bound_contextvars, clear_contextvars

_STANDARD_FIELDS = (
    "correlation_id",
    "job_id",
    "sample_id",
    "pipeline_stage",
    "duration_ms",
    "result",
    "error_code",
)
_configured = False


def _ensure_fields(_: Any, __: str, event_dict: Any) -> Any:
    for f in _STANDARD_FIELDS:
        event_dict.setdefault(f, None)
    return event_dict


def configure_logging(level: str = "INFO", json_output: bool = True) -> None:
    global _configured
    logging.basicConfig(format="%(message)s", stream=sys.stderr, level=getattr(logging, level.upper()))
    renderer: Any = structlog.processors.JSONRenderer() if json_output else structlog.dev.ConsoleRenderer()
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True, key="timestamp"),
            _ensure_fields,
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level.upper())),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=False,
    )
    _configured = True


def get_logger(name: str = "cad2ml") -> Any:
    if not _configured:
        configure_logging()
    return structlog.get_logger(name)


def bind(**kwargs: Any) -> None:
    bind_contextvars(**kwargs)


def clear() -> None:
    clear_contextvars()


@contextmanager
def stage(name: str, timings: dict[str, float] | None = None, **ctx: Any) -> Iterator[None]:
    """Log a pipeline stage with duration and result; record duration into ``timings``."""
    log = get_logger()
    t0 = time.perf_counter()
    with bound_contextvars(pipeline_stage=name, **ctx):
        try:
            yield
        except Exception as exc:
            dt = (time.perf_counter() - t0) * 1000
            log.warning(
                "stage_failed",
                duration_ms=round(dt, 2),
                result="error",
                error_code=getattr(exc, "code", type(exc).__name__),
            )
            raise
        dt = (time.perf_counter() - t0) * 1000
        if timings is not None:
            timings[name] = round(dt / 1000, 6)
        log.info("stage_completed", duration_ms=round(dt, 2), result="ok")
