"""Run untrusted-geometry work in a separate process with a hard timeout.

OCCT is native code: a malformed STEP file can hang or crash the interpreter. All
kernel work therefore happens in a ``spawn``-ed child. The parent only receives
JSON-serialisable results, progress messages, or a structured error. On POSIX the
child additionally applies ``RLIMIT_AS``/``RLIMIT_CPU``; Windows has no stdlib
equivalent, so there only the wall-clock timeout applies (documented limitation).
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import time
import traceback
from collections.abc import Callable
from typing import Any

from cad2ml.errors import PipelineError

_CTX = mp.get_context("spawn")
_PROGRESS_CONN: Any = None


def report_progress(stage: str, **info: Any) -> None:
    """Called inside an isolated child to emit a live stage event to the parent."""
    if _PROGRESS_CONN is not None:
        _PROGRESS_CONN.send(("progress", {"stage": stage, **info}))


def _apply_limits(memory_mb: int, cpu_s: int) -> None:
    if sys.platform.startswith("win"):
        return
    import resource

    if memory_mb > 0:
        b = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (b, b))
    if cpu_s > 0:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 5))


def _child(conn: Any, fn: Callable[..., Any], args: tuple[Any, ...], memory_mb: int, cpu_s: int) -> None:
    global _PROGRESS_CONN
    _PROGRESS_CONN = conn
    try:
        _apply_limits(memory_mb, cpu_s)
        result = fn(*args)
        conn.send(("ok", result))
    except PipelineError as e:
        conn.send(("pipeline_error", {"code": e.code, "message": e.message, "stage": e.stage}))
    except MemoryError:
        conn.send(
            (
                "pipeline_error",
                {"code": "CHILD_CRASHED", "message": "memory limit exceeded", "stage": "isolated"},
            )
        )
    except BaseException as e:
        conn.send(
            (
                "exception",
                {
                    "type": type(e).__name__,
                    "message": str(e)[:2000],
                    "traceback": traceback.format_exc()[-4000:],
                },
            )
        )
    finally:
        conn.close()


def run_isolated(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    *,
    timeout_s: float,
    stage: str,
    timeout_code: str,
    memory_mb: int = 0,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> Any:
    """Execute ``fn(*args)`` in a spawned child. ``fn`` must be a module-level function."""
    parent, child = _CTX.Pipe(duplex=False)
    proc = _CTX.Process(target=_child, args=(child, fn, args, memory_mb, int(timeout_s) + 30), daemon=True)
    proc.start()
    child.close()
    deadline = time.monotonic() + timeout_s
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not parent.poll(remaining):
                raise PipelineError(timeout_code, f"exceeded {timeout_s:.0f}s hard timeout", stage)
            try:
                kind, payload = parent.recv()
            except EOFError as e:
                proc.join(5)
                raise PipelineError(
                    "CHILD_CRASHED", f"child exited with code {proc.exitcode} without result", stage
                ) from e
            if kind == "progress":
                if on_progress is not None:
                    on_progress(payload)
                continue
            break
    finally:
        if proc.is_alive():
            proc.kill()
        proc.join(10)
        parent.close()
    if kind == "ok":
        return payload
    if kind == "pipeline_error":
        raise PipelineError(payload["code"], payload["message"], payload["stage"])
    raise PipelineError(
        "INTERNAL_ERROR", f"{payload['type']}: {payload['message']}\n{payload['traceback']}", stage
    )
