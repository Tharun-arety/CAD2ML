"""CAD2ML: STEP -> canonical B-Rep -> aligned, traceable ML representations and datasets."""

import os as _os

# Geometry children are single-threaded by design. Multi-threaded OpenBLAS in many short-lived
# spawned processes exhausted its per-process buffer pool on Windows (observed CHILD_CRASHED,
# "OpenBLAS error: Memory allocation still failed"), see DECISIONS.md D-009.
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

__version__ = "1.0.0"
