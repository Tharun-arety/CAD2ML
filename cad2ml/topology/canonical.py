"""Deterministic canonical ordering and entity fingerprints (pure Python, kernel-free).

Scope of canonical IDs (documented in docs/canonical_ids.md):
* Deterministic for a given B-Rep: independent of the kernel's traversal order,
  because ordering is derived only from quantized intrinsic geometry.
* Reproducible across re-processing of identical geometry with the same pipeline
  version and configuration.
* NOT stable across CAD edits: adding a hole can shift every subsequent ID.
  Approximate cross-revision matching uses fingerprints (``match_entities``) and is
  labelled experimental.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

QUANT_MM = 1e-3


def q(x: float, step: float = QUANT_MM) -> int:
    return round(x / step)


def qv(v: Sequence[float], step: float = QUANT_MM) -> tuple[int, ...]:
    return tuple(q(x, step) for x in v)


def _h(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def intrinsic_face_signature(
    surface_type: str, area: float, analytic: dict[str, Any], loops: int, n_edges: int
) -> dict[str, Any]:
    """Pose-independent attributes (no centroid/axis position), quantized coarsely."""
    shape_params = {
        k: round(float(v), 3)
        for k, v in sorted(analytic.items())
        if k in ("radius", "major_radius", "minor_radius", "semi_angle_rad", "ref_radius")
    }
    return {"t": surface_type, "a": q(area, 1e-2), "p": shape_params, "l": loops, "e": n_edges}


def face_fingerprint(intrinsic: dict[str, Any], neighbour_types: Sequence[str]) -> str:
    hist: dict[str, int] = {}
    for t in neighbour_types:
        hist[t] = hist.get(t, 0) + 1
    return _h({"i": intrinsic, "n": dict(sorted(hist.items()))})


def face_sort_key(
    surface_type_rank: int,
    centroid: Sequence[float],
    area: float,
    bbox_min: Sequence[float],
    bbox_max: Sequence[float],
    fingerprint: str,
) -> tuple[Any, ...]:
    return (surface_type_rank, qv(centroid), q(area, 1e-3), qv(bbox_min), qv(bbox_max), fingerprint)


def edge_fingerprint(curve_type: str, length: float, convexity: str, adjacent_face_fps: Sequence[str]) -> str:
    return _h({"t": curve_type, "L": q(length, 1e-2), "c": convexity, "f": sorted(adjacent_face_fps)})


def edge_sort_key(
    curve_type_rank: int,
    midpoint: Sequence[float],
    length: float,
    start: Sequence[float],
    end: Sequence[float],
    fingerprint: str,
) -> tuple[Any, ...]:
    ends = sorted([qv(start), qv(end)])
    return (curve_type_rank, qv(midpoint), q(length, 1e-3), ends[0], ends[1], fingerprint)


def canonical_order(keys: Sequence[tuple[Any, ...]]) -> list[int]:
    """Return the original indices sorted by key (stable, total order)."""
    return sorted(range(len(keys)), key=lambda i: (keys[i], i))


def face_id(rank: int) -> str:
    return f"F{rank:03d}"


def edge_id(rank: int) -> str:
    return f"E{rank:03d}"


def match_entities(fps_a: Sequence[str], fps_b: Sequence[str]) -> dict[int, int]:
    """EXPERIMENTAL: one-to-one match of entities between two revisions by identical fingerprint.

    Only unambiguous (unique on both sides) fingerprints are matched. This is a candidate
    generator for revision diffing, not a validated correspondence method.
    """
    from collections import Counter

    ca, cb = Counter(fps_a), Counter(fps_b)
    idx_b = {fp: i for i, fp in enumerate(fps_b)}
    return {i: idx_b[fp] for i, fp in enumerate(fps_a) if ca[fp] == 1 and cb.get(fp) == 1}
