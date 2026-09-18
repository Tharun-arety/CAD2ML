"""Project synthetic ground truth onto processed faces.

Uses only (a) the sidecar ground-truth JSON written by the generator and (b) the
artifacts produced by the public pipeline (face records + point cloud). A face is
assigned a feature label when >= ``min_fraction`` of its surface samples lie on the
analytic tool boundary. Fillet faces are matched by surface type + radius. This is
independent from the rule-based recognizer, so the recognizer can be evaluated.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from cad2ml.schemas.manifest import FaceRecord
from cad2ml.synthetic.families import LABELS
from cad2ml.synthetic.tools import ExtrudedRoundedRect

LABEL_PRIORITY = ("hole", "slot", "pocket")


def label_faces(
    faces: Sequence[FaceRecord],
    points_mm: NDArray[np.float64],
    face_point_range: NDArray[np.int64],
    gt: dict[str, Any],
    tol_mm: float = 2e-3,
    min_fraction: float = 0.95,
) -> tuple[list[str], list[str | None]]:
    """Returns (label per face, ground-truth feature key per face or None)."""
    tools: list[tuple[str, str, ExtrudedRoundedRect]] = []
    for k, feat in enumerate(gt["features"]):
        if feat.get("tool"):
            tools.append(
                (
                    feat["label"],
                    f"gt{k:03d}:{feat['feature_type']}",
                    ExtrudedRoundedRect.from_dict(feat["tool"]),
                )
            )
    tools.sort(key=lambda t: LABEL_PRIORITY.index(t[0]) if t[0] in LABEL_PRIORITY else 99)
    fillet_r = [float(r) for r in gt.get("fillet_radii", [])]
    labels: list[str] = []
    keys: list[str | None] = []
    for i, f in enumerate(faces):
        s, c = face_point_range[i]
        pts = points_mm[s : s + c]
        lab, key = None, None
        for tl, tk, tool in tools:
            if len(pts) and float(np.mean(tool.on_boundary(pts, tol_mm))) >= min_fraction:
                lab, key = tl, tk
                break
        if lab is None and f.surface_type in ("cylinder", "torus") and fillet_r:
            ap = f.analytic_params
            r = float(ap.get("radius", ap.get("minor_radius", -1.0)))  # type: ignore[arg-type]
            if any(abs(r - fr) < 1e-3 for fr in fillet_r):
                lab, key = "fillet", "gt:fillet"
        if lab is None:
            lab = "planar" if f.surface_type == "plane" else "other"
        assert lab in LABELS
        labels.append(lab)
        keys.append(key)
    return labels, keys
