"""Golden cases: generated parts with fixed parameters and the outputs we compare (tolerance-based).

Compared quantities avoid kernel traversal order: counts, measurements, surface-type histograms and the
multiset of recognized features with their key parameters.
"""

from __future__ import annotations

from typing import Any

import numpy as np

CASES = [
    ("slotted_plate", "slots_fillet", 7101),
    ("flange", "hub", 7102),
    ("housing", "blind_holes", 7103),
    ("clevis", "filleted", 7104),
    ("mounting_bracket", "slotted", 7105),
    ("shaft_support", "rounded", 7106),
]


def summarize(manifest: Any) -> dict[str, Any]:
    g = manifest.geometry
    feats = []
    for f in manifest.features:
        if f.feature_type in ("planar_face", "unknown"):
            continue
        key = {
            k: v for k, v in f.parameters.items() if k in ("diameter_mm", "width_mm", "depth_mm", "radius_mm")
        }
        feats.append({"type": f.feature_type, "n_faces": len(f.participating_faces), **key})
    feats.sort(key=lambda d: (d["type"], sorted((k, str(v)) for k, v in d.items())))
    return {
        "counts": [g.solid_count, g.face_count, g.edge_count, g.vertex_count],
        "volume_mm3": g.volume_mm3,
        "surface_area_mm2": g.surface_area_mm2,
        "bounding_box_mm": g.bounding_box_mm,
        "surface_type_histogram": g.surface_type_histogram,
        "features": feats,
        "n_planar_faces": sum(f.feature_type == "planar_face" for f in manifest.features),
    }


def close(a: Any, b: Any, rel: float = 1e-6, abs_: float = 1e-4) -> bool:
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(close(a[k], b[k], rel, abs_) for k in a)
    if isinstance(a, list):
        return (
            isinstance(b, list)
            and len(a) == len(b)
            and all(close(x, y, rel, abs_) for x, y in zip(a, b, strict=True))
        )
    if isinstance(a, float) or isinstance(b, float):
        return bool(np.isclose(float(a), float(b), rtol=rel, atol=abs_))
    return a == b
