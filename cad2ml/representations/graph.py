"""Face-adjacency graph (PyTorch-Geometric compatible, stored as NumPy arrays).

Node i  <-> canonical face ``F{i:03d}`` (node order == canonical face order).
Directed edges are emitted in both directions for every pair of distinct faces that
share at least one B-Rep edge; seam edges never create self-loops.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from cad2ml.schemas.manifest import EdgeRecord, FaceRecord

GRAPH_SURFACE_TYPES = ["plane", "cylinder", "cone", "sphere", "torus", "freeform"]
FACE_FEATURE_NAMES: list[str] = [f"surface_{t}" for t in GRAPH_SURFACE_TYPES] + [
    "area_fraction",
    "log_area_mm2",
    "normal_x",
    "normal_y",
    "normal_z",
    "mean_curvature_scaled",
    "gaussian_curvature_scaled",
    "radius_scaled",
    "log_radius_mm",
    "orientation_reversed",
    "loop_count",
    "n_boundary_edges",
    "frac_convex_edges",
    "frac_concave_edges",
    "frac_smooth_edges",
    "bbox_extent_sorted_0",
    "bbox_extent_sorted_1",
    "bbox_extent_sorted_2",
    "n_adjacent_faces",
]
ADJ_FEATURE_NAMES = [
    "convex",
    "concave",
    "smooth",
    "other",
    "dihedral_over_180",
    "shared_length_fraction",
    "n_shared_edges",
    "curve_line",
    "curve_circle",
    "curve_other",
]
GLOBAL_FEATURE_NAMES = [
    "log_volume_mm3",
    "log_area_mm2",
    "bbox_sorted_0",
    "bbox_sorted_1",
    "bbox_sorted_2",
    "log_face_count",
    "log_edge_count",
] + [f"hist_{t}" for t in GRAPH_SURFACE_TYPES]


def _stype(t: str) -> str:
    return t if t in GRAPH_SURFACE_TYPES else "freeform"


@dataclass
class FaceGraph:
    face_features: NDArray[np.float32]
    edge_index: NDArray[np.int64]
    adjacency_features: NDArray[np.float32]
    global_features: NDArray[np.float32]
    node_face_ids: list[str]
    face_labels: NDArray[np.int64] | None = None

    def to_npz(self) -> dict[str, NDArray[Any]]:
        out: dict[str, NDArray[Any]] = {
            "face_features": self.face_features,
            "edge_index": self.edge_index,
            "adjacency_features": self.adjacency_features,
            "global_features": self.global_features,
            "node_face_ids": np.asarray(self.node_face_ids),
        }
        if self.face_labels is not None:
            out["face_labels"] = self.face_labels
        return out

    @staticmethod
    def from_npz(d: Any) -> FaceGraph:
        return FaceGraph(
            face_features=d["face_features"],
            edge_index=d["edge_index"],
            adjacency_features=d["adjacency_features"],
            global_features=d["global_features"],
            node_face_ids=[str(x) for x in d["node_face_ids"]],
            face_labels=d["face_labels"] if "face_labels" in d else None,  # noqa: SIM401 (NpzFile has no .get)
        )


def build_face_graph(
    faces: Sequence[FaceRecord],
    edges: Sequence[EdgeRecord],
    volume: float,
    area: float,
    bbox_dims: Sequence[float],
) -> FaceGraph:
    fidx = {f.face_id: i for i, f in enumerate(faces)}
    scale = float(np.linalg.norm(bbox_dims) / 2) or 1.0
    edge_by_id = {e.edge_id: e for e in edges}
    F = len(faces)
    X = np.zeros((F, len(FACE_FEATURE_NAMES)), dtype=np.float64)
    for i, f in enumerate(faces):
        oh = [1.0 if _stype(f.surface_type) == t else 0.0 for t in GRAPH_SURFACE_TYPES]
        n = f.normal_at_centroid or [0.0, 0.0, 0.0]
        H = f.curvature.get("mean_curvature_mean", 0.0)
        K = f.curvature.get("gaussian_curvature_mean", 0.0)
        ap = f.analytic_params
        radius = ap.get("radius", ap.get("minor_radius", ap.get("ref_radius", 0.0)))
        radius = float(radius) if isinstance(radius, int | float) else 0.0
        be = [edge_by_id[e] for e in f.boundary_edge_ids]
        real = [e for e in be if e.convexity not in ("seam",)] or be
        nb = max(len(real), 1)
        ext = sorted((np.asarray(f.bbox_max_mm) - np.asarray(f.bbox_min_mm)) / (2 * scale), reverse=True)
        X[i] = oh + [  # noqa: RUF005
            f.area_mm2 / area if area > 0 else 0.0,
            math.log1p(f.area_mm2),
            *n,
            H * scale,
            K * scale * scale,
            radius / scale,
            math.log1p(radius),
            1.0 if f.orientation == "reversed" else 0.0,
            float(f.loop_count),
            float(len(f.boundary_edge_ids)),
            sum(e.convexity == "convex" for e in real) / nb,
            sum(e.convexity == "concave" for e in real) / nb,
            sum(e.convexity == "smooth" for e in real) / nb,
            *ext,
            float(len(f.adjacent_face_ids)),
        ]
    pair: dict[tuple[int, int], list[EdgeRecord]] = {}
    for e in edges:
        ids = sorted({fidx[a] for a in e.adjacent_face_ids})
        for a in ids:
            for b in ids:
                if a != b:
                    pair.setdefault((a, b), []).append(e)
    keys = sorted(pair)
    ei = np.asarray(keys, dtype=np.int64).T if keys else np.zeros((2, 0), dtype=np.int64)
    A = np.zeros((len(keys), len(ADJ_FEATURE_NAMES)), dtype=np.float64)
    for k, (a, b) in enumerate(keys):
        es = pair[(a, b)]
        total = sum(e.length_mm for e in es) or 1.0
        main = max(es, key=lambda e: e.length_mm)
        conv = main.convexity
        per = total
        A[k] = [
            conv == "convex",
            conv == "concave",
            conv == "smooth",
            conv not in ("convex", "concave", "smooth"),
            (main.dihedral_angle_deg or 180.0) / 180.0,
            per / (sum(edge_by_id[x].length_mm for x in faces[a].boundary_edge_ids) or 1.0),
            float(len(es)),
            main.curve_type == "line",
            main.curve_type == "circle",
            main.curve_type not in ("line", "circle"),
        ]
    hist = [sum(_stype(f.surface_type) == t for f in faces) / max(F, 1) for t in GRAPH_SURFACE_TYPES]
    bd = sorted((float(x) / (2 * scale) for x in bbox_dims), reverse=True)
    G = np.asarray(
        [math.log1p(abs(volume)), math.log1p(area), *bd, math.log1p(F), math.log1p(len(edges)), *hist]
    )
    return FaceGraph(
        X.astype(np.float32), ei, A.astype(np.float32), G.astype(np.float32), [f.face_id for f in faces]
    )


def check_graph_invariants(
    g: FaceGraph, faces: Sequence[FaceRecord], edges: Sequence[EdgeRecord]
) -> dict[str, bool]:
    F = len(faces)
    ei = g.edge_index
    shared: set[tuple[int, int]] = set()
    fidx = {f.face_id: i for i, f in enumerate(faces)}
    for e in edges:
        ids = {fidx[a] for a in e.adjacent_face_ids}
        shared |= {(a, b) for a in ids for b in ids if a != b}
    pairs = set(map(tuple, ei.T.tolist())) if ei.size else set()
    return {
        "nodes_match_faces": g.face_features.shape[0] == F and g.node_face_ids == [f.face_id for f in faces],
        "edge_index_in_range": bool(ei.size == 0 or (ei.min() >= 0 and ei.max() < F)),
        "no_self_loops": bool(ei.size == 0 or not np.any(ei[0] == ei[1])),
        "adjacency_equals_shared_topology": pairs == shared,
        "symmetric": all((b, a) in pairs for a, b in pairs),
        "feature_dims_consistent": g.face_features.shape[1] == len(FACE_FEATURE_NAMES)
        and g.adjacency_features.shape == (ei.shape[1], len(ADJ_FEATURE_NAMES))
        and g.global_features.shape == (len(GLOBAL_FEATURE_NAMES),),
        "all_finite": bool(
            np.isfinite(g.face_features).all()
            and np.isfinite(g.adjacency_features).all()
            and np.isfinite(g.global_features).all()
        ),
    }
