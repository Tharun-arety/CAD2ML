"""Surface point cloud with exact per-point face correspondence.

Algorithm (deterministic for a fixed seed):
1. Allocate ``min_points_per_face`` to every face, distribute the remainder by area
   (largest-remainder rounding) so small but valid faces are always represented.
2. For each face (canonical order) sample points on its own triangles, area-weighted;
   optionally a fraction on the face's boundary polyline (edge bias).
3. Project each sample onto the exact B-Rep surface; take position, outward normal and
   analytic mean/Gaussian curvature from the surface evaluation.
Points are stored contiguously per face, so each face owns a point index range.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from cad2ml.config import PointCloudConfig
from cad2ml.geometry import occ
from cad2ml.representations.mesh import MeshData, triangle_areas
from cad2ml.topology.entities import BRepModel


@dataclass
class PointCloud:
    points: NDArray[np.float32]  # normalized
    points_mm: NDArray[np.float64]  # physical
    normals: NDArray[np.float32]
    face_index: NDArray[np.int32]
    curvature: NDArray[np.float32]  # [N,1] mean curvature (1/mm), convex positive
    gaussian_curvature: NDArray[np.float32]  # [N,1]
    on_boundary: NDArray[np.bool_]
    face_point_range: NDArray[np.int64]  # [F,2] (start, count)
    center: NDArray[np.float64]
    scale: float
    max_projection_distance_mm: float
    max_projection_ratio: (
        float  # max over points of distance / (2 x achieved deflection of its face); <= 1 is ok
    )
    settings: dict[str, Any]

    def to_npz(self) -> dict[str, NDArray[Any]]:
        return {
            "points": self.points,
            "points_mm": self.points_mm,
            "normals": self.normals,
            "face_ids": self.face_index,
            "curvature": self.curvature,
            "gaussian_curvature": self.gaussian_curvature,
            "on_boundary": self.on_boundary,
            "face_point_range": self.face_point_range,
            "center": self.center,
            "scale": np.asarray([self.scale]),
        }


def normalization_transform(bmin: list[float], bmax: list[float]) -> tuple[NDArray[np.float64], float]:
    lo, hi = np.asarray(bmin, float), np.asarray(bmax, float)
    center = (lo + hi) / 2
    scale = float(np.linalg.norm(hi - lo) / 2) or 1.0
    return center, scale


def normalize(p: NDArray[np.float64], center: NDArray[np.float64], scale: float) -> NDArray[np.float64]:
    return (np.asarray(p, float) - center) / scale


def denormalize(p: NDArray[Any], center: NDArray[np.float64], scale: float) -> NDArray[np.float64]:
    return np.asarray(p, float) * scale + center


def allocate(areas: NDArray[np.float64], total: int, minimum: int) -> NDArray[np.int64]:
    n = len(areas)
    base = np.full(n, minimum, dtype=np.int64)
    rest = total - minimum * n
    if rest <= 0 or areas.sum() <= 0:
        return base
    share = areas / areas.sum() * rest
    extra = np.floor(share).astype(np.int64)
    left = rest - int(extra.sum())
    order = np.argsort(-(share - extra), kind="stable")
    extra[order[:left]] += 1
    return base + extra


def _boundary_segments(tris: NDArray[np.int32]) -> NDArray[np.int64]:
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    key = np.sort(e, axis=1)
    uniq, counts = np.unique(key, axis=0, return_counts=True)
    return uniq[counts == 1]


def sample_point_cloud(
    model: BRepModel, mesh: MeshData, cfg: PointCloudConfig, bmin: list[float], bmax: list[float]
) -> PointCloud:
    rng = np.random.default_rng(cfg.seed)
    areas = np.asarray([fr.area_mm2 for fr in model.face_records], dtype=np.float64)
    alloc = allocate(areas, cfg.num_points, cfg.min_points_per_face)
    tri_area = triangle_areas(mesh)
    pts_l, nrm_l, fid_l, cur_l, gau_l, bnd_l = [], [], [], [], [], []
    ranges = np.zeros((len(model.faces), 2), dtype=np.int64)
    worst = 0.0
    worst_ratio = 0.0
    offset = 0
    for fi, face in enumerate(model.faces):
        k = int(alloc[fi])
        start, count = mesh.face_tri_range[fi]
        ftris = mesh.triangles[start : start + count]
        fa = tri_area[start : start + count]
        n_edge = int(round(k * cfg.edge_bias_fraction))
        n_area = k - n_edge
        samples = []
        if n_area:
            p = fa / fa.sum() if fa.sum() > 0 else np.full(len(fa), 1 / len(fa))
            choice = rng.choice(len(ftris), size=n_area, p=p)
            r1, r2 = rng.random(n_area), rng.random(n_area)
            s1 = np.sqrt(r1)
            a, b, c = (mesh.vertices[ftris[choice, j]] for j in range(3))
            samples.append((1 - s1)[:, None] * a + (s1 * (1 - r2))[:, None] * b + (s1 * r2)[:, None] * c)
        is_b = np.zeros(k, dtype=bool)
        if n_edge:
            segs = _boundary_segments(ftris)
            seg_len = np.linalg.norm(mesh.vertices[segs[:, 1]] - mesh.vertices[segs[:, 0]], axis=1)
            pc = seg_len / seg_len.sum()
            ch = rng.choice(len(segs), size=n_edge, p=pc)
            tt = rng.random(n_edge)[:, None]
            samples.append((1 - tt) * mesh.vertices[segs[ch, 0]] + tt * mesh.vertices[segs[ch, 1]])
            is_b[n_area:] = True
        raw = np.concatenate(samples) if samples else np.zeros((0, 3))
        ev = occ.FaceEvaluator(face)
        out_p = np.empty((k, 3))
        out_n = np.empty((k, 3), dtype=np.float32)
        out_c = np.empty(k, dtype=np.float32)
        out_g = np.empty(k, dtype=np.float32)
        for i, q in enumerate(raw):
            try:
                u, v, dist = ev.project(q)
                pnt, nrm, H, K = ev.at_uv(u, v)
            except ValueError:
                pnt, nrm, H, K, dist = list(q), None, float("nan"), float("nan"), 0.0
            if nrm is None:
                # singular point (e.g. cone apex): use mesh-interpolated normal of nearest vertex
                nearest = int(np.argmin(np.linalg.norm(mesh.vertices[ftris].reshape(-1, 3) - q, axis=1)))
                nrm = list(mesh.vertex_normals[ftris.reshape(-1)[nearest]])
                pnt = list(q)
            worst = max(worst, dist)
            bound = (
                2.0 * max(float(mesh.face_deflection[fi]), float(mesh.settings["linear_deflection_mm"]))
                + 1e-4
            )
            worst_ratio = max(worst_ratio, dist / bound)
            out_p[i], out_n[i] = pnt, nrm
            out_c[i] = H if np.isfinite(H) else 0.0
            out_g[i] = K if np.isfinite(K) else 0.0
        ln = np.linalg.norm(out_n, axis=1, keepdims=True)
        out_n = (out_n / np.where(ln > 0, ln, 1)).astype(np.float32)
        pts_l.append(out_p)
        nrm_l.append(out_n)
        fid_l.append(np.full(k, fi, dtype=np.int32))
        cur_l.append(out_c)
        gau_l.append(out_g)
        bnd_l.append(is_b)
        ranges[fi] = (offset, k)
        offset += k
    pts_mm = np.concatenate(pts_l)
    center, scale = normalization_transform(bmin, bmax)
    return PointCloud(
        points=normalize(pts_mm, center, scale).astype(np.float32),
        points_mm=pts_mm,
        normals=np.concatenate(nrm_l),
        face_index=np.concatenate(fid_l),
        curvature=np.concatenate(cur_l)[:, None],
        gaussian_curvature=np.concatenate(gau_l)[:, None],
        on_boundary=np.concatenate(bnd_l),
        face_point_range=ranges,
        center=center,
        scale=scale,
        max_projection_distance_mm=float(worst),
        max_projection_ratio=float(worst_ratio),
        settings={
            **cfg.model_dump(),
            "effective_num_points": int(offset),
            "allocation": "min_per_face + area-proportional largest-remainder",
            "normalization": "p_norm = (p_mm - bbox_center) / (bbox_diagonal / 2)",
            "curvature": "analytic surface mean curvature (1/mm), convex positive",
        },
    )
