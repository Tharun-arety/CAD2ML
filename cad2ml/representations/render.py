"""Headless multi-view observations via a NumPy z-buffer rasterizer.

Why not OpenGL/EGL/OSMesa: those backends depend on host GPU/driver or Mesa builds
that differ between Windows development and Linux containers. A software rasterizer
over the OCCT tessellation is exact for what we need (depth, normals, face IDs),
deterministic, and identical in every environment. See DECISIONS.md (D-006).

Outputs per view: shaded RGB (Lambert, derived from the normal buffer), depth along
the camera axis (mm, 0 = background), world-space outward normals, face-ID mask
(uint16, 0 = background, k+1 = canonical face index k), camera extrinsics/intrinsics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from cad2ml.config import RenderConfig
from cad2ml.representations.mesh import MeshData


@dataclass
class Views:
    rgb: NDArray[np.uint8]  # [V,H,W,3]
    depth: NDArray[np.float32]  # [V,H,W]
    normals: NDArray[np.float16]  # [V,H,W,3]
    face_id: NDArray[np.uint16]  # [V,H,W]
    cameras: list[dict[str, Any]]


def look_at(
    eye: NDArray[np.float64], target: NDArray[np.float64], up: NDArray[np.float64]
) -> NDArray[np.float64]:
    """World->camera 4x4 (camera looks along +z in camera coords, y down, x right)."""
    f = target - eye
    f /= np.linalg.norm(f)
    if abs(f @ up) > 0.99:
        up = np.array([0.0, 1.0, 0.0])
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    d = np.cross(f, r)  # "down"
    R = np.stack([r, d, f])
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = -R @ eye
    return M


def camera_rig(center: NDArray[np.float64], radius: float, cfg: RenderConfig) -> list[dict[str, Any]]:
    cams = []
    fov = math.radians(cfg.fov_deg)
    dist = radius / math.sin(fov / 2) * 1.05
    f_px = (cfg.resolution / 2) / math.tan(fov / 2)
    for k in range(cfg.num_views):
        az = 2 * math.pi * k / cfg.num_views + math.pi / 4
        el = math.radians(cfg.elevation_deg if k % 2 == 0 else -cfg.elevation_deg)
        direction = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        eye = center + dist * direction
        W2C = look_at(eye, center.copy(), np.array([0.0, 0.0, 1.0]))
        cams.append(
            {
                "view_index": k,
                "azimuth_deg": round(math.degrees(az), 6),
                "elevation_deg": round(math.degrees(el), 6),
                "eye_mm": eye.tolist(),
                "target_mm": center.tolist(),
                "world_to_camera": W2C.tolist(),
                "intrinsics": {
                    "fx": f_px,
                    "fy": f_px,
                    "cx": cfg.resolution / 2,
                    "cy": cfg.resolution / 2,
                    "width": cfg.resolution,
                    "height": cfg.resolution,
                },
                "projection": "pinhole; pixel centres at (i+0.5, j+0.5); depth = camera-z in mm",
            }
        )
    return cams


def _rasterize(mesh: MeshData, cam: dict[str, Any]) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    K = cam["intrinsics"]
    W, H = K["width"], K["height"]
    M = np.asarray(cam["world_to_camera"])
    vc = mesh.vertices @ M[:3, :3].T + M[:3, 3]
    z = vc[:, 2]
    px = K["fx"] * vc[:, 0] / z + K["cx"]
    py = K["fy"] * vc[:, 1] / z + K["cy"]
    depth = np.full((H, W), np.inf)
    fid = np.zeros((H, W), dtype=np.uint16)
    nrm = np.zeros((H, W, 3), dtype=np.float32)
    tris = mesh.triangles
    x0, y0, x1, y1 = px[tris[:, 0]], py[tris[:, 0]], px[tris[:, 1]], py[tris[:, 1]]
    x2, y2 = px[tris[:, 2]], py[tris[:, 2]]
    area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    # back-face culling: triangles are CCW seen from outside; in image coords (y down) that is area < 0
    keep = np.flatnonzero((area < -1e-12) & (z[tris].min(axis=1) > 1e-6))
    xmin = np.clip(np.floor(np.minimum(np.minimum(x0, x1), x2) - 0.5), 0, W - 1).astype(int)
    xmax = np.clip(np.ceil(np.maximum(np.maximum(x0, x1), x2) - 0.5), 0, W - 1).astype(int)
    ymin = np.clip(np.floor(np.minimum(np.minimum(y0, y1), y2) - 0.5), 0, H - 1).astype(int)
    ymax = np.clip(np.ceil(np.maximum(np.maximum(y0, y1), y2) - 0.5), 0, H - 1).astype(int)
    vn = mesh.vertex_normals
    for t in keep:
        if xmax[t] < xmin[t] or ymax[t] < ymin[t]:
            continue
        xs = np.arange(xmin[t], xmax[t] + 1) + 0.5
        ys = np.arange(ymin[t], ymax[t] + 1) + 0.5
        gx, gy = np.meshgrid(xs, ys)
        a = area[t]
        w0 = ((x1[t] - gx) * (y2[t] - gy) - (x2[t] - gx) * (y1[t] - gy)) / a
        w1 = ((x2[t] - gx) * (y0[t] - gy) - (x0[t] - gx) * (y2[t] - gy)) / a
        w2 = 1.0 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue
        i0, i1, i2 = tris[t]
        # perspective-correct interpolation
        iz = w0 / z[i0] + w1 / z[i1] + w2 / z[i2]
        zz = 1.0 / np.where(iz > 0, iz, np.inf)
        sub = depth[ymin[t] : ymax[t] + 1, xmin[t] : xmax[t] + 1]
        win = inside & (zz < sub)
        if not win.any():
            continue
        sub[win] = zz[win]
        fid[ymin[t] : ymax[t] + 1, xmin[t] : xmax[t] + 1][win] = mesh.tri_face_index[t] + 1
        p0, p1, p2 = w0 / z[i0] * zz, w1 / z[i1] * zz, w2 / z[i2] * zz
        n = p0[..., None] * vn[i0] + p1[..., None] * vn[i1] + p2[..., None] * vn[i2]
        n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)
        nrm[ymin[t] : ymax[t] + 1, xmin[t] : xmax[t] + 1][win] = n[win]
    depth[~np.isfinite(depth)] = 0.0
    return depth.astype(np.float32), fid, nrm


def render_views(mesh: MeshData, bmin: list[float], bmax: list[float], cfg: RenderConfig) -> Views:
    lo, hi = np.asarray(bmin, float), np.asarray(bmax, float)
    center = (lo + hi) / 2
    radius = float(np.linalg.norm(hi - lo) / 2) or 1.0
    cams = camera_rig(center, radius, cfg)
    R = cfg.resolution
    rgb = np.zeros((len(cams), R, R, 3), dtype=np.uint8)
    dep = np.zeros((len(cams), R, R), dtype=np.float32)
    nor = np.zeros((len(cams), R, R, 3), dtype=np.float16)
    fid = np.zeros((len(cams), R, R), dtype=np.uint16)
    for k, cam in enumerate(cams):
        d, f, n = _rasterize(mesh, cam)
        view_dir = center - np.asarray(cam["eye_mm"])
        view_dir /= np.linalg.norm(view_dir)
        light = -view_dir
        lam = np.clip(n @ light, 0, 1)
        shade = (0.25 + 0.75 * lam) * (f > 0)
        base = np.array([0.72, 0.76, 0.82])
        img = (shade[..., None] * base * 255).astype(np.uint8)
        img[f == 0] = 255
        rgb[k], dep[k], nor[k], fid[k] = img, d, n.astype(np.float16), f
    return Views(rgb, dep, nor, fid, cams)


def unproject(
    depth: NDArray[Any], cam: dict[str, Any], rows: NDArray[Any], cols: NDArray[Any]
) -> NDArray[Any]:
    K = cam["intrinsics"]
    z = depth[rows, cols].astype(np.float64)
    x = (cols + 0.5 - K["cx"]) / K["fx"] * z
    y = (rows + 0.5 - K["cy"]) / K["fy"] * z
    pc = np.stack([x, y, z], axis=1)
    M = np.asarray(cam["world_to_camera"])
    return (pc - M[:3, 3]) @ M[:3, :3]
