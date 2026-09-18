"""Visual evidence generated from real artifacts (views.npz face-ID masks + shaded views)."""

from __future__ import annotations

import io
import json
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from cad2ml.storage.base import ArtifactStore

LABEL_COLORS = {
    "planar": (170, 178, 190),
    "hole": (220, 60, 60),
    "slot": (240, 160, 40),
    "pocket": (60, 140, 220),
    "fillet": (70, 180, 90),
    "other": (150, 90, 200),
}


def _views(store: ArtifactStore, sample_id: str) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(store.get_bytes(f"samples/{sample_id}/views.npz"))) as d:
        return {k: d[k] for k in d.files}


def _rgb(store: ArtifactStore, sample_id: str, view: int) -> np.ndarray:
    return np.asarray(
        Image.open(io.BytesIO(store.get_bytes(f"samples/{sample_id}/views/view_{view:02d}_rgb.png")))
    )


def face_at_pixel(store: ArtifactStore, sample_id: str, view: int, x: int, y: int) -> str | None:
    fid = _views(store, sample_id)["face_id"][view]
    if not (0 <= y < fid.shape[0] and 0 <= x < fid.shape[1]) or fid[y, x] == 0:
        return None
    return f"F{int(fid[y, x]) - 1:03d}"


def highlight_face(store: ArtifactStore, sample_id: str, view: int, face_id: str) -> bytes:
    rgb = _rgb(store, sample_id, view).astype(np.float32)
    mask = _views(store, sample_id)["face_id"][view] == int(face_id[1:]) + 1
    rgb[mask] = 0.35 * rgb[mask] + 0.65 * np.array([255, 120, 0])
    buf = io.BytesIO()
    Image.fromarray(rgb.clip(0, 255).astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def label_overlay(
    store: ArtifactStore,
    sample_id: str,
    view: int,
    face_labels: dict[str, str],
    mismatches: set[str] | None = None,
) -> np.ndarray:
    rgb = _rgb(store, sample_id, view).astype(np.float32)
    fid = _views(store, sample_id)["face_id"][view]
    shade = rgb.mean(axis=-1, keepdims=True) / 255.0
    out = rgb.copy()
    for face, lab in face_labels.items():
        m = fid == int(face[1:]) + 1
        if not m.any():
            continue
        color = np.array(LABEL_COLORS[lab], dtype=np.float32)
        out[m] = (0.35 + 0.65 * shade[m]) * color
        if mismatches and face in mismatches:
            yy, xx = np.nonzero(m)
            stripe = ((xx + yy) % 6) < 2
            out[yy[stripe], xx[stripe]] = (0, 0, 0)
    return out.clip(0, 255).astype(np.uint8)


def prediction_panel(
    store: ArtifactStore, run_id: str, sample_id: str, predictions: list[dict[str, Any]], view: int = 0
) -> bytes:
    """Side-by-side ground truth vs prediction with a per-face table (face, GT, predicted, confidence)."""
    gt = {r["face_id"]: r["ground_truth"] for r in predictions if r["ground_truth"]}
    pr = {r["face_id"]: r["predicted"] for r in predictions}
    wrong = {f for f in pr if gt.get(f) and gt[f] != pr[f]}
    left = label_overlay(store, sample_id, view, gt) if gt else _rgb(store, sample_id, view)
    right = label_overlay(store, sample_id, view, pr, wrong)
    H, W = left.shape[:2]
    rows = sorted(predictions, key=lambda r: r["face_id"])
    table_h = 18 * (len(rows) + 2)
    canvas = Image.new("RGB", (W * 2 + 360, max(H + 40, table_h + 40)), (255, 255, 255))
    canvas.paste(Image.fromarray(left), (0, 30))
    canvas.paste(Image.fromarray(right), (W, 30))
    d = ImageDraw.Draw(canvas)
    d.text((6, 8), "ground truth (synthetic)", fill=(0, 0, 0))
    d.text((W + 6, 8), "prediction (hatched = wrong)", fill=(0, 0, 0))
    x0 = 2 * W + 10
    d.text((x0, 8), f"{sample_id}  run {run_id}", fill=(0, 0, 0))
    d.text((x0, 26), "face   ground truth  predicted  conf", fill=(0, 0, 0))
    for i, r in enumerate(rows):
        color = (200, 0, 0) if r["face_id"] in wrong else (0, 0, 0)
        d.text(
            (x0, 44 + 18 * i),
            f"{r['face_id']:<6} {r['ground_truth']!s:<13} {r['predicted']:<10} " f"{r['confidence']:.2f}",
            fill=color,
        )
    for j, (lab, col) in enumerate(LABEL_COLORS.items()):
        d.rectangle([6 + j * 70, H + 34, 18 + j * 70, H + 46], fill=col)
        d.text((22 + j * 70, H + 34), lab, fill=(0, 0, 0))
    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()


def lineage_figure(store: ArtifactStore, sample_id: str, face_id: str, view: int = 0) -> bytes:
    """Highlighted face in a view + its point-cloud points projected into that view.

    Points are depth-tested against the stored depth map, so occluded samples are not drawn (drawing them
    would suggest the face is visible where it is not).
    """
    lin = json.loads(store.get_bytes(f"samples/{sample_id}/lineage.json"))["faces"][face_id]
    cams = json.loads(store.get_bytes(f"samples/{sample_id}/cameras.json"))["views"]
    with np.load(io.BytesIO(store.get_bytes(f"samples/{sample_id}/pointcloud.npz"))) as pc:
        s, e = lin["point_range"]
        pts = pc["points_mm"][s:e]
    depth = _views(store, sample_id)["depth"][view]
    img = Image.open(io.BytesIO(highlight_face(store, sample_id, view, face_id))).convert("RGB")
    cam = cams[view]
    M = np.asarray(cam["world_to_camera"])
    K = cam["intrinsics"]
    pc_cam = pts @ M[:3, :3].T + M[:3, 3]
    u = K["fx"] * pc_cam[:, 0] / pc_cam[:, 2] + K["cx"]
    v = K["fy"] * pc_cam[:, 1] / pc_cam[:, 2] + K["cy"]
    diag = float(np.linalg.norm(np.asarray(cam["eye_mm"]) - np.asarray(cam["target_mm"])))
    tol = 2e-3 * diag
    d = ImageDraw.Draw(img)
    for x, y, z in zip(u, v, pc_cam[:, 2], strict=True):
        col, row = int(x), int(y)
        if not (0 <= row < depth.shape[0] and 0 <= col < depth.shape[1]):
            continue
        dz = float(depth[row, col])
        if dz > 0 and z <= dz + tol:
            d.point((float(x), float(y)), fill=(0, 60, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
