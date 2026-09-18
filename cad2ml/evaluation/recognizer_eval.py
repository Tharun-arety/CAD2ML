"""Evaluate pipeline outputs against synthetic ground truth (only data read from stored artifacts).

* global dimensions: extracted bounding box vs dimensions implied by generator parameters
* face-level: rule-recognizer face labels vs ground-truth face labels (confusion + P/R/F1)
* feature-level: holes (type, diameter, axis position), slots, pockets matched by geometry
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from cad2ml.datasets.builder import iter_manifests, load_ground_truth
from cad2ml.evaluation.metrics import classification_report
from cad2ml.schemas.manifest import FaceRecord, Manifest
from cad2ml.semantics.recognizer import face_labels_from_features
from cad2ml.storage.base import ArtifactStore
from cad2ml.synthetic.families import LABELS


def expected_bbox(gt: dict[str, Any]) -> list[float]:
    p, fam = gt["params"], gt["family"]
    if fam == "slotted_plate":
        return [p["length_mm"], p["width_mm"], p["thickness_mm"]]
    if fam == "mounting_bracket":
        return [p["length_mm"], p["base_width_mm"], p["height_mm"]]
    if fam == "flange":
        return [
            2 * p["outer_radius_mm"],
            2 * p["outer_radius_mm"],
            p["thickness_mm"] + p.get("hub_height_mm", 0.0),
        ]
    if fam == "shaft_support":
        return [p["length_mm"], p["width_mm"], p["height_mm"]]
    if fam in ("housing", "clevis"):
        return [p["length_mm"], p["width_mm"], p["height_mm"]]
    raise KeyError(fam)


def _num(v: Any) -> float:
    return float(v)


def _line_dist(p: np.ndarray, o: np.ndarray, d: np.ndarray) -> float:
    v = p - o
    return float(np.linalg.norm(v - (v @ d) * d))


def evaluate(store: ArtifactStore, corpus: Path, sample_ids: set[str] | None = None) -> dict[str, Any]:
    from cad2ml.synthetic.labels import label_faces

    gts = load_ground_truth(corpus)
    dims_err: list[float] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    idx = {lab: i for i, lab in enumerate(LABELS)}
    feat_stats: dict[str, dict[str, int]] = {
        k: {"tp": 0, "fp": 0, "fn": 0} for k in ("through_hole", "blind_hole", "slot", "pocket")
    }
    diam_err: list[float] = []
    per_sample: list[dict[str, Any]] = []
    for m in iter_manifests(store):
        if (
            m.status != "completed"
            or m.source.sha256 not in gts
            or (sample_ids and m.sample_id not in sample_ids)
        ):
            continue
        gt = gts[m.source.sha256]
        base = f"samples/{m.sample_id}"
        assert m.geometry is not None
        err = np.abs(np.asarray(m.geometry.bounding_box_mm) - np.asarray(expected_bbox(gt)))
        dims_err.append(float(err.max()))
        faces = [
            FaceRecord.model_validate(f) for f in json.loads(store.get_bytes(f"{base}/brep.json"))["faces"]
        ]
        import io

        with np.load(io.BytesIO(store.get_bytes(f"{base}/pointcloud.npz"))) as pc:
            labs, _ = label_faces(faces, pc["points_mm"], pc["face_point_range"], gt)
        rule = face_labels_from_features(faces, m.features)
        wrong = []
        for f, lab in zip(faces, labs, strict=True):
            y_true.append(idx[lab])
            y_pred.append(idx[rule[f.face_id]])
            if lab != rule[f.face_id]:
                wrong.append(
                    {"face_id": f.face_id, "gt": lab, "rule": rule[f.face_id], "surface": f.surface_type}
                )
        _match_features(m, gt, feat_stats, diam_err)
        per_sample.append(
            {
                "sample_id": m.sample_id,
                "file": m.source.filename,
                "group": gt["group"],
                "bbox_max_abs_err_mm": float(err.max()),
                "face_label_errors": wrong,
            }
        )
    fs = {}
    for k, v in feat_stats.items():
        p = v["tp"] / (v["tp"] + v["fp"]) if v["tp"] + v["fp"] else 0.0
        r = v["tp"] / (v["tp"] + v["fn"]) if v["tp"] + v["fn"] else 0.0
        fs[k] = {**v, "precision": round(p, 4), "recall": round(r, 4)}
    return {
        "samples": len(per_sample),
        "global_dimensions": {
            "max_abs_bbox_error_mm": max(dims_err) if dims_err else None,
            "mean_max_abs_bbox_error_mm": float(np.mean(dims_err)) if dims_err else None,
        },
        "face_level_rule_vs_ground_truth": classification_report(y_true, y_pred, LABELS),
        "feature_level": fs,
        "hole_diameter_abs_error_mm": {
            "max": max(diam_err) if diam_err else None,
            "mean": float(np.mean(diam_err)) if diam_err else None,
            "n": len(diam_err),
        },
        "samples_with_face_label_errors": [s for s in per_sample if s["face_label_errors"]],
    }


def _match_features(
    m: Manifest, gt: dict[str, Any], stats: dict[str, dict[str, int]], diam_err: list[float]
) -> None:
    pred = [f for f in m.features if f.feature_type in stats]
    used: set[str] = set()
    for g in gt["features"]:
        kind = g["feature_type"]
        if kind not in stats:
            continue
        match = None
        for f in pred:
            if f.feature_id in used or f.feature_type != kind:
                continue
            if kind in ("through_hole", "blind_hole"):
                gp = g["parameters"]
                ok = (
                    abs(_num(f.parameters["diameter_mm"]) - gp["diameter_mm"]) < 1e-3
                    and _line_dist(
                        np.asarray(gp["center_mm"], float),
                        np.asarray(f.parameters["axis_point_mm"], float),
                        np.asarray(f.parameters["axis"], float),
                    )
                    < 1e-2
                )
            elif kind == "slot":
                ok = abs(_num(f.parameters["width_mm"]) - g["parameters"]["width_mm"]) < 1e-2
            else:
                ok = abs(_num(f.parameters["depth_mm"]) - g["parameters"]["depth_mm"]) < 1e-2
            if ok:
                match = f
                break
        if match is None:
            stats[kind]["fn"] += 1
        else:
            used.add(match.feature_id)
            stats[kind]["tp"] += 1
            if kind in ("through_hole", "blind_hole"):
                diam_err.append(abs(_num(match.parameters["diameter_mm"]) - g["parameters"]["diameter_mm"]))
    for f in pred:
        if f.feature_id not in used:
            stats[f.feature_type]["fp"] += 1
