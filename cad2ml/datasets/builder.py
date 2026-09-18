"""Immutable, versioned dataset construction from processed samples.

Selection policy
  * include: status == completed, current pipeline/schema/config, ground-truth sidecar present
  * exclude (with reason): rejected/quarantined/failed samples (by error code), config mismatch,
    missing ground truth, missing artifacts, exact duplicate source hash
Splitting
  * unit = ``design group`` (``family/variant``) in ``group`` mode, or ``family`` in ``family`` mode
  * near-duplicate candidates (geometry signature) are unioned into the same split unit
  * a leakage check asserts no split unit (and, in family mode, no family) spans two splits
Identity
  * dataset_id = "ds_" + sha256(canonical content)[:16]; creation timestamp is not hashed.
    Rebuilding with identical inputs yields the same id; an existing version is never overwritten.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from cad2ml.config import PIPELINE_VERSION, SCHEMA_VERSION, PipelineConfig
from cad2ml.schemas.manifest import Manifest
from cad2ml.storage.base import ArtifactStore
from cad2ml.synthetic.families import LABELS
from cad2ml.synthetic.labels import label_faces

LABEL_POLICY_VERSION = "gt_tool_boundary_v1"
REQUIRED_ARTIFACTS = ("brep", "graph", "pointcloud", "mesh", "views", "cameras", "features", "lineage")


def _npz(store: ArtifactStore, key: str) -> dict[str, Any]:
    with np.load(io.BytesIO(store.get_bytes(key)), allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def geometry_signature(m: Manifest) -> tuple[Any, ...]:
    """Coarse, pose-invariant signature for near-duplicate *candidate* detection."""
    g = m.geometry
    assert g is not None
    dims = sorted(g.bounding_box_mm)
    return (
        round(g.volume_mm3 / 50.0),
        round(g.surface_area_mm2 / 25.0),
        tuple(round(d / 0.5) for d in dims),
        g.face_count,
        tuple(sorted(g.surface_type_histogram.items())),
    )


def iter_manifests(store: ArtifactStore) -> list[Manifest]:
    out = []
    for key in store.list("samples"):
        if key.endswith("/manifest.json") and key.count("/") == 2:
            out.append(Manifest.model_validate_json(store.get_bytes(key)))
    return sorted(out, key=lambda m: m.sample_id)


def load_ground_truth(corpus: Path) -> dict[str, dict[str, Any]]:
    gts = {}
    for p in sorted((corpus / "parts").glob("*.gt.json")):
        d = json.loads(p.read_text())
        gts[d["source_sha256"]] = d
    return gts


class _UF:
    def __init__(self) -> None:
        self.p: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def assign_splits(units: dict[str, str], seed: int, mode: str) -> dict[str, str]:
    """Map split-unit -> split. ``units`` maps unit -> family.

    group mode: per family, shuffle its units; the first held-out unit goes to val or test
    (alternating across families), all others to train. family mode: whole families are
    assigned train/val/test (~4/1/1 for 6 families).
    """
    rng = np.random.default_rng(seed)
    out: dict[str, str] = {}
    if mode == "family":
        fams = sorted(set(units.values()))
        order = [fams[i] for i in rng.permutation(len(fams))]
        n_hold = max(1, round(len(fams) / 6))
        split_of = {f: "test" for f in order[:n_hold]}
        split_of.update({f: "val" for f in order[n_hold : 2 * n_hold]})
        for u, f in units.items():
            out[u] = split_of.get(f, "train")
        return out
    by_fam: dict[str, list[str]] = defaultdict(list)
    for u, f in sorted(units.items()):
        by_fam[f].append(u)
    fams = sorted(by_fam)
    fam_order = [fams[i] for i in rng.permutation(len(fams))]
    for k, fam in enumerate(fam_order):
        us = sorted(by_fam[fam])
        us = [us[i] for i in rng.permutation(len(us))]
        for j, u in enumerate(us):
            if j == 0 and len(us) >= 2:
                out[u] = "val" if k % 2 == 0 else "test"
            elif j == 1 and len(us) >= 4:
                out[u] = "test" if k % 2 == 0 else "val"
            else:
                out[u] = "train"
    return out


def check_leakage(samples: dict[str, dict[str, Any]], key: str) -> dict[str, list[str]]:
    seen: dict[str, set[str]] = defaultdict(set)
    for s in samples.values():
        seen[s[key]].add(s["split"])
    return {k: sorted(v) for k, v in seen.items() if len(v) > 1}


def build_dataset(
    store: ArtifactStore,
    corpus: Path,
    seed: int = 7,
    split_mode: str = "group",
    cfg: PipelineConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or PipelineConfig()
    chash = cfg.config_hash()
    gts = load_ground_truth(corpus)
    included: dict[str, dict[str, Any]] = {}
    excluded: list[dict[str, str]] = []
    reasons: Counter[str] = Counter()
    by_sha: dict[str, str] = {}
    manifests: dict[str, Manifest] = {}
    for m in iter_manifests(store):
        reason = None
        if m.status != "completed":
            reason = f"{m.status.value}:{m.rejection.code if m.rejection else 'unknown'}"
        elif (
            m.processing.configuration_hash,
            m.processing.pipeline_version,
            m.processing.schema_version,
        ) != (chash, PIPELINE_VERSION, SCHEMA_VERSION):
            reason = "config_or_version_mismatch"
        elif m.source.sha256 in by_sha:
            reason = f"exact_duplicate_of:{by_sha[m.source.sha256]}"
        elif m.source.sha256 not in gts:
            reason = "no_ground_truth_sidecar"
        elif any(a not in m.artifacts for a in REQUIRED_ARTIFACTS):
            reason = "missing_artifacts"
        if reason:
            excluded.append({"sample_id": m.sample_id, "filename": m.source.filename, "reason": reason})
            reasons[reason.split(":")[0] if reason.startswith("exact") else reason] += 1
            continue
        by_sha[m.source.sha256] = m.sample_id
        gt = gts[m.source.sha256]
        manifests[m.sample_id] = m
        included[m.sample_id] = {
            "sample_id": m.sample_id,
            "source_sha256": m.source.sha256,
            "filename": m.source.filename,
            "family": gt["family"],
            "group": gt["group"],
            "signature": geometry_signature(m),
        }

    # near-duplicate candidates -> same split unit
    uf = _UF()
    for s in included.values():
        uf.find(s["group"])
    by_sig: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    for sid, s in included.items():
        by_sig[s["signature"]].append(sid)
    near_dups = []
    for sig_ids in by_sig.values():
        if len(sig_ids) > 1:
            near_dups.append(sorted(sig_ids))
            for other in sig_ids[1:]:
                uf.union(included[sig_ids[0]]["group"], included[other]["group"])
    unit_key = "family" if split_mode == "family" else "group"
    units = {
        uf.find(s[unit_key]) if unit_key == "group" else s["family"]: s["family"] for s in included.values()
    }
    split_of_unit = assign_splits(units, seed, split_mode)
    for s in included.values():
        s["split_unit"] = uf.find(s["group"]) if unit_key == "group" else s["family"]
        s["split"] = split_of_unit[s["split_unit"]]

    leak_unit = check_leakage(included, "split_unit")
    leak_family = check_leakage(included, "family") if split_mode == "family" else {}
    if leak_unit or leak_family:
        raise AssertionError(f"split leakage detected: units={leak_unit} families={leak_family}")

    # ground-truth face labels + recognizer agreement
    label_idx = {lab: i for i, lab in enumerate(LABELS)}
    class_counts: dict[str, Counter[str]] = defaultdict(Counter)
    rule_agree: Counter[str] = Counter()
    labels_payload: dict[str, dict[str, Any]] = {}
    completeness: Counter[str] = Counter()
    from cad2ml.semantics.recognizer import face_labels_from_features

    for sid, s in sorted(included.items()):
        m = manifests[sid]
        base = f"samples/{sid}"
        brep = json.loads(store.get_bytes(f"{base}/brep.json"))
        from cad2ml.schemas.manifest import FaceRecord

        faces = [FaceRecord.model_validate(f) for f in brep["faces"]]
        pc = _npz(store, f"{base}/pointcloud.npz")
        labs, keys = label_faces(faces, pc["points_mm"], pc["face_point_range"], gts[m.source.sha256])
        rule = face_labels_from_features(faces, m.features)
        for f, lab in zip(faces, labs, strict=True):
            class_counts[s["split"]][lab] += 1
            rule_agree["agree" if rule[f.face_id] == lab else "disagree"] += 1
        labels_payload[sid] = {
            "face_ids": [f.face_id for f in faces],
            "labels": labs,
            "label_index": [label_idx[x] for x in labs],
            "gt_feature_keys": keys,
        }
        for a in REQUIRED_ARTIFACTS:
            completeness[a] += int(store.exists(f"{base}/{m.artifacts[a].key}"))
        s.pop("signature")

    splits = {
        sp: sorted(sid for sid, s in included.items() if s["split"] == sp) for sp in ("train", "val", "test")
    }
    content = {
        "schema": "cad2ml.dataset/1.0.0",
        "pipeline_version": PIPELINE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "configuration_hash": chash,
        "representation_schema": {
            "graph": "graph.npz (face_features, edge_index, adjacency_features, "
            "global_features, node_face_ids)",
            "pointcloud": "pointcloud.npz",
            "mesh": "mesh.npz",
            "views": "views.npz",
        },
        "label_vocabulary": list(LABELS),
        "label_policy": LABEL_POLICY_VERSION,
        "normalization": {
            "pointcloud": "bbox centre / half diagonal",
            "graph": "per-feature train-split " "standardisation at training time",
        },
        "sampling": cfg.pointcloud.model_dump(),
        "split_strategy": {
            "mode": split_mode,
            "unit": "design group family/variant (near-duplicate candidates merged)"
            if split_mode == "group"
            else "part family",
            "seed": seed,
        },
        "samples": {sid: {k: v for k, v in s.items()} for sid, s in sorted(included.items())},
        "splits": splits,
        "excluded": sorted(excluded, key=lambda x: x["sample_id"]),
        "label_content_sha256": hashlib.sha256(
            json.dumps(labels_payload, sort_keys=True).encode()
        ).hexdigest(),
    }
    dataset_id = "ds_" + hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()[:16]
    quality = {
        "included": len(included),
        "excluded": len(excluded),
        "rejection_reasons": dict(sorted(reasons.items())),
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "families_per_split": {
            sp: sorted({included[s]["family"] for s in ids}) for sp, ids in splits.items()
        },
        "groups_per_split": {sp: sorted({included[s]["group"] for s in ids}) for sp, ids in splits.items()},
        "class_distribution": {sp: dict(sorted(c.items())) for sp, c in class_counts.items()},
        "representation_completeness": {
            a: completeness[a] / max(len(included), 1) for a in REQUIRED_ARTIFACTS
        },
        "near_duplicate_candidate_clusters": near_dups,
        "leakage": {
            "split_units_crossing_splits": leak_unit,
            "families_crossing_splits": leak_family,
            "passed": not leak_unit and not leak_family,
        },
        "rule_recognizer_face_label_agreement": rule_agree["agree"] / max(sum(rule_agree.values()), 1),
    }
    content["quality"] = quality
    key = f"datasets/{dataset_id}/dataset.json"
    if store.exists(key):
        return json.loads(store.get_bytes(key))  # immutable: never overwrite an existing version
    content["dataset_id"] = dataset_id
    content["created_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    store.put_bytes(f"datasets/{dataset_id}/labels.json", json.dumps(labels_payload, sort_keys=True).encode())
    store.put_bytes(key, json.dumps(content, indent=1, sort_keys=True).encode())
    return content
