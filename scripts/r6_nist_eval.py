"""R6 real-world evaluation on the NIST MBE PMI test models (public domain, not redistributed here).

    python scripts/r6_nist_eval.py --store data/r6 --report docs/evidence/r6_nist_eval.json

The NIST archive holds 11 distinct parts, each exported several ways (AP203 geometry only, AP203 with PMI,
AP242, some tessellated). That supports two measurements without hand labels:

1. robustness: outcome, error code, warnings, repair operations and timing per real STEP file;
2. cross-export consistency: for exports of the *same* part, do measured geometry, canonical-entity
   fingerprints, recognized features and hole diameters agree?

Hole/feature correctness vs. the NIST drawings is audited separately (docs/r6_real_world.md).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cad2ml.datasets.builder import geometry_signature, is_near_duplicate, iter_manifests  # noqa: E402
from cad2ml.schemas.manifest import Manifest  # noqa: E402
from cad2ml.storage.base import LocalFSStore  # noqa: E402

NAME = re.compile(r"nist_(ctc|ftc|stc)_(\d+)_asme1_(.+)\.stp$", re.IGNORECASE)


def part_key(filename: str) -> tuple[str, str]:
    m = NAME.match(filename)
    if not m:
        return filename, "?"
    kind, num, variant = m.groups()
    # STC n is a simplified-PMI version of FTC n with the same geometry intent
    return f"{'ctc' if kind == 'ctc' else 'ftc'}_{num}", f"{kind}:{variant}"


def summarize(store: LocalFSStore, m: Manifest) -> dict[str, Any]:
    row: dict[str, Any] = {
        "file": m.source.filename,
        "status": m.status.value,
        "code": m.rejection.code if m.rejection else None,
        "step_schema": m.source.step_schema,
        "originating_system": m.source.originating_system,
        "units": m.source.original_units,
        "seconds": round(
            m.processing.stage_timings_s.get("parse_isolated", 0.0)
            + m.processing.stage_timings_s.get("extract_isolated", 0.0),
            2,
        ),
    }
    if m.status != "completed":
        row["message"] = (m.rejection.message if m.rejection else "")[:200]
        return row
    g, v = m.geometry, m.validation
    assert g is not None and v is not None
    feats = Counter(f.feature_type for f in m.features)
    holes = sorted(
        round(float(f.parameters["diameter_mm"]), 3)  # type: ignore[arg-type]
        for f in m.features
        if f.feature_type in ("through_hole", "blind_hole")
    )
    brep = json.loads(store.get_bytes(f"samples/{m.sample_id}/brep.json"))
    row.update(
        {
            "faces": g.face_count,
            "edges": g.edge_count,
            "volume_mm3": g.volume_mm3,
            "area_mm2": g.surface_area_mm2,
            "bbox_mm": g.bounding_box_mm,
            "surface_types": g.surface_type_histogram,
            "source_valid": v.source_valid,
            "repair_operations": v.repair_operations,
            "warnings": v.warnings,
            "features": dict(sorted(feats.items())),
            "hole_diameters_mm": holes,
            "unknown_faces": feats.get("unknown", 0),
            "fingerprints": [f["fingerprint"] for f in brep["faces"]],
            "signature": geometry_signature(m),
        }
    )
    return row


def rel_spread(xs: list[float]) -> float:
    return float((max(xs) - min(xs)) / max(abs(np.mean(xs)), 1e-12)) if xs else float("nan")


def consistency(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if r["status"] == "completed"]
    out: dict[str, Any] = {"exports": len(rows), "completed": len(ok)}
    if len(ok) < 2:
        return out
    out["volume_rel_spread"] = rel_spread([r["volume_mm3"] for r in ok])
    out["area_rel_spread"] = rel_spread([r["area_mm2"] for r in ok])
    out["bbox_max_abs_diff_mm"] = float(
        max(np.max(np.abs(np.subtract(a["bbox_mm"], b["bbox_mm"]))) for a, b in combinations(ok, 2))
    )
    out["face_counts"] = sorted({r["faces"] for r in ok})
    jac = []
    exact_ids = []
    for a, b in combinations(ok, 2):
        sa, sb = Counter(a["fingerprints"]), Counter(b["fingerprints"])
        inter, union = sum((sa & sb).values()), sum((sa | sb).values())
        jac.append(inter / union if union else 1.0)
        exact_ids.append(a["fingerprints"] == b["fingerprints"])
    out["fingerprint_jaccard_min"] = round(min(jac), 4)
    out["fingerprint_jaccard_mean"] = round(float(np.mean(jac)), 4)
    out["identical_canonical_sequences_pairs"] = f"{sum(exact_ids)}/{len(exact_ids)}"
    out["hole_diameter_multisets_agree"] = len({tuple(r["hole_diameters_mm"]) for r in ok}) == 1
    out["hole_counts"] = sorted({len(r["hole_diameters_mm"]) for r in ok})
    out["feature_count_sets"] = sorted({json.dumps(r["features"], sort_keys=True) for r in ok})
    out["all_exports_detected_as_near_duplicates"] = all(
        is_near_duplicate(a["signature"], b["signature"]) for a, b in combinations(ok, 2)
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="data/r6")
    ap.add_argument("--report", default="docs/evidence/r6_nist_eval.json")
    a = ap.parse_args()
    store = LocalFSStore(Path(a.store))
    rows = [summarize(store, m) for m in iter_manifests(store)]
    by_part: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        key, variant = part_key(r["file"])
        r["part"], r["variant"] = key, variant
        by_part[key].append(r)
    parts = {k: consistency(v) for k, v in sorted(by_part.items())}
    outcomes = Counter(r["status"] for r in rows)
    codes = Counter(r["code"] for r in rows if r["code"])
    done = [r for r in rows if r["status"] == "completed"]
    summary = {
        "files": len(rows),
        "distinct_parts": len(by_part),
        "outcomes": dict(outcomes),
        "codes": dict(codes),
        "files_with_auxiliary_geometry_ignored": sum(
            any("auxiliary" in w for w in r.get("warnings", [])) for r in done
        ),
        "files_repaired": sum(bool(r.get("repair_operations")) for r in done),
        "source_invalid_files": sum(not r.get("source_valid", True) for r in done),
        "faces_total": int(sum(r["faces"] for r in done)),
        "unknown_face_fraction": round(
            sum(r["unknown_faces"] for r in done) / max(sum(r["faces"] for r in done), 1), 4
        ),
        "surface_types_total": dict(
            sum((Counter(r["surface_types"]) for r in done), Counter()).most_common()
        ),
        "processing_seconds_p50": float(np.median([r["seconds"] for r in rows])) if rows else None,
        "processing_seconds_max": max((r["seconds"] for r in rows), default=None),
        "parts_volume_consistent_1e-3": sum(
            1 for p in parts.values() if p.get("completed", 0) >= 2 and p["volume_rel_spread"] < 1e-3
        ),
        "parts_hole_diameters_consistent": sum(
            1 for p in parts.values() if p.get("hole_diameter_multisets_agree")
        ),
        "parts_identical_canonical_ids_all_pairs": sum(
            1
            for p in parts.values()
            if p.get("completed", 0) >= 2
            and p["identical_canonical_sequences_pairs"].split("/")[0]
            == p["identical_canonical_sequences_pairs"].split("/")[1]
        ),
        "parts_with_2plus_completed_exports": sum(1 for p in parts.values() if p.get("completed", 0) >= 2),
    }
    # near-duplicate detection vs. the known "same part" relation (pairwise precision/recall)
    tp = fp = fn = 0
    for ra, rb in combinations(done, 2):
        same, pred = ra["part"] == rb["part"], is_near_duplicate(ra["signature"], rb["signature"])
        tp += same and pred
        fp += (not same) and pred
        fn += same and (not pred)
    summary["near_duplicate_pairs"] = {
        "true_pos": tp,
        "false_pos": fp,
        "false_neg": fn,
        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "recall": round(tp / (tp + fn), 4) if tp + fn else None,
    }
    for r in rows:
        r.pop("fingerprints", None)
        r.pop("signature", None)
    report = {
        "dataset": "NIST MBE PMI test models (public domain)",
        "summary": summary,
        "parts": parts,
        "files": rows,
    }
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(json.dumps(report, indent=1))
    print(json.dumps(summary, indent=1))
    for k, p in parts.items():
        print(
            k,
            json.dumps(
                {
                    x: p.get(x)
                    for x in (
                        "exports",
                        "completed",
                        "volume_rel_spread",
                        "face_counts",
                        "fingerprint_jaccard_mean",
                        "identical_canonical_sequences_pairs",
                        "hole_counts",
                        "hole_diameter_multisets_agree",
                    )
                }
            ),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
