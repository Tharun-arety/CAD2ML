"""Repeatable baseline experiment: GNN vs per-face MLP vs rule recognizer, several seeds, two split modes.

    python scripts/train_baseline.py --data data --corpus fixtures/corpus --report docs/evidence/baselines.json

Builds (or reuses, by content id) the group-split and family-holdout datasets, trains each model for each
seed, and evaluates the deterministic rule recognizer on exactly the same test faces. Reports mean/std of
test accuracy and macro-F1, plus per-class metrics and confusion matrices for seed 0.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cad2ml.datasets.builder import build_dataset  # noqa: E402
from cad2ml.evaluation.metrics import classification_report  # noqa: E402
from cad2ml.pipeline import load_manifest  # noqa: E402
from cad2ml.schemas.manifest import FaceRecord  # noqa: E402
from cad2ml.semantics.recognizer import face_labels_from_features  # noqa: E402
from cad2ml.storage.base import LocalFSStore  # noqa: E402
from cad2ml.synthetic.families import LABELS  # noqa: E402
from cad2ml.training.train import TrainConfig, run_training  # noqa: E402


def rule_baseline(store: LocalFSStore, ds: dict[str, Any]) -> dict[str, Any]:
    labels = json.loads(store.get_bytes(f"datasets/{ds['dataset_id']}/labels.json"))
    idx = {lab: i for i, lab in enumerate(LABELS)}
    yt, yp = [], []
    for sid in ds["splits"]["test"]:
        m = load_manifest(store, sid)
        assert m is not None
        faces = [
            FaceRecord.model_validate(f)
            for f in json.loads(store.get_bytes(f"samples/{sid}/brep.json"))["faces"]
        ]
        rule = face_labels_from_features(faces, m.features)
        for fid, lab in zip(labels[sid]["face_ids"], labels[sid]["labels"], strict=True):
            yt.append(idx[lab])
            yp.append(idx[rule[fid]])
    return classification_report(yt, yp, LABELS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--corpus", default="fixtures/corpus")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--dataset-seed", type=int, default=7)
    ap.add_argument("--report", default="docs/evidence/baselines.json")
    a = ap.parse_args()
    store = LocalFSStore(Path(a.data))
    report: dict[str, Any] = {"epochs": a.epochs, "seeds": a.seeds, "labels": list(LABELS), "splits": {}}
    for mode in ("group", "family"):
        ds = build_dataset(store, Path(a.corpus), seed=a.dataset_seed, split_mode=mode)
        entry: dict[str, Any] = {
            "dataset_id": ds["dataset_id"],
            "split_sizes": ds["quality"]["split_sizes"],
            "families_per_split": ds["quality"]["families_per_split"],
            "test_class_distribution": ds["quality"]["class_distribution"].get("test", {}),
            "rule_recognizer_test": rule_baseline(store, ds),
            "models": {},
        }
        for model in ("gnn", "mlp"):
            runs = []
            for seed in a.seeds:
                exp = run_training(
                    store, ds["dataset_id"], TrainConfig(model=model, epochs=a.epochs, seed=seed)
                )
                s = exp["metrics"]["test"]["summary"]
                runs.append(
                    {
                        "seed": seed,
                        "run_id": exp["run_id"],
                        "best_epoch": exp["best_epoch"],
                        "duration_s": exp["duration_s"],
                        **s,
                    }
                )
                print(json.dumps({"mode": mode, "model": model, **runs[-1]}), flush=True)
                if seed == a.seeds[0]:
                    first = exp["metrics"]["test"]
            acc = [r["accuracy"] for r in runs]
            f1 = [r["macro_f1_present_classes"] for r in runs]
            entry["models"][model] = {
                "test_accuracy_mean": round(float(np.mean(acc)), 4),
                "test_accuracy_std": round(float(np.std(acc)), 4),
                "test_macro_f1_mean": round(float(np.mean(f1)), 4),
                "test_macro_f1_std": round(float(np.std(f1)), 4),
                "runs": runs,
                "seed0_per_class": first["per_class"],
                "seed0_confusion": first["confusion_matrix"],
            }
        report["splits"][mode] = entry
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(json.dumps(report, indent=1))
    for mode, e in report["splits"].items():
        print(
            mode,
            e["dataset_id"],
            "rule:",
            e["rule_recognizer_test"]["summary"],
            {
                m: (v["test_accuracy_mean"], v["test_accuracy_std"], v["test_macro_f1_mean"])
                for m, v in e["models"].items()
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
