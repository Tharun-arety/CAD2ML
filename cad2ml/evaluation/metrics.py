"""Classification metrics computed from scratch (no hidden library defaults)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], n: int) -> np.ndarray:
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred, strict=True):
        cm[t, p] += 1
    return cm


def classification_report(
    y_true: Sequence[int], y_pred: Sequence[int], labels: Sequence[str]
) -> dict[str, Any]:
    n = len(labels)
    cm = confusion_matrix(y_true, y_pred, n)
    per: dict[str, dict[str, float | int]] = {}
    f1s = []
    for i, lab in enumerate(labels):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        support = int(cm[i, :].sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per[lab] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": support,
        }
        if support > 0:
            f1s.append(f1)
    total = int(cm.sum())
    return {
        "confusion_matrix": {"labels": list(labels), "rows_true_cols_pred": cm.tolist()},
        "per_class": per,
        "summary": {
            "accuracy": round(float(np.trace(cm)) / total, 4) if total else 0.0,
            "macro_f1_present_classes": round(float(np.mean(f1s)), 4) if f1s else 0.0,
            "n_faces": total,
            "classes_present": len(f1s),
        },
    }
