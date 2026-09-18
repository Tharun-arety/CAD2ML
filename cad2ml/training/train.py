"""Deterministic CPU training + evaluation of face classifiers on a dataset version.

Outputs under ``training_runs/<run_id>/``:
    experiment.json   config, dataset id, versions, seed, timings, metrics (train/val/test)
    checkpoint.pt     best-on-validation weights + standardizer
    predictions.json  per (sample_id, face_id): ground truth, prediction, confidence
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from cad2ml.config import PIPELINE_VERSION
from cad2ml.evaluation.metrics import classification_report
from cad2ml.storage.base import ArtifactStore
from cad2ml.synthetic.families import LABELS
from cad2ml.training.data import (
    Standardizer,
    apply_standardizer,
    fit_standardizer,
    load_dataset_manifest,
    load_graphs,
)
from cad2ml.training.models import FaceGNN, FaceMLP


@dataclass(frozen=True)
class TrainConfig:
    model: str = "gnn"  # "gnn" | "mlp"
    epochs: int = 150
    lr: float = 3e-3
    weight_decay: float = 1e-4
    hidden: int = 64
    batch_size: int = 8
    seed: int = 0
    class_weighting: str = "inverse_sqrt_frequency"
    threads: int = 4
    gnn_aggr: str = "mean"  # "add" | "mean"; selected on validation macro-F1 (DECISIONS D-014)


def seed_everything(seed: int, threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def _git_sha() -> str:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=False
            ).stdout.strip()
            or "unknown"
        )
    except Exception:
        return "unknown"


def build_model(cfg: TrainConfig, in_dim: int, edge_dim: int) -> torch.nn.Module:
    if cfg.model == "mlp":
        return FaceMLP(in_dim, len(LABELS), cfg.hidden)
    if cfg.model == "gnn":
        return FaceGNN(in_dim, edge_dim, len(LABELS), cfg.hidden, aggr=cfg.gnn_aggr)
    raise ValueError(cfg.model)


@torch.no_grad()
def predict(model: torch.nn.Module, graphs: list[Any]) -> list[dict[str, Any]]:
    model.eval()
    rows = []
    for g in graphs:
        prob = torch.softmax(model(g.x, g.edge_index, g.edge_attr), dim=-1)
        conf, pred = prob.max(dim=-1)
        for i, fid in enumerate(g.face_ids):
            rows.append(
                {
                    "sample_id": g.sample_id,
                    "face_id": fid,
                    "graph_node": i,
                    "ground_truth": LABELS[int(g.y[i])] if g.y is not None else None,
                    "predicted": LABELS[int(pred[i])],
                    "confidence": round(float(conf[i]), 4),
                }
            )
    return rows


def _report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    idx = {lab: i for i, lab in enumerate(LABELS)}
    return classification_report(
        [idx[r["ground_truth"]] for r in rows], [idx[r["predicted"]] for r in rows], LABELS
    )


def run_id_for(dataset_id: str, cfg: TrainConfig) -> str:
    raw = json.dumps(
        {"dataset": dataset_id, "cfg": asdict(cfg), "pipeline": PIPELINE_VERSION}, sort_keys=True
    )
    return "tr_" + hashlib.sha256(raw.encode()).hexdigest()[:16]


def run_training(store: ArtifactStore, dataset_id: str, cfg: TrainConfig) -> dict[str, Any]:
    run_id = run_id_for(dataset_id, cfg)
    t_start = time.perf_counter()
    seed_everything(cfg.seed, cfg.threads)
    ds = load_dataset_manifest(store, dataset_id)
    raw = load_graphs(store, dataset_id)
    std = fit_standardizer(raw["train"])
    data = {k: apply_standardizer(v, std) for k, v in raw.items()}
    in_dim, edge_dim = data["train"][0].x.shape[1], data["train"][0].edge_attr.shape[1]
    model = build_model(cfg, in_dim, edge_dim)
    y = torch.cat([g.y for g in data["train"]])
    counts = torch.bincount(y, minlength=len(LABELS)).float()
    weights = torch.where(counts > 0, 1.0 / torch.sqrt(counts.clamp(min=1)), torch.zeros_like(counts))
    weights = weights / weights[counts > 0].mean()
    lossf = torch.nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    gen = torch.Generator().manual_seed(cfg.seed)
    loader = DataLoader(data["train"], batch_size=cfg.batch_size, shuffle=True, generator=gen)
    best = (-1.0, -1)
    best_state: dict[str, Any] = {}
    history = []
    for epoch in range(cfg.epochs):
        model.train()
        total = 0.0
        for batch in loader:
            opt.zero_grad()
            loss = lossf(model(batch.x, batch.edge_index, batch.edge_attr), batch.y)
            loss.backward()
            opt.step()
            total += float(loss) * batch.num_graphs
        val = _report(predict(model, data["val"]))["summary"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(total / len(data["train"]), 5),
                "val_macro_f1": val["macro_f1_present_classes"],
                "val_accuracy": val["accuracy"],
            }
        )
        if val["macro_f1_present_classes"] > best[0]:
            best = (val["macro_f1_present_classes"], epoch)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    preds = {split: predict(model, data[split]) for split in ("train", "val", "test")}
    metrics = {split: _report(rows) for split, rows in preds.items()}
    buf = io.BytesIO()
    torch.save(
        {
            "state_dict": best_state,
            "standardizer": std.to_dict(),
            "config": asdict(cfg),
            "in_dim": in_dim,
            "edge_dim": edge_dim,
            "labels": list(LABELS),
        },
        buf,
    )
    base = f"training_runs/{run_id}"
    store.put_bytes(f"{base}/checkpoint.pt", buf.getvalue())
    store.put_bytes(f"{base}/predictions.json", json.dumps(preds, indent=0).encode())
    experiment = {
        "run_id": run_id,
        "dataset_id": dataset_id,
        "config": asdict(cfg),
        "labels": list(LABELS),
        "best_epoch": best[1],
        "selection": "max validation macro-F1 (present classes)",
        "versions": {"pipeline": PIPELINE_VERSION, "torch": torch.__version__, "git_sha": _git_sha()},
        "dataset_split_sizes": {k: len(v) for k, v in ds["splits"].items()},
        "class_weights": [round(float(w), 4) for w in weights],
        "metrics": metrics,
        "history": history,
        "duration_s": round(time.perf_counter() - t_start, 2),
        "device": "cpu",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "artifacts": {"checkpoint": f"{base}/checkpoint.pt", "predictions": f"{base}/predictions.json"},
    }
    store.put_bytes(f"{base}/experiment.json", json.dumps(experiment, indent=1).encode())
    return experiment


def load_model(store: ArtifactStore, run_id: str) -> tuple[torch.nn.Module, Standardizer]:
    ck = torch.load(io.BytesIO(store.get_bytes(f"training_runs/{run_id}/checkpoint.pt")), weights_only=True)
    cfg = TrainConfig(**ck["config"])
    model = build_model(cfg, ck["in_dim"], ck["edge_dim"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, Standardizer.from_dict(ck["standardizer"])


def infer_sample(store: ArtifactStore, run_id: str, sample_id: str) -> list[dict[str, Any]]:
    """Predict face labels for any completed sample (no ground truth required)."""
    from torch_geometric.data import Data

    model, std = load_model(store, run_id)
    with np.load(io.BytesIO(store.get_bytes(f"samples/{sample_id}/graph.npz")), allow_pickle=False) as g:
        d = Data(
            x=torch.from_numpy(g["face_features"].copy()),
            edge_index=torch.from_numpy(g["edge_index"].copy()),
            edge_attr=torch.from_numpy(g["adjacency_features"].copy()),
            y=None,
            sample_id=sample_id,
            face_ids=[str(x) for x in g["node_face_ids"]],
        )
    (d,) = apply_standardizer([d], std)
    return predict(model, [d])
