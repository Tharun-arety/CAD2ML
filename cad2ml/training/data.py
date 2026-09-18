"""Load a dataset version into PyTorch Geometric ``Data`` objects (face classification)."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data

from cad2ml.storage.base import ArtifactStore


@dataclass
class Standardizer:
    x_mean: np.ndarray
    x_std: np.ndarray
    e_mean: np.ndarray
    e_std: np.ndarray

    def to_dict(self) -> dict[str, list[float]]:
        return {k: getattr(self, k).tolist() for k in ("x_mean", "x_std", "e_mean", "e_std")}

    @staticmethod
    def from_dict(d: dict[str, list[float]]) -> Standardizer:
        return Standardizer(
            *(np.asarray(d[k], dtype=np.float32) for k in ("x_mean", "x_std", "e_mean", "e_std"))
        )


def load_dataset_manifest(store: ArtifactStore, dataset_id: str) -> dict[str, Any]:
    return json.loads(store.get_bytes(f"datasets/{dataset_id}/dataset.json"))


def load_graphs(store: ArtifactStore, dataset_id: str) -> dict[str, list[Data]]:
    ds = load_dataset_manifest(store, dataset_id)
    labels = json.loads(store.get_bytes(f"datasets/{dataset_id}/labels.json"))
    out: dict[str, list[Data]] = {}
    for split, ids in ds["splits"].items():
        graphs = []
        for sid in ids:
            with np.load(io.BytesIO(store.get_bytes(f"samples/{sid}/graph.npz")), allow_pickle=False) as g:
                node_ids = [str(x) for x in g["node_face_ids"]]
                lab = labels[sid]
                if lab["face_ids"] != node_ids:
                    raise ValueError(f"label/graph face order mismatch for {sid}")
                graphs.append(
                    Data(
                        x=torch.from_numpy(g["face_features"].copy()),
                        edge_index=torch.from_numpy(g["edge_index"].copy()),
                        edge_attr=torch.from_numpy(g["adjacency_features"].copy()),
                        y=torch.tensor(lab["label_index"], dtype=torch.long),
                        sample_id=sid,
                        face_ids=node_ids,
                    )
                )
        out[split] = graphs
    return out


def fit_standardizer(graphs: list[Data]) -> Standardizer:
    x = torch.cat([g.x for g in graphs]).numpy()
    e = torch.cat([g.edge_attr for g in graphs]).numpy()
    return Standardizer(
        x.mean(0),
        np.where(x.std(0) > 1e-6, x.std(0), 1.0),
        e.mean(0),
        np.where(e.std(0) > 1e-6, e.std(0), 1.0),
    )


def apply_standardizer(graphs: list[Data], s: Standardizer) -> list[Data]:
    out = []
    for g in graphs:
        h = g.clone()
        h.x = (g.x - torch.from_numpy(s.x_mean)) / torch.from_numpy(s.x_std)
        h.edge_attr = (g.edge_attr - torch.from_numpy(s.e_mean)) / torch.from_numpy(s.e_std)
        out.append(h)
    return out
