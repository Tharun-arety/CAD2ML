"""Baseline face classifiers.

* ``FaceMLP`` - per-face MLP on node features only (no graph context): the non-graph baseline.
* ``FaceGNN`` - 3-layer GINE network using node features and adjacency (edge convexity,
  dihedral angle, ...) features. Intentionally small; the goal is dataset usability.
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GINEConv


class FaceMLP(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FaceGNN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        edge_dim: int,
        n_classes: int,
        hidden: int = 64,
        layers: int = 3,
        dropout: float = 0.1,
        aggr: str = "mean",
    ) -> None:
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList(
            GINEConv(
                nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)),
                edge_dim=edge_dim,
                aggr=aggr,
            )
            for _ in range(layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, n_classes))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        h0 = torch.relu(self.inp(x))
        h = h0
        for conv, norm in zip(self.convs, self.norms, strict=True):
            h = h + self.drop(torch.relu(norm(conv(h, edge_index, edge_attr))))
        return self.head(torch.cat([h, h0], dim=-1))
