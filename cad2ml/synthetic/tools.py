"""Analytic cutting-tool descriptors used as synthetic ground truth.

Every subtractive feature in the synthetic corpus is produced by cutting an
*extruded rounded rectangle* (which covers cylinders, stadium slots and rounded
pockets). Because the tool is stored analytically in the ground-truth sidecar, face
labels can later be assigned by testing whether the surface points of a processed
face lie on the tool boundary - independently of the rule-based recognizer and
without trusting in-memory CAD objects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray


def _unit(v: tuple[float, float, float] | list[float]) -> NDArray[np.float64]:
    a = np.asarray(v, dtype=np.float64)
    return a / np.linalg.norm(a)


@dataclass(frozen=True)
class ExtrudedRoundedRect:
    """Solid = {origin + a*u + b*v + t*n : (a,b) in rounded rect, t in [t0, t1]}.

    ``half_u``/``half_v`` are the outer half extents; ``corner_r`` <= min(half_u, half_v).
    cylinder: half_u = half_v = corner_r = radius; stadium slot: corner_r = half_v.
    """

    origin: tuple[float, float, float]
    u: tuple[float, float, float]
    n: tuple[float, float, float]
    half_u: float
    half_v: float
    corner_r: float
    t0: float
    t1: float

    @property
    def v(self) -> NDArray[np.float64]:
        return np.asarray(np.cross(_unit(self.n), _unit(self.u)), dtype=np.float64)

    def to_dict(self) -> dict[str, object]:
        return {"kind": "extruded_rounded_rect", **asdict(self)}

    @staticmethod
    def from_dict(d: dict[str, object]) -> ExtrudedRoundedRect:
        return ExtrudedRoundedRect(
            origin=tuple(d["origin"]),  # type: ignore[arg-type]
            u=tuple(d["u"]),  # type: ignore[arg-type]
            n=tuple(d["n"]),  # type: ignore[arg-type]
            half_u=float(d["half_u"]),  # type: ignore[arg-type]
            half_v=float(d["half_v"]),  # type: ignore[arg-type]
            corner_r=float(d["corner_r"]),  # type: ignore[arg-type]
            t0=float(d["t0"]),  # type: ignore[arg-type]
            t1=float(d["t1"]),  # type: ignore[arg-type]
        )

    def sdf(self, pts: NDArray[np.float64]) -> NDArray[np.float64]:
        """Signed distance (negative inside) for points [N,3]."""
        u, n = _unit(self.u), _unit(self.n)
        v = np.cross(n, u)
        rel = np.asarray(pts, dtype=np.float64) - np.asarray(self.origin)
        a, b, t = rel @ u, rel @ v, rel @ n
        r = self.corner_r
        qx = np.abs(a) - (self.half_u - r)
        qy = np.abs(b) - (self.half_v - r)
        d2 = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - r
        tc, ht = (self.t0 + self.t1) / 2, (self.t1 - self.t0) / 2
        dz = np.abs(t - tc) - ht
        return np.hypot(np.maximum(d2, 0), np.maximum(dz, 0)) + np.minimum(np.maximum(d2, dz), 0)

    def on_boundary(self, pts: NDArray[np.float64], tol: float) -> NDArray[np.bool_]:
        return np.abs(self.sdf(pts)) <= tol
