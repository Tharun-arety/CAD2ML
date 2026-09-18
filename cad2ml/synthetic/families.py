"""Parameterized synthetic part families (CadQuery).

Each generator returns ``(solid, ground_truth)``. Every subtractive feature is cut
with an explicit analytic tool (``ExtrudedRoundedRect``) whose descriptor is stored in
the ground truth, so the sidecar is authoritative and independent of the recognizer.

Label policy (face-level, used for training):
    planar  - planar faces not belonging to any feature
    hole    - walls and floors of through/blind holes
    slot    - walls/ends/floor of slots
    pocket  - walls/floor/corner faces of pockets
    fillet  - blend faces created by fillet operations
    other   - any remaining face (e.g. convex outer cylinders)

Hole radii are kept distinct from fillet radii so fillet faces can be matched
analytically (radius + surface type) without ambiguity.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cad2ml.synthetic.tools import ExtrudedRoundedRect

HOLE_DIAMETERS = (4.0, 5.0, 6.5, 8.5, 10.5, 13.0)
FILLET_RADII = (1.0, 1.5, 3.5)  # never equal to any hole/slot radius used below
SLOT_WIDTHS = (5.0, 6.5, 8.5)

FaceLabel = str
LABELS: tuple[FaceLabel, ...] = ("planar", "hole", "slot", "pocket", "fillet", "other")


@dataclass
class GroundTruth:
    family: str
    variant: str
    seed: int
    params: dict[str, Any]
    features: list[dict[str, Any]] = field(default_factory=list)
    fillet_radii: list[float] = field(default_factory=list)

    def add_hole(
        self,
        kind: str,
        center: tuple[float, float, float],
        axis: tuple[float, float, float],
        diameter: float,
        t0: float,
        t1: float,
        depth: float | None,
    ) -> ExtrudedRoundedRect:
        n = np.asarray(axis, float)
        helper = (1.0, 0.0, 0.0) if abs(n[0]) < 0.9 else (0.0, 1.0, 0.0)
        u = tuple(float(x) for x in np.cross(n, helper) / np.linalg.norm(np.cross(n, helper)))
        r = diameter / 2
        tool = ExtrudedRoundedRect(center, u, axis, r, r, r, t0, t1)  # type: ignore[arg-type]
        self.features.append(
            {
                "feature_type": kind,
                "label": "hole",
                "tool": tool.to_dict(),
                "parameters": {
                    "diameter_mm": diameter,
                    "axis": list(axis),
                    "center_mm": list(center),
                    **({"depth_mm": depth} if depth is not None else {}),
                },
            }
        )
        return tool

    def add_region(
        self, kind: str, label: str, tool: ExtrudedRoundedRect, parameters: dict[str, Any]
    ) -> ExtrudedRoundedRect:
        self.features.append(
            {"feature_type": kind, "label": label, "tool": tool.to_dict(), "parameters": parameters}
        )
        return tool

    def add_fillet(self, radius: float, where: str) -> None:
        self.fillet_radii.append(radius)
        self.features.append(
            {
                "feature_type": "fillet",
                "label": "fillet",
                "tool": None,
                "parameters": {"radius_mm": radius, "location": where},
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "variant": self.variant,
            "seed": self.seed,
            "params": self.params,
            "features": self.features,
            "fillet_radii": self.fillet_radii,
            "label_vocabulary": list(LABELS),
        }


def _cq() -> Any:
    import cadquery as cq

    return cq


def _tool_solid(tool: ExtrudedRoundedRect) -> Any:
    """Build the OCCT solid for an analytic tool."""
    cq = _cq()
    u = np.asarray(tool.u, float)
    n = np.asarray(tool.n, float)
    origin = np.asarray(tool.origin, float) + n * tool.t0
    plane = cq.Plane(origin=tuple(origin), xDir=tuple(u), normal=tuple(n))
    height = tool.t1 - tool.t0
    wp = cq.Workplane(plane)
    w, h, r = 2 * tool.half_u, 2 * tool.half_v, tool.corner_r
    if abs(tool.half_u - r) < 1e-9 and abs(tool.half_v - r) < 1e-9:
        return wp.circle(r).extrude(height).val()
    if abs(tool.half_v - r) < 1e-9:
        return wp.slot2D(w, h, 0).extrude(height).val()
    if r <= 1e-9:
        return wp.rect(w, h).extrude(height).val()
    return wp.sketch().rect(w, h).vertices().fillet(r).finalize().extrude(height).val()


def _cut(solid: Any, tool: ExtrudedRoundedRect) -> Any:
    return solid.cut(_tool_solid(tool))


def _pick(rng: np.random.Generator, seq: tuple[Any, ...] | list[Any]) -> Any:
    return seq[int(rng.integers(0, len(seq)))]


def _fillet_edge_near(solid: Any, point: tuple[float, float, float], radius: float) -> Any:
    cq = _cq()
    wp = cq.Workplane("XY").add(solid).edges(cq.selectors.NearestToPointSelector(point))
    return wp.fillet(radius).val()


def _fillet_vertical_edges(solid: Any, radius: float) -> Any:
    cq = _cq()
    return cq.Workplane("XY").add(solid).edges("|Z").fillet(radius).val()


# --------------------------------------------------------------------------- families


def slotted_plate(rng: np.random.Generator, seed: int, variant: str) -> tuple[Any, GroundTruth]:
    cq = _cq()
    L = float(rng.uniform(90, 160))
    W = float(rng.uniform(50, 90))
    T = float(_pick(rng, (5.0, 6.0, 8.0, 10.0)))
    gt = GroundTruth("slotted_plate", variant, seed, {"length_mm": L, "width_mm": W, "thickness_mm": T})
    solid = cq.Solid.makeBox(L, W, T)
    if variant in ("slots_fillet",):
        rf = float(_pick(rng, FILLET_RADII))
        solid = _fillet_vertical_edges(solid, rf)
        gt.add_fillet(rf, "vertical plate corners")
    d = float(_pick(rng, HOLE_DIAMETERS[:4]))
    margin = max(8.0, d)
    # holes at the four corners
    for x, y in [(margin, margin), (L - margin, margin), (margin, W - margin), (L - margin, W - margin)]:
        solid = _cut(solid, gt.add_hole("through_hole", (x, y, 0.0), (0, 0, 1), d, -1.0, T + 1.0, None))
    gt.params["corner_hole_diameter_mm"] = d
    if variant in ("slots", "slots_fillet"):
        n_slots = int(rng.integers(1, 3))
        sw = float(_pick(rng, SLOT_WIDTHS))
        usable = W - 2 * (margin + d)
        pitch = usable / n_slots
        for i in range(n_slots):
            cy = margin + d + pitch * (i + 0.5)
            sl = float(rng.uniform(0.35, 0.55)) * L
            tool = ExtrudedRoundedRect(
                (L / 2, cy, 0.0), (1, 0, 0), (0, 0, 1), sl / 2, sw / 2, sw / 2, -1.0, T + 1.0
            )
            gt.add_region(
                "slot",
                "slot",
                tool,
                {
                    "width_mm": sw,
                    "length_mm": sl,
                    "depth_mm": T,
                    "through": "yes",
                    "center_mm": [L / 2, cy, T / 2],
                    "direction": [1.0, 0.0, 0.0],
                },
            )
            solid = _cut(solid, tool)
    else:
        dc = float(_pick(rng, HOLE_DIAMETERS[3:]))
        solid = _cut(
            solid, gt.add_hole("through_hole", (L / 2, W / 2, 0.0), (0, 0, 1), dc, -1.0, T + 1.0, None)
        )
    return solid, gt


def mounting_bracket(rng: np.random.Generator, seed: int, variant: str) -> tuple[Any, GroundTruth]:
    cq = _cq()
    L = float(rng.uniform(50, 100))
    W = float(rng.uniform(45, 70))
    H = float(rng.uniform(40, 70))
    T = float(_pick(rng, (5.0, 6.0, 8.0)))
    gt = GroundTruth(
        "mounting_bracket",
        variant,
        seed,
        {"length_mm": L, "base_width_mm": W, "height_mm": H, "thickness_mm": T},
    )
    base = cq.Solid.makeBox(L, W, T)
    upright = cq.Solid.makeBox(L, T, H)
    solid = base.fuse(upright).clean()
    rf = 0.0
    if variant in ("filleted", "slotted"):
        rf = float(_pick(rng, FILLET_RADII))
        solid = _fillet_edge_near(solid, (L / 2, T, T), rf)
        gt.add_fillet(rf, "inner bend")
    d = float(_pick(rng, HOLE_DIAMETERS[:4]))
    gt.params["hole_diameter_mm"] = d
    y_base = T + rf + (W - T - rf) * 0.62
    xs = [L * 0.25, L * 0.75]
    if variant == "slotted":
        sw = float(_pick(rng, SLOT_WIDTHS))
        sl = L * 0.5
        tool = ExtrudedRoundedRect(
            (L / 2, y_base, 0.0), (1, 0, 0), (0, 0, 1), sl / 2, sw / 2, sw / 2, -1.0, T + 1.0
        )
        gt.add_region(
            "slot",
            "slot",
            tool,
            {
                "width_mm": sw,
                "length_mm": sl,
                "depth_mm": T,
                "through": "yes",
                "center_mm": [L / 2, y_base, T / 2],
                "direction": [1.0, 0.0, 0.0],
            },
        )
        solid = _cut(solid, tool)
    else:
        for x in xs:
            solid = _cut(
                solid, gt.add_hole("through_hole", (x, y_base, 0.0), (0, 0, 1), d, -1.0, T + 1.0, None)
            )
    z_up = T + rf + (H - T - rf) * 0.6
    for x in xs:
        solid = _cut(solid, gt.add_hole("through_hole", (x, 0.0, z_up), (0, 1, 0), d, -1.0, T + 1.0, None))
    return solid, gt


def flange(rng: np.random.Generator, seed: int, variant: str) -> tuple[Any, GroundTruth]:
    cq = _cq()
    R = float(rng.uniform(35, 60))
    T = float(_pick(rng, (8.0, 10.0, 12.0)))
    bore = float(_pick(rng, (13.0, 17.0, 21.0)))
    gt = GroundTruth(
        "flange", variant, seed, {"outer_radius_mm": R, "thickness_mm": T, "bore_diameter_mm": bore}
    )
    solid = cq.Solid.makeCylinder(R, T)
    top = T
    if variant == "hub":
        Rh = bore / 2 + float(rng.uniform(6, 10))
        Hh = float(rng.uniform(8, 15))
        solid = solid.fuse(cq.Solid.makeCylinder(Rh, Hh, cq.Vector(0, 0, T))).clean()
        rf = float(_pick(rng, FILLET_RADII))
        root = [
            e
            for e in solid.Edges()
            if e.geomType() == "CIRCLE" and abs(e.radius() - Rh) < 1e-6 and abs(e.Center().z - T) < 1e-6
        ]
        solid = solid.fillet(rf, root)
        gt.add_fillet(rf, "hub root")
        gt.params.update({"hub_radius_mm": Rh, "hub_height_mm": Hh})
        top = T + Hh
    solid = _cut(solid, gt.add_hole("through_hole", (0.0, 0.0, 0.0), (0, 0, 1), bore, -1.0, top + 1.0, None))
    n = int(rng.integers(3, 9))
    d = float(_pick(rng, HOLE_DIAMETERS[:3]))
    inner = bore / 2 + (gt.params.get("hub_radius_mm", bore / 2) - bore / 2)
    pcd_r = (max(inner, bore / 2) + d + R - d) / 2
    gt.params.update({"bolt_count": n, "bolt_hole_diameter_mm": d, "bolt_circle_radius_mm": pcd_r})
    blind = variant == "blind"
    depth = T * 0.6
    phase = float(rng.uniform(0, math.pi / n))
    for i in range(n):
        a = phase + 2 * math.pi * i / n
        c = (pcd_r * math.cos(a), pcd_r * math.sin(a), 0.0)
        if blind:
            tool = gt.add_hole("blind_hole", c, (0, 0, 1), d, T - depth, T + 1.0, depth)
        else:
            tool = gt.add_hole("through_hole", c, (0, 0, 1), d, -1.0, T + 1.0, None)
        solid = _cut(solid, tool)
    return solid, gt


def shaft_support(rng: np.random.Generator, seed: int, variant: str) -> tuple[Any, GroundTruth]:
    cq = _cq()
    L = float(rng.uniform(90, 140))
    W = float(rng.uniform(30, 45))
    Tb = float(_pick(rng, (8.0, 10.0, 12.0)))
    Wu = float(rng.uniform(36, 50))
    # keep the rounded cap (radius Wu/2 centred at H - Wu/2) above the base top (found by bbox evaluation)
    H = float(rng.uniform(max(40.0, Wu + Tb + 2.0), max(60.0, Wu + Tb + 12.0)))
    bore = float(_pick(rng, (17.0, 21.0, 25.0)))
    gt = GroundTruth(
        "shaft_support",
        variant,
        seed,
        {
            "length_mm": L,
            "width_mm": W,
            "base_thickness_mm": Tb,
            "upright_width_mm": Wu,
            "height_mm": H,
            "bore_diameter_mm": bore,
        },
    )
    base = cq.Solid.makeBox(L, W, Tb)
    if variant == "filleted":
        rf = float(_pick(rng, FILLET_RADII))
        base = _fillet_vertical_edges(base, rf)
        gt.add_fillet(rf, "base corners")
    x0 = (L - Wu) / 2
    zc = H - Wu / 2
    if variant == "rounded":
        body = cq.Solid.makeBox(Wu, W, zc).translate(cq.Vector(x0, 0, 0))
        cap = cq.Solid.makeCylinder(Wu / 2, W, cq.Vector(L / 2, 0, zc), cq.Vector(0, 1, 0))
        upright = body.fuse(cap).clean()
    else:
        upright = cq.Solid.makeBox(Wu, W, H).translate(cq.Vector(x0, 0, 0))
    solid = base.fuse(upright).clean()
    solid = _cut(solid, gt.add_hole("through_hole", (L / 2, 0.0, zc), (0, 1, 0), bore, -1.0, W + 1.0, None))
    d = float(_pick(rng, HOLE_DIAMETERS[:4]))
    gt.params["mount_hole_diameter_mm"] = d
    for x in (x0 / 2, L - x0 / 2):
        solid = _cut(solid, gt.add_hole("through_hole", (x, W / 2, 0.0), (0, 0, 1), d, -1.0, Tb + 1.0, None))
    return solid, gt


def housing(rng: np.random.Generator, seed: int, variant: str) -> tuple[Any, GroundTruth]:
    cq = _cq()
    L = float(rng.uniform(70, 120))
    W = float(rng.uniform(50, 90))
    H = float(rng.uniform(25, 45))
    rim = float(rng.uniform(12, 16))
    depth = H * float(rng.uniform(0.45, 0.7))
    rc = float(_pick(rng, (2.5, 4.0, 6.0)))
    gt = GroundTruth(
        "housing",
        variant,
        seed,
        {
            "length_mm": L,
            "width_mm": W,
            "height_mm": H,
            "rim_mm": rim,
            "pocket_depth_mm": depth,
            "pocket_corner_radius_mm": rc,
        },
    )
    solid = cq.Solid.makeBox(L, W, H)
    if variant == "filleted":
        rf = float(_pick(rng, FILLET_RADII))
        solid = _fillet_vertical_edges(solid, rf)
        gt.add_fillet(rf, "outer vertical edges")
    tool = ExtrudedRoundedRect(
        (L / 2, W / 2, 0.0), (1, 0, 0), (0, 0, 1), L / 2 - rim, W / 2 - rim, rc, H - depth, H + 1.0
    )
    gt.add_region(
        "pocket",
        "pocket",
        tool,
        {
            "length_mm": L - 2 * rim,
            "width_mm": W - 2 * rim,
            "depth_mm": depth,
            "corner_radius_mm": rc,
            "floor_z_mm": H - depth,
        },
    )
    solid = _cut(solid, tool)
    d = float(_pick(rng, HOLE_DIAMETERS[:2]))
    gt.params["rim_hole_diameter_mm"] = d
    for x, y in [
        (rim / 2, rim / 2),
        (L - rim / 2, rim / 2),
        (rim / 2, W - rim / 2),
        (L - rim / 2, W - rim / 2),
    ]:
        solid = _cut(solid, gt.add_hole("through_hole", (x, y, 0.0), (0, 0, 1), d, -1.0, H + 1.0, None))
    if variant == "blind_holes":
        db = float(_pick(rng, HOLE_DIAMETERS[:3]))
        bd = (H - depth) * 0.5
        floor = H - depth
        for x in (L / 2 - (L / 2 - rim) / 2, L / 2 + (L / 2 - rim) / 2):
            solid = _cut(
                solid, gt.add_hole("blind_hole", (x, W / 2, 0.0), (0, 0, 1), db, floor - bd, floor + 0.5, bd)
            )
    return solid, gt


def clevis(rng: np.random.Generator, seed: int, variant: str) -> tuple[Any, GroundTruth]:
    cq = _cq()
    L = float(rng.uniform(40, 60))  # x
    W = float(rng.uniform(25, 40))  # y
    H = float(rng.uniform(45, 70))  # z
    gap = float(rng.uniform(0.3, 0.45)) * L
    Hb = H * float(rng.uniform(0.35, 0.5))
    gt = GroundTruth(
        "clevis",
        variant,
        seed,
        {"length_mm": L, "width_mm": W, "height_mm": H, "gap_mm": gap, "base_height_mm": Hb},
    )
    solid = cq.Solid.makeBox(L, W, H)
    tool = ExtrudedRoundedRect(
        (L / 2, -1.0, (Hb + H + 1.0) / 2),
        (1, 0, 0),
        (0, 1, 0),
        gap / 2,
        (H + 1.0 - Hb) / 2,
        0.0,
        0.0,
        W + 2.0,
    )
    gt.add_region(
        "slot",
        "slot",
        tool,
        {
            "width_mm": gap,
            "length_mm": W,
            "depth_mm": H - Hb,
            "through": "open",
            "center_mm": [L / 2, W / 2, (Hb + H) / 2],
            "direction": [0.0, 1.0, 0.0],
        },
    )
    solid = _cut(solid, tool)
    if variant == "filleted":
        rf = float(_pick(rng, FILLET_RADII))
        for x in (L / 2 - gap / 2, L / 2 + gap / 2):
            solid = _fillet_edge_near(solid, (x, W / 2, Hb), rf)
        gt.add_fillet(rf, "fork root")
    dp = float(_pick(rng, HOLE_DIAMETERS[2:5]))
    zp = Hb + (H - Hb) * 0.6
    gt.params["pin_diameter_mm"] = dp
    solid = _cut(solid, gt.add_hole("through_hole", (0.0, W / 2, zp), (1, 0, 0), dp, -1.0, L + 1.0, None))
    if variant == "base_holes":
        d = float(_pick(rng, HOLE_DIAMETERS[:2]))
        for x in (L * 0.2, L * 0.8):
            solid = _cut(
                solid, gt.add_hole("blind_hole", (x, W / 2, 0.0), (0, 0, 1), d, -1.0, Hb * 0.6, Hb * 0.6)
            )
        gt.params["base_hole_diameter_mm"] = d
    return solid, gt


Generator = Callable[[np.random.Generator, int, str], tuple[Any, GroundTruth]]

FAMILIES: dict[str, tuple[Generator, tuple[str, ...]]] = {
    "mounting_bracket": (mounting_bracket, ("plain", "filleted", "slotted")),
    "flange": (flange, ("flat", "hub", "blind")),
    "shaft_support": (shaft_support, ("plain", "rounded", "filleted")),
    "slotted_plate": (slotted_plate, ("holes_only", "slots", "slots_fillet")),
    "housing": (housing, ("plain", "blind_holes", "filleted")),
    "clevis": (clevis, ("plain", "filleted", "base_holes")),
}
