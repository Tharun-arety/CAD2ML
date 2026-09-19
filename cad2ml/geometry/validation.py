"""Geometry validation and controlled, recorded repair.

Policy:
1. Analyse the transferred shape (BRepCheck, closed shells, orientation, volume sign).
2. If invalid and repair is enabled, try in order: orientation fix, sewing of open
   face sets into a closed solid, ``ShapeFix_Shape``. Every attempted operation is
   recorded; nothing is changed silently.
3. Measure the deviation between original and repaired geometry (sampled point
   distances + relative volume change). If it exceeds configured tolerances, or the
   result is still invalid, the part is quarantined instead of forced through.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from OCP.BRep import BRep_Tool
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeSolid, BRepBuilderAPI_MakeVertex, BRepBuilderAPI_Sewing
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.gp import gp_Pnt
from OCP.ShapeFix import ShapeFix_Shape
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_OUT, TopAbs_SHELL, TopAbs_SOLID, TopAbs_VERTEX
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS, TopoDS_Shape

from cad2ml.config import RepairConfig
from cad2ml.errors import PipelineError
from cad2ml.geometry import occ


@dataclass
class ShapeDiagnosis:
    valid: bool
    solid_count: int
    shell_count: int
    face_count: int
    closed_shells: bool
    volume: float
    orientation_ok: bool
    warnings: list[str] = field(default_factory=list)


def diagnose(shape: TopoDS_Shape) -> ShapeDiagnosis:
    solids = occ.index_map(shape, TopAbs_SOLID)
    shells = occ.index_map(shape, TopAbs_SHELL)
    faces = occ.index_map(shape, TopAbs_FACE)
    valid = bool(BRepCheck_Analyzer(shape).IsValid())
    closed = shells.Extent() > 0 and all(
        BRep_Tool.IsClosed_s(TopoDS.Shell_s(shells.FindKey(i))) for i in range(1, shells.Extent() + 1)
    )
    volume = occ.volume_props(shape)[0] if solids.Extent() else 0.0
    orientation_ok = False
    warnings: list[str] = []
    if solids.Extent() == 1:
        solid = TopoDS.Solid_s(solids.FindKey(1))
        clf = BRepClass3d_SolidClassifier(solid)
        clf.PerformInfinitePoint(1e-7)
        orientation_ok = clf.State() == TopAbs_OUT and volume > 0
        if not orientation_ok:
            warnings.append("solid orientation inverted (infinite point not OUT or negative volume)")
    if not valid:
        warnings.append("BRepCheck_Analyzer reported invalid topology/geometry")
    if shells.Extent() and not closed:
        warnings.append("open shell detected")
    return ShapeDiagnosis(
        valid, solids.Extent(), shells.Extent(), faces.Extent(), closed, volume, orientation_ok, warnings
    )


def _sample_surface_points(shape: TopoDS_Shape, max_points: int = 400) -> np.ndarray:
    BRepMesh_IncrementalMesh(shape, 0.5, False, 0.5, True)
    pts: list[list[float]] = []
    fm = occ.index_map(shape, TopAbs_FACE)
    for i in range(1, fm.Extent() + 1):
        f = TopoDS.Face_s(fm.FindKey(i))
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(f, loc)
        if tri is None:
            continue
        trsf = loc.Transformation()
        for k in range(1, tri.NbNodes() + 1):
            pts.append(occ.pnt(tri.Node(k).Transformed(trsf)))
    arr = np.asarray(pts, dtype=np.float64)
    if len(arr) > max_points:
        idx = np.linspace(0, len(arr) - 1, max_points).astype(int)
        arr = arr[idx]
    return arr


def max_deviation(original: TopoDS_Shape, repaired: TopoDS_Shape) -> float:
    """Max distance from sampled points of the original surface to the repaired shape."""
    pts = _sample_surface_points(original)
    worst = 0.0
    for p in pts:
        v = BRepExtrema_DistShapeShape(BRepBuilderAPI_MakeVertex(gp_Pnt(*p)).Vertex(), repaired)
        if v.IsDone():
            worst = max(worst, float(v.Value()))
    return worst


@dataclass
class RepairOutcome:
    shape: TopoDS_Shape
    source_valid: bool
    closed_solid: bool
    orientation_ok: bool
    repair_attempted: bool
    repaired_valid: bool | None
    operations: list[str]
    max_deviation_mm: float | None
    volume_relative_change: float | None
    warnings: list[str]


def _count_free(shape: TopoDS_Shape, kind: Any, avoid: Any) -> int:
    exp = TopExp_Explorer(shape, kind, avoid)
    n = 0
    while exp.More():
        n += 1
        exp.Next()
    return n


def isolate_single_solid(shape: TopoDS_Shape) -> tuple[TopoDS_Shape, list[str]]:
    """If the shape holds exactly one solid plus auxiliary geometry (PMI/construction curves, reference
    surfaces, points), return the solid alone and describe what was ignored. Never silent."""
    solids = occ.index_map(shape, TopAbs_SOLID)
    if solids.Extent() != 1 or shape.ShapeType() == TopAbs_SOLID:
        return shape, []
    aux = {
        "shells": _count_free(shape, TopAbs_SHELL, TopAbs_SOLID),
        "faces": _count_free(shape, TopAbs_FACE, TopAbs_SHELL),
        "edges": _count_free(shape, TopAbs_EDGE, TopAbs_FACE),
        "vertices": _count_free(shape, TopAbs_VERTEX, TopAbs_EDGE),
    }
    solid = TopoDS.Solid_s(solids.FindKey(1))
    ignored = {k: v for k, v in aux.items() if v}
    if not ignored:
        return solid, []
    detail = ", ".join(f"{v} {k}" for k, v in ignored.items())
    return solid, [f"auxiliary geometry outside the solid ignored (not part of the part): {detail}"]


def validate_and_repair(shape: TopoDS_Shape, cfg: RepairConfig) -> RepairOutcome:
    shape, aux_warnings = isolate_single_solid(shape)
    d0 = diagnose(shape)
    d0.warnings[:0] = aux_warnings
    ok0 = d0.valid and d0.solid_count == 1 and d0.closed_shells and d0.orientation_ok
    if ok0:
        return RepairOutcome(shape, True, True, True, False, None, [], None, None, d0.warnings)
    if not cfg.enabled:
        code = "NO_SOLID" if d0.solid_count == 0 else "INVALID_GEOMETRY"
        raise PipelineError(code, "; ".join(d0.warnings) or "invalid", "normalizing")

    ops: list[str] = []
    work: Any = shape
    if d0.solid_count == 0:
        if d0.face_count == 0:
            raise PipelineError("NO_SOLID", "no faces to build a solid from", "normalizing")
        sew = BRepBuilderAPI_Sewing(cfg.fix_tolerance_mm)
        sew.Add(work)
        sew.Perform()
        sewn = sew.SewedShape()
        ops.append(f"BRepBuilderAPI_Sewing(tol={cfg.fix_tolerance_mm})")
        shells = occ.index_map(sewn, TopAbs_SHELL)
        if shells.Extent() != 1 or not BRep_Tool.IsClosed_s(TopoDS.Shell_s(shells.FindKey(1))):
            raise PipelineError(
                "NO_SOLID",
                f"surface/shell model is not closed after sewing (free edges: "
                f"{sew.NbFreeEdges()}); refusing to fabricate a solid",
                "normalizing",
            )
        mk = BRepBuilderAPI_MakeSolid(TopoDS.Shell_s(shells.FindKey(1)))
        work = mk.Solid()
        ops.append("BRepBuilderAPI_MakeSolid(closed shell)")

    fixer = ShapeFix_Shape(work)
    fixer.SetPrecision(cfg.fix_tolerance_mm)
    fixer.SetMaxTolerance(cfg.fix_tolerance_mm * 10)
    fixer.Perform()
    fixed = fixer.Shape()
    ops.append(f"ShapeFix_Shape(precision={cfg.fix_tolerance_mm})")

    d1 = diagnose(fixed)
    if d1.solid_count == 1 and not d1.orientation_ok and d1.volume < 0:
        fixed = fixed.Reversed()
        ops.append("reverse_solid_orientation")
        d1 = diagnose(fixed)

    vol_change = None
    if d0.volume != 0 and d1.volume != 0:
        vol_change = abs(abs(d1.volume) - abs(d0.volume)) / abs(d0.volume)
    dev = max_deviation(shape, fixed)
    warnings = d0.warnings + [f"after repair: {w}" for w in d1.warnings]
    repaired_ok = d1.valid and d1.solid_count == 1 and d1.closed_shells and d1.orientation_ok
    if not repaired_ok:
        raise PipelineError(
            "INVALID_GEOMETRY", f"still invalid after {ops}: {'; '.join(d1.warnings)}", "normalizing"
        )
    if dev > cfg.max_deviation_mm or (vol_change is not None and vol_change > cfg.max_relative_volume_change):
        raise PipelineError(
            "REPAIR_DEVIATION_EXCEEDED", f"deviation {dev:.4g} mm, volume change {vol_change}", "normalizing"
        )
    return RepairOutcome(
        fixed,
        d0.valid,
        True,
        True,
        True,
        True,
        ops,
        round(dev, 6),
        None if vol_change is None else round(vol_change, 8),
        warnings,
    )
