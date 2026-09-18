"""Thin, typed-at-the-boundary helpers over OCP (OpenCascade) used by extraction.

All functions return plain Python / NumPy values so the rest of the codebase does not
depend on OCP object lifetimes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from OCP.Bnd import Bnd_Box
from OCP.BRep import BRep_Builder, BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepClass import BRepClass_FaceClassifier
from OCP.BRepGProp import BRepGProp
from OCP.BRepLProp import BRepLProp_SLProps
from OCP.BRepTools import BRepTools
from OCP.GeomAbs import (
    GeomAbs_BezierCurve,
    GeomAbs_BezierSurface,
    GeomAbs_BSplineCurve,
    GeomAbs_BSplineSurface,
    GeomAbs_Circle,
    GeomAbs_Cone,
    GeomAbs_Cylinder,
    GeomAbs_Ellipse,
    GeomAbs_Hyperbola,
    GeomAbs_Line,
    GeomAbs_OffsetCurve,
    GeomAbs_OffsetSurface,
    GeomAbs_OtherCurve,
    GeomAbs_OtherSurface,
    GeomAbs_Parabola,
    GeomAbs_Plane,
    GeomAbs_Sphere,
    GeomAbs_SurfaceOfExtrusion,
    GeomAbs_SurfaceOfRevolution,
    GeomAbs_Torus,
)
from OCP.GeomAPI import GeomAPI_ProjectPointOnSurf
from OCP.gp import gp_Pnt, gp_Pnt2d
from OCP.GProp import GProp_GProps
from OCP.TopAbs import TopAbs_IN, TopAbs_ON, TopAbs_REVERSED
from OCP.TopoDS import TopoDS_Shape
from OCP.TopTools import TopTools_IndexedMapOfShape

SURFACE_TYPES: dict[Any, str] = {
    GeomAbs_Plane: "plane",
    GeomAbs_Cylinder: "cylinder",
    GeomAbs_Cone: "cone",
    GeomAbs_Sphere: "sphere",
    GeomAbs_Torus: "torus",
    GeomAbs_BezierSurface: "bezier",
    GeomAbs_BSplineSurface: "bspline",
    GeomAbs_SurfaceOfRevolution: "revolution",
    GeomAbs_SurfaceOfExtrusion: "extrusion",
    GeomAbs_OffsetSurface: "offset",
    GeomAbs_OtherSurface: "other",
}
SURFACE_TYPE_ORDER = [
    "plane",
    "cylinder",
    "cone",
    "sphere",
    "torus",
    "bspline",
    "bezier",
    "revolution",
    "extrusion",
    "offset",
    "other",
]
CURVE_TYPES: dict[Any, str] = {
    GeomAbs_Line: "line",
    GeomAbs_Circle: "circle",
    GeomAbs_Ellipse: "ellipse",
    GeomAbs_Hyperbola: "hyperbola",
    GeomAbs_Parabola: "parabola",
    GeomAbs_BezierCurve: "bezier",
    GeomAbs_BSplineCurve: "bspline",
    GeomAbs_OffsetCurve: "offset",
    GeomAbs_OtherCurve: "other",
}
CURVE_TYPE_ORDER = [
    "line",
    "circle",
    "ellipse",
    "hyperbola",
    "parabola",
    "bspline",
    "bezier",
    "offset",
    "other",
    "degenerate",
]


def pnt(p: gp_Pnt) -> list[float]:
    return [p.X(), p.Y(), p.Z()]


def vec(d: Any) -> list[float]:
    return [d.X(), d.Y(), d.Z()]


def read_brep(path: str) -> TopoDS_Shape:
    shape = TopoDS_Shape()
    ok = BRepTools.Read_s(shape, path, BRep_Builder())
    if not ok or shape.IsNull():
        raise OSError(f"failed to read brep {path}")
    return shape


def write_brep(shape: TopoDS_Shape, path: str) -> None:
    if not BRepTools.Write_s(shape, path):
        raise OSError(f"failed to write brep {path}")


def index_map(shape: TopoDS_Shape, kind: Any) -> TopTools_IndexedMapOfShape:
    from OCP.TopExp import TopExp

    m = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, kind, m)
    return m


def iter_list(lst: Any) -> list[Any]:
    return list(lst)


def bbox(shape: TopoDS_Shape, optimal: bool = True) -> tuple[list[float], list[float]]:
    b = Bnd_Box()
    if optimal:
        BRepBndLib.AddOptimal_s(shape, b, False, False)
    else:
        BRepBndLib.Add_s(shape, b, True)
    xmin, ymin, zmin, xmax, ymax, zmax = b.Get()
    return [xmin, ymin, zmin], [xmax, ymax, zmax]


def volume_props(shape: TopoDS_Shape) -> tuple[float, list[float]]:
    p = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, p)
    return p.Mass(), pnt(p.CentreOfMass())


def surface_props(shape: TopoDS_Shape) -> tuple[float, list[float]]:
    p = GProp_GProps()
    BRepGProp.SurfaceProperties_s(shape, p)
    return p.Mass(), pnt(p.CentreOfMass())


def linear_length(shape: TopoDS_Shape) -> float:
    p = GProp_GProps()
    BRepGProp.LinearProperties_s(shape, p)
    return float(p.Mass())


def uv_bounds(face: Any) -> list[float]:
    return [float(x) for x in BRepTools.UVBounds_s(face)]


def is_reversed(shape: Any) -> bool:
    return bool(shape.Orientation() == TopAbs_REVERSED)


def surface_type(face: Any) -> str:
    return SURFACE_TYPES.get(BRepAdaptor_Surface(face, True).GetType(), "other")


def curve_type(edge: Any) -> str:
    if BRep_Tool.Degenerated_s(edge):
        return "degenerate"
    return CURVE_TYPES.get(BRepAdaptor_Curve(edge).GetType(), "other")


def face_uv_of_point(face: Any, xyz: np.ndarray | list[float]) -> tuple[float, float, float]:
    """Project a 3D point onto the face's underlying surface. Returns (u, v, distance)."""
    surf = BRep_Tool.Surface_s(face)
    proj = GeomAPI_ProjectPointOnSurf(gp_Pnt(float(xyz[0]), float(xyz[1]), float(xyz[2])), surf)
    if proj.NbPoints() == 0:
        raise ValueError("projection failed")
    u, v = proj.LowerDistanceParameters()
    return float(u), float(v), float(proj.LowerDistance())


class FaceEvaluator:
    """Oriented normal and curvature evaluation on one face (material-outward normals)."""

    def __init__(self, face: Any) -> None:
        self.face = face
        self.adaptor = BRepAdaptor_Surface(face, True)
        self.surface = BRep_Tool.Surface_s(face)
        self.sign = -1.0 if is_reversed(face) else 1.0
        self.props = BRepLProp_SLProps(self.adaptor, 2, 1e-7)

    def at_uv(self, u: float, v: float) -> tuple[list[float], list[float] | None, float, float]:
        """Returns (point, oriented unit normal or None, mean curvature, gaussian curvature).

        Curvature sign follows the oriented normal: positive mean curvature = convex.
        """
        self.props.SetParameters(u, v)
        p = pnt(self.props.Value())
        if not self.props.IsNormalDefined():
            return p, None, float("nan"), float("nan")
        n = self.props.Normal()
        normal = [self.sign * n.X(), self.sign * n.Y(), self.sign * n.Z()]
        if self.props.IsCurvatureDefined():
            # OCCT curvature is w.r.t. the surface normal; convert to "convex positive" w.r.t. outward normal.
            mean = -self.sign * self.props.MeanCurvature()
            gauss = self.props.GaussianCurvature()
        else:
            mean, gauss = float("nan"), float("nan")
        return p, normal, float(mean), float(gauss)

    def project(self, xyz: np.ndarray | list[float]) -> tuple[float, float, float]:
        proj = GeomAPI_ProjectPointOnSurf(gp_Pnt(float(xyz[0]), float(xyz[1]), float(xyz[2])), self.surface)
        if proj.NbPoints() == 0:
            raise ValueError("projection failed")
        u, v = proj.LowerDistanceParameters()
        return float(u), float(v), float(proj.LowerDistance())

    def contains_uv(self, u: float, v: float, tol: float = 1e-6) -> bool:
        c = BRepClass_FaceClassifier(self.face, gp_Pnt2d(u, v), tol)
        return c.State() in (TopAbs_IN, TopAbs_ON)
