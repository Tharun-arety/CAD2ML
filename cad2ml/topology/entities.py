"""Canonical B-Rep face/edge extraction with deterministic IDs.

Produces ``BRepModel``: ordered OCCT face/edge handles (in canonical order) plus
schema records. Kernel handles never leave the isolated extraction process.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCP.BRepTools import BRepTools
from OCP.gp import gp_Pnt, gp_Vec
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_WIRE
from OCP.TopExp import TopExp, TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Shape
from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape

from cad2ml.geometry import occ
from cad2ml.schemas.manifest import EdgeRecord, FaceRecord
from cad2ml.topology import canonical as C

SMOOTH_ANGLE_DEG = 2.0


def _r(x: float, nd: int = 6) -> float:
    return round(float(x), nd)


def _rv(v: Any, nd: int = 6) -> list[float]:
    return [_r(x, nd) for x in v]


def _analytic_params(face: Any) -> dict[str, float | list[float]]:
    ad = BRepAdaptor_Surface(face, True)
    t = occ.SURFACE_TYPES.get(ad.GetType(), "other")
    if t == "plane":
        pl = ad.Plane()
        ax = pl.Axis()
        return {"origin": _rv(occ.pnt(ax.Location())), "axis": _rv(occ.vec(ax.Direction()))}
    if t == "cylinder":
        cy = ad.Cylinder()
        ax = cy.Axis()
        return {
            "radius": _r(cy.Radius()),
            "axis_origin": _rv(occ.pnt(ax.Location())),
            "axis": _rv(occ.vec(ax.Direction())),
        }
    if t == "cone":
        co = ad.Cone()
        ax = co.Axis()
        return {
            "ref_radius": _r(co.RefRadius()),
            "semi_angle_rad": _r(co.SemiAngle()),
            "axis_origin": _rv(occ.pnt(ax.Location())),
            "axis": _rv(occ.vec(ax.Direction())),
            "apex": _rv(occ.pnt(co.Apex())),
        }
    if t == "sphere":
        sp = ad.Sphere()
        return {"radius": _r(sp.Radius()), "center": _rv(occ.pnt(sp.Location()))}
    if t == "torus":
        to = ad.Torus()
        ax = to.Axis()
        return {
            "major_radius": _r(to.MajorRadius()),
            "minor_radius": _r(to.MinorRadius()),
            "axis_origin": _rv(occ.pnt(ax.Location())),
            "axis": _rv(occ.vec(ax.Direction())),
        }
    return {}


def _curvature_stats(ev: occ.FaceEvaluator, uvb: list[float], n: int = 5) -> dict[str, float]:
    means, gauss = [], []
    for i in range(n):
        for j in range(n):
            u = uvb[0] + (uvb[1] - uvb[0]) * (i + 0.5) / n
            v = uvb[2] + (uvb[3] - uvb[2]) * (j + 0.5) / n
            if not ev.contains_uv(u, v):
                continue
            _, nrm, H, K = ev.at_uv(u, v)
            if nrm is not None and math.isfinite(H):
                means.append(H)
                gauss.append(K)
    if not means:
        return {"samples": 0.0}
    return {
        "samples": float(len(means)),
        "mean_curvature_mean": _r(float(np.mean(means))),
        "mean_curvature_min": _r(float(np.min(means))),
        "mean_curvature_max": _r(float(np.max(means))),
        "gaussian_curvature_mean": _r(float(np.mean(gauss))),
    }


def _normal_at_point(ev: occ.FaceEvaluator, xyz: list[float]) -> list[float] | None:
    try:
        u, v, _ = ev.project(xyz)
    except ValueError:
        return None
    _, n, _, _ = ev.at_uv(u, v)
    return n


@dataclass
class BRepModel:
    shape: TopoDS_Shape
    faces: list[Any]  # TopoDS_Face in canonical order
    edges: list[Any]  # TopoDS_Edge in canonical order
    face_records: list[FaceRecord]
    edge_records: list[EdgeRecord]
    face_index: dict[str, int]
    edge_index: dict[str, int]


def _edge_convexity(
    edge: Any, f1: Any, f2: Any, ev1: occ.FaceEvaluator, ev2: occ.FaceEvaluator, smooth_deg: float
) -> tuple[str, float | None]:
    """Classify an edge shared by two distinct faces using oriented normals at the midpoint."""
    curve = BRepAdaptor_Curve(edge)
    tm = 0.5 * (curve.FirstParameter() + curve.LastParameter())
    p, d1 = gp_Pnt(), gp_Vec()
    curve.D1(tm, p, d1)
    xyz = occ.pnt(p)
    n1, n2 = _normal_at_point(ev1, xyz), _normal_at_point(ev2, xyz)
    if n1 is None or n2 is None or d1.Magnitude() < 1e-12:
        return "unknown", None
    a1, a2 = np.asarray(n1), np.asarray(n2)
    cosang = float(np.clip(a1 @ a2, -1.0, 1.0))
    dihedral = math.degrees(math.acos(cosang))  # angle between outward normals; 0 = tangent
    if dihedral < smooth_deg:
        return "smooth", _r(180.0 - dihedral, 4)
    # orientation of this edge as used inside f1 (face orientation already composed by the explorer)
    t = np.array(occ.vec(d1))
    exp = TopExp_Explorer(f1, TopAbs_EDGE)
    found = None
    while exp.More():
        if exp.Current().IsSame(edge):
            found = exp.Current()
            break
        exp.Next()
    if found is None:
        return "unknown", None
    if occ.is_reversed(found):
        t = -t
    s = float(np.cross(a1, a2) @ t)
    interior = 180.0 - dihedral  # interior (material) angle for convex edges
    return ("convex", _r(interior, 4)) if s > 0 else ("concave", _r(180.0 + dihedral, 4))


def extract_entities(shape: TopoDS_Shape, smooth_deg: float = SMOOTH_ANGLE_DEG) -> BRepModel:
    fmap = occ.index_map(shape, TopAbs_FACE)
    emap = occ.index_map(shape, TopAbs_EDGE)
    e2f = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shape, TopAbs_EDGE, TopAbs_FACE, e2f)

    faces = [TopoDS.Face_s(fmap.FindKey(i)) for i in range(1, fmap.Extent() + 1)]
    edges = [TopoDS.Edge_s(emap.FindKey(i)) for i in range(1, emap.Extent() + 1)]
    evals = [occ.FaceEvaluator(f) for f in faces]

    # --- edge -> faces (by traversal index, distinct) ---------------------------------
    edge_faces: list[list[int]] = []
    edge_is_seam: list[bool] = []
    for e in edges:
        anc = occ.iter_list(e2f.FindFromKey(e)) if e2f.Contains(e) else []
        idx = [fmap.FindIndex(a) - 1 for a in anc]
        distinct = sorted(set(idx))
        edge_faces.append(distinct)
        edge_is_seam.append(len(idx) >= 2 and len(distinct) == 1)

    face_edges: list[set[int]] = [set() for _ in faces]
    face_adj: list[set[int]] = [set() for _ in faces]
    for ei, fl in enumerate(edge_faces):
        for fi in fl:
            face_edges[fi].add(ei)
        for a in fl:
            for b in fl:
                if a != b:
                    face_adj[a].add(b)

    # --- raw face attributes ----------------------------------------------------------
    raw: list[dict[str, Any]] = []
    for fi, f in enumerate(faces):
        stype = occ.surface_type(f)
        area, centroid = occ.surface_props(f)
        bmin, bmax = occ.bbox(f, optimal=True)
        uvb = occ.uv_bounds(f)
        analytic = _analytic_params(f)
        outer = BRepTools.OuterWire_s(f)
        outer_edges = set()
        if not outer.IsNull():
            ow = TopExp_Explorer(outer, TopAbs_EDGE)
            while ow.More():
                outer_edges.add(emap.FindIndex(ow.Current()) - 1)
                ow.Next()
        loops = 0
        w = TopExp_Explorer(f, TopAbs_WIRE)
        while w.More():
            loops += 1
            w.Next()
        ev = evals[fi]
        normal = _normal_at_point(ev, centroid)
        if normal is None:  # e.g. centroid on a cylinder axis: use an interior UV sample instead
            for fu, fv in ((0.5, 0.5), (0.25, 0.5), (0.5, 0.25), (0.75, 0.75)):
                u, v = uvb[0] + fu * (uvb[1] - uvb[0]), uvb[2] + fv * (uvb[3] - uvb[2])
                if ev.contains_uv(u, v):
                    _, normal, _, _ = ev.at_uv(u, v)
                    if normal is not None:
                        break
        raw.append(
            {
                "outer": outer_edges,
                "type": stype,
                "area": area,
                "centroid": centroid,
                "bmin": bmin,
                "bmax": bmax,
                "uvb": uvb,
                "analytic": analytic,
                "loops": loops,
                "normal": normal,
                "curv": _curvature_stats(ev, uvb),
                "intrinsic": C.intrinsic_face_signature(stype, area, analytic, loops, len(face_edges[fi])),
            }
        )
    fps = [
        C.face_fingerprint(r["intrinsic"], [raw[j]["type"] for j in face_adj[i]]) for i, r in enumerate(raw)
    ]
    rank = {t: i for i, t in enumerate(occ.SURFACE_TYPE_ORDER)}
    fkeys = [
        C.face_sort_key(rank.get(r["type"], 99), r["centroid"], r["area"], r["bmin"], r["bmax"], fps[i])
        for i, r in enumerate(raw)
    ]
    f_order = C.canonical_order(fkeys)
    f_canon = {orig: pos for pos, orig in enumerate(f_order)}  # traversal idx -> canonical rank

    # --- raw edge attributes ------------------------------------------------------------
    eraw: list[dict[str, Any]] = []
    for ei, e in enumerate(edges):
        ctype = occ.curve_type(e)
        degenerated = ctype == "degenerate"
        v1, v2 = TopExp.FirstVertex_s(e), TopExp.LastVertex_s(e)
        start = occ.pnt(BRep_Tool.Pnt_s(v1)) if not v1.IsNull() else [0.0, 0.0, 0.0]
        end = occ.pnt(BRep_Tool.Pnt_s(v2)) if not v2.IsNull() else start
        closed = (not v1.IsNull()) and v1.IsSame(v2)
        length = 0.0 if degenerated else occ.linear_length(e)
        if degenerated:
            mid = start
        else:
            c = BRepAdaptor_Curve(e)
            mid = occ.pnt(c.Value(0.5 * (c.FirstParameter() + c.LastParameter())))
        fl = edge_faces[ei]
        dihedral: float | None = None
        if degenerated:
            convexity = "unknown"
        elif edge_is_seam[ei]:
            convexity = "seam"
        elif len(fl) == 0:
            convexity = "unknown"  # edge not used by any face (only reachable without solid isolation)
        elif len(fl) == 1:
            convexity = "boundary"
        elif len(fl) > 2:
            convexity = "non_manifold"
        else:
            convexity, dihedral = _edge_convexity(
                e, faces[fl[0]], faces[fl[1]], evals[fl[0]], evals[fl[1]], smooth_deg
            )
        eraw.append(
            {
                "type": ctype,
                "start": start,
                "end": end,
                "mid": mid,
                "length": length,
                "closed": closed,
                "degenerated": degenerated,
                "convexity": convexity,
                "dihedral": dihedral,
            }
        )
    efps = [
        C.edge_fingerprint(r["type"], r["length"], r["convexity"], [fps[j] for j in edge_faces[i]])
        for i, r in enumerate(eraw)
    ]
    crank = {t: i for i, t in enumerate(occ.CURVE_TYPE_ORDER)}
    ekeys = [
        C.edge_sort_key(crank.get(r["type"], 99), r["mid"], r["length"], r["start"], r["end"], efps[i])
        for i, r in enumerate(eraw)
    ]
    e_order = C.canonical_order(ekeys)
    e_canon = {orig: pos for pos, orig in enumerate(e_order)}

    face_records: list[FaceRecord] = []
    for pos, orig in enumerate(f_order):
        r = raw[orig]
        face_records.append(
            FaceRecord(
                face_id=C.face_id(pos),
                traversal_index=orig,
                surface_type=r["type"],
                orientation="reversed" if occ.is_reversed(faces[orig]) else "forward",
                area_mm2=_r(r["area"]),
                centroid_mm=_rv(r["centroid"]),
                normal_at_centroid=_rv(r["normal"]) if r["normal"] is not None else None,
                bbox_min_mm=_rv(r["bmin"]),
                bbox_max_mm=_rv(r["bmax"]),
                uv_bounds=_rv(r["uvb"]),
                analytic_params=r["analytic"],
                curvature=r["curv"],
                adjacent_face_ids=sorted(C.face_id(f_canon[j]) for j in face_adj[orig]),
                boundary_edge_ids=sorted(C.edge_id(e_canon[j]) for j in face_edges[orig]),
                outer_loop_edge_ids=sorted(C.edge_id(e_canon[j]) for j in r["outer"] if j >= 0),
                loop_count=r["loops"],
                fingerprint=fps[orig],
            )
        )
    edge_records: list[EdgeRecord] = []
    for pos, orig in enumerate(e_order):
        r = eraw[orig]
        edge_records.append(
            EdgeRecord(
                edge_id=C.edge_id(pos),
                traversal_index=orig,
                curve_type=r["type"],
                length_mm=_r(r["length"]),
                start_mm=_rv(r["start"]),
                end_mm=_rv(r["end"]),
                closed=bool(r["closed"]),
                degenerated=bool(r["degenerated"]),
                adjacent_face_ids=sorted(C.face_id(f_canon[j]) for j in edge_faces[orig]),
                convexity=r["convexity"],
                dihedral_angle_deg=r["dihedral"],
                fingerprint=efps[orig],
            )
        )
    return BRepModel(
        shape=shape,
        faces=[faces[i] for i in f_order],
        edges=[edges[i] for i in e_order],
        face_records=face_records,
        edge_records=edge_records,
        face_index={fr.face_id: i for i, fr in enumerate(face_records)},
        edge_index={er.edge_id: i for i, er in enumerate(edge_records)},
    )
