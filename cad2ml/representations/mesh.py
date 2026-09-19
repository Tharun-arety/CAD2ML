"""Controlled tessellation of the canonical B-Rep with exact triangle -> face correspondence.

Triangles are emitted face by face in canonical order, so each face owns a contiguous
triangle range. Vertices are *not* welded across faces: this keeps correspondence
exact and lets per-vertex normals follow the true surface on each side of sharp edges.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from OCP.BRep import BRep_Tool
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepTools import BRepTools
from OCP.TopLoc import TopLoc_Location

from cad2ml.config import TessellationConfig
from cad2ml.errors import PipelineError
from cad2ml.geometry import occ
from cad2ml.topology.entities import BRepModel


@dataclass
class MeshData:
    vertices: NDArray[np.float64]  # [V,3] mm
    vertex_normals: NDArray[np.float32]  # [V,3] outward unit
    triangles: NDArray[np.int32]  # [T,3] CCW seen from outside
    tri_face_index: NDArray[np.int32]  # [T] canonical face index
    face_tri_range: NDArray[np.int64]  # [F,2] (start, count)
    face_deflection: NDArray[np.float64]  # [F] chord deflection OCCT reports it achieved per face (mm)
    settings: dict[str, Any]

    def to_npz(self) -> dict[str, NDArray[Any]]:
        return {
            "vertices": self.vertices,
            "vertex_normals": self.vertex_normals,
            "triangles": self.triangles,
            "tri_face_index": self.tri_face_index,
            "face_tri_range": self.face_tri_range,
            "face_deflection": self.face_deflection,
        }


def tessellate(model: BRepModel, cfg: TessellationConfig, bbox_diag: float) -> MeshData:
    lin = min(cfg.linear_deflection_mm, max(cfg.relative_linear_deflection * bbox_diag, 1e-3))
    BRepTools.Clean_s(model.shape)
    mesher = BRepMesh_IncrementalMesh(model.shape, lin, False, cfg.angular_deflection_rad, True)
    mesher.Perform()
    verts: list[NDArray[np.float64]] = []
    norms: list[NDArray[np.float32]] = []
    tris: list[NDArray[np.int32]] = []
    tface: list[NDArray[np.int32]] = []
    ranges = np.zeros((len(model.faces), 2), dtype=np.int64)
    v_off = 0
    t_off = 0
    deflections: list[float] = []
    for fi, face in enumerate(model.faces):
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is None or tri.NbTriangles() < cfg.min_triangles_per_face:
            raise PipelineError(
                "TESSELLATION_FAILED", f"face {model.face_records[fi].face_id} has no triangles", "extracting"
            )
        trsf = loc.Transformation()
        deflections.append(float(tri.Deflection()))
        nn, nt = tri.NbNodes(), tri.NbTriangles()
        v = np.empty((nn, 3), dtype=np.float64)
        for k in range(1, nn + 1):
            v[k - 1] = occ.pnt(tri.Node(k).Transformed(trsf))
        t: NDArray[np.int32] = np.empty((nt, 3), dtype=np.int32)
        for k in range(1, nt + 1):
            a, b, c = tri.Triangle(k).Get()
            t[k - 1] = (a - 1, b - 1, c - 1)
        if occ.is_reversed(face):
            t = t[:, [0, 2, 1]]
        ev = occ.FaceEvaluator(face)
        n = np.zeros((nn, 3), dtype=np.float32)
        have_uv = tri.HasUVNodes()
        for k in range(1, nn + 1):
            normal = None
            if have_uv:
                uv = tri.UVNode(k)
                _, normal, _, _ = ev.at_uv(uv.X(), uv.Y())
            n[k - 1] = normal if normal is not None else (0.0, 0.0, 0.0)
        # fall back to area-weighted triangle normals where the surface normal is undefined
        tri_n = np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]])
        missing = np.linalg.norm(n, axis=1) < 0.5
        if missing.any():
            acc = np.zeros((nn, 3))
            for j in range(3):
                np.add.at(acc, t[:, j], tri_n)
            ln = np.linalg.norm(acc, axis=1, keepdims=True)
            acc = np.divide(acc, ln, out=np.zeros_like(acc), where=ln > 0)
            n[missing] = acc[missing]
        verts.append(v)
        norms.append(n)
        tris.append(t + v_off)
        tface.append(np.full(nt, fi, dtype=np.int32))
        ranges[fi] = (t_off, nt)
        v_off += nn
        t_off += nt
    return MeshData(
        vertices=np.concatenate(verts),
        vertex_normals=np.concatenate(norms),
        triangles=np.concatenate(tris),
        tri_face_index=np.concatenate(tface),
        face_tri_range=ranges,
        face_deflection=np.asarray(deflections, dtype=np.float64),
        settings={
            "linear_deflection_mm": lin,
            "angular_deflection_rad": cfg.angular_deflection_rad,
            "relative": False,
            "parallel": True,
            "min_triangles_per_face": cfg.min_triangles_per_face,
            "normal_source": "surface_uv_evaluation_with_triangle_fallback",
            "vertex_welding": "none (per-face vertices)",
        },
    )


def triangle_areas(mesh: MeshData) -> NDArray[np.float64]:
    v, t = mesh.vertices, mesh.triangles
    return 0.5 * np.linalg.norm(np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]]), axis=1)
