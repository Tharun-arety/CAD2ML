"""Isolated extraction stage: validated B-Rep -> all aligned representations.

Runs inside a spawned child (``parsers.isolation.run_isolated``). Writes every artifact
into a private staging directory and returns a JSON-serialisable summary. The parent
builds and validates the manifest and atomically promotes the staging directory.
"""

from __future__ import annotations

import hashlib
import io
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from cad2ml.config import PipelineConfig
from cad2ml.parsers.isolation import report_progress


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class _Writer:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.refs: dict[str, dict[str, Any]] = {}

    def _ref(self, name: str, rel: str, media: str, desc: str) -> None:
        p = self.root / rel
        self.refs[name] = {
            "key": rel,
            "sha256": _sha(p),
            "bytes": p.stat().st_size,
            "media_type": media,
            "description": desc,
        }

    def json(self, name: str, rel: str, obj: Any, desc: str) -> None:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj, indent=1, sort_keys=True, allow_nan=False))
        self._ref(name, rel, "application/json", desc)

    def npz(self, name: str, rel: str, arrays: dict[str, Any], desc: str) -> None:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        np.savez_compressed(buf, **arrays)  # type: ignore[arg-type]
        p.write_bytes(buf.getvalue())
        self._ref(name, rel, "application/x-npz", desc)

    def png(self, name: str, rel: str, arr: Any, desc: str) -> None:
        from PIL import Image

        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if arr.dtype == np.uint16:
            Image.fromarray(arr.astype(np.uint16)).save(p, format="PNG")
        else:
            Image.fromarray(arr).save(p, format="PNG", optimize=False)
        self._ref(name, rel, "image/png", desc)


def extract_child(brep_path: str, staging_dir: str, cfg_json: str, meta: dict[str, Any]) -> dict[str, Any]:
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID, TopAbs_VERTEX

    from cad2ml.errors import PipelineError
    from cad2ml.geometry import occ
    from cad2ml.geometry.validation import validate_and_repair
    from cad2ml.representations.graph import (
        ADJ_FEATURE_NAMES,
        FACE_FEATURE_NAMES,
        GLOBAL_FEATURE_NAMES,
        FaceGraph,
        build_face_graph,
        check_graph_invariants,
    )
    from cad2ml.representations.mesh import tessellate, triangle_areas
    from cad2ml.representations.pointcloud import denormalize, sample_point_cloud
    from cad2ml.representations.render import render_views, unproject
    from cad2ml.semantics.recognizer import recognize
    from cad2ml.topology.entities import extract_entities

    cfg = PipelineConfig.model_validate_json(cfg_json)
    timings: dict[str, float] = {}
    root = Path(staging_dir)
    root.mkdir(parents=True, exist_ok=True)
    w = _Writer(root)

    def tick(name: str, t0: float) -> float:
        timings[name] = round(time.perf_counter() - t0, 6)
        return time.perf_counter()

    t = time.perf_counter()
    report_progress("normalizing")
    shape = occ.read_brep(brep_path)
    rep = validate_and_repair(shape, cfg.repair)
    shape = rep.shape
    volume, com = occ.volume_props(shape)
    area, _ = occ.surface_props(shape)
    if volume <= 0 or area <= 0:
        raise PipelineError("DEGENERATE_GEOMETRY", f"volume={volume}, area={area}", "normalizing")
    bmin, bmax = occ.bbox(shape)
    dims = [hi - lo for lo, hi in zip(bmin, bmax, strict=True)]
    diag = float(np.linalg.norm(dims))
    t = tick("validate_repair", t)

    report_progress("extracting")
    model = extract_entities(shape)
    t = tick("canonical_entities", t)
    mesh = tessellate(model, cfg.tessellation, diag)
    t = tick("tessellation", t)
    pc = sample_point_cloud(model, mesh, cfg.pointcloud, bmin, bmax)
    t = tick("point_cloud", t)
    graph = build_face_graph(model.face_records, model.edge_records, volume, area, dims)
    t = tick("face_graph", t)
    views = render_views(mesh, bmin, bmax, cfg.render)
    t = tick("multi_view", t)
    observations, features = recognize(model.face_records, model.edge_records, cfg.recognizer)
    t = tick("feature_recognition", t)

    report_progress("validating_outputs")
    F = len(model.face_records)
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    checks.update(
        {
            f"graph.{k}": v
            for k, v in check_graph_invariants(graph, model.face_records, model.edge_records).items()
        }
    )
    checks["pointcloud.face_ids_valid"] = bool(pc.face_index.min() >= 0 and pc.face_index.max() < F)
    checks["pointcloud.every_face_sampled"] = bool(np.all(pc.face_point_range[:, 1] > 0))
    nl = np.linalg.norm(pc.normals, axis=1)
    checks["pointcloud.normals_finite_unit"] = bool(
        np.isfinite(pc.normals).all() and np.abs(nl - 1).max() < 1e-3
    )
    checks["pointcloud.on_surface"] = (
        pc.max_projection_distance_mm <= max(cfg.tessellation.linear_deflection_mm, 1e-3) * 2
    )
    inv = denormalize(pc.points, pc.center, pc.scale)
    details["pointcloud.inverse_transform_max_err_mm"] = float(np.abs(inv - pc.points_mm).max())
    checks["pointcloud.inverse_transform"] = (
        details["pointcloud.inverse_transform_max_err_mm"] < 1e-4 * pc.scale + 1e-4
    )
    details["pointcloud.max_projection_distance_mm"] = pc.max_projection_distance_mm
    checks["mesh.tri_face_index_valid"] = bool(
        mesh.tri_face_index.min() >= 0 and mesh.tri_face_index.max() < F
    )
    checks["mesh.every_face_triangulated"] = bool(np.all(mesh.face_tri_range[:, 1] > 0))
    mesh_area = float(triangle_areas(mesh).sum())
    details["mesh.area_relative_error"] = abs(mesh_area - area) / area
    checks["mesh.area_close_to_brep"] = details["mesh.area_relative_error"] < 0.02
    ids_in_views = np.unique(views.face_id)
    ids_in_views = ids_in_views[ids_in_views > 0] - 1
    checks["views.mask_ids_are_canonical_faces"] = bool(np.all(ids_in_views < F))
    # unprojected mask pixels must lie within their face's bounding box (expanded by mesh deflection + 1 px)
    worst = 0.0
    for k, cam in enumerate(views.cameras):
        rows, cols = np.nonzero(views.face_id[k])
        if len(rows) == 0:
            continue
        sel = np.linspace(0, len(rows) - 1, min(400, len(rows))).astype(int)
        r_, c_ = rows[sel], cols[sel]
        xyz = unproject(views.depth[k], cam, r_, c_)
        fi = views.face_id[k][r_, c_].astype(int) - 1
        lo = np.asarray([model.face_records[i].bbox_min_mm for i in fi])
        hi = np.asarray([model.face_records[i].bbox_max_mm for i in fi])
        out = np.maximum(lo - xyz, 0) + np.maximum(xyz - hi, 0)
        worst = max(worst, float(np.linalg.norm(out, axis=1).max()))
    px_mm = 2 * diag / cfg.render.resolution
    details["views.max_pixel_to_face_bbox_mm"] = worst
    checks["views.pixels_backproject_to_their_face"] = (
        worst <= px_mm + mesh.settings["linear_deflection_mm"] * 4
    )
    feat_faces = {fid for ft in features for fid in ft.participating_faces}
    checks["features.every_face_assigned"] = feat_faces == {fr.face_id for fr in model.face_records}
    # graph serialisation round trip
    buf = io.BytesIO()
    np.savez_compressed(buf, **graph.to_npz())  # type: ignore[arg-type]
    buf.seek(0)
    with np.load(buf) as d:
        g2 = FaceGraph.from_npz(d)
    checks["graph.serialization_roundtrip"] = bool(
        np.array_equal(g2.face_features, graph.face_features)
        and np.array_equal(g2.edge_index, graph.edge_index)
        and np.array_equal(g2.adjacency_features, graph.adjacency_features)
        and g2.node_face_ids == graph.node_face_ids
    )
    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise PipelineError(
            "OUTPUT_INVARIANT_VIOLATION", f"failed checks: {failed}; details={details}", "validating_outputs"
        )
    t = tick("output_validation", t)

    # ---- artifacts ----------------------------------------------------------------------
    occ.write_brep(shape, str(root / "canonical.brep"))
    w._ref(
        "canonical_brep",
        "canonical.brep",
        "application/x-occt-brep",
        "Validated (and, if recorded, repaired) B-Rep in millimetres; basis for all derived artifacts",
    )
    w.json(
        "brep",
        "brep.json",
        {
            "faces": [f.model_dump() for f in model.face_records],
            "edges": [e.model_dump() for e in model.edge_records],
        },
        "Canonical face and edge records with deterministic IDs",
    )
    w.npz("graph", "graph.npz", graph.to_npz(), "Face-adjacency graph (node i = canonical face i)")
    w.json(
        "graph_schema",
        "graph_schema.json",
        {
            "face_feature_names": FACE_FEATURE_NAMES,
            "adjacency_feature_names": ADJ_FEATURE_NAMES,
            "global_feature_names": GLOBAL_FEATURE_NAMES,
            "node_order": "canonical face order",
            "edge_index": "directed, both directions, pairs of distinct faces sharing >= 1 B-Rep edge",
        },
        "Graph feature definitions",
    )
    w.npz("pointcloud", "pointcloud.npz", pc.to_npz(), "Surface point cloud with per-point face index")
    w.npz("mesh", "mesh.npz", mesh.to_npz(), "Tessellation with triangle -> face index")
    for k in range(len(views.cameras)):
        w.png(f"view_{k}_rgb", f"views/view_{k:02d}_rgb.png", views.rgb[k], f"Shaded view {k}")
        w.png(
            f"view_{k}_face_id",
            f"views/view_{k:02d}_face_id.png",
            views.face_id[k],
            f"Face-ID mask view {k} (uint16, 0=background, k+1=face index k)",
        )
    w.npz(
        "views",
        "views.npz",
        {"depth": views.depth, "normals": views.normals, "face_id": views.face_id},
        "Depth (mm), world normals, face-ID masks for all views",
    )
    w.json(
        "cameras",
        "cameras.json",
        {
            "views": views.cameras,
            "render_settings": cfg.render.model_dump(),
            "backend": "cad2ml numpy z-buffer rasterizer v1",
        },
        "Camera poses/intrinsics",
    )
    w.json(
        "features",
        "features.json",
        {
            "observations": [o.model_dump() for o in observations],
            "features": [f.model_dump() for f in features],
        },
        "Geometric observations (computed) and inferred engineering features",
    )

    face_features: dict[str, list[str]] = {}
    for ft in features:
        for fid in ft.participating_faces:
            face_features.setdefault(fid, []).append(ft.feature_id)
    lineage_faces = {}
    for i, fr in enumerate(model.face_records):
        ps, pcnt = (int(x) for x in pc.face_point_range[i])
        ts, tcnt = (int(x) for x in mesh.face_tri_range[i])
        lineage_faces[fr.face_id] = {
            "graph_node": i,
            "surface_type": fr.surface_type,
            "kernel_traversal_index": fr.traversal_index,
            "point_range": [ps, ps + pcnt],
            "triangle_range": [ts, ts + tcnt],
            "view_pixels": {
                str(k): int(np.count_nonzero(views.face_id[k] == i + 1)) for k in range(len(views.cameras))
            },
            "feature_ids": face_features.get(fr.face_id, []),
            "fingerprint": fr.fingerprint,
        }
    w.json(
        "lineage",
        "lineage.json",
        {
            "sample_id": meta["sample_id"],
            "source": meta["source"],
            "versions": meta["versions"],
            "range_convention": "half-open [start, end)",
            "faces": lineage_faces,
        },
        "Entity-level cross-representation lineage",
    )
    tick("artifact_write", t)

    def hist(kind: Any, fn: Any) -> dict[str, int]:
        m = occ.index_map(shape, kind)
        out: dict[str, int] = {}
        from OCP.TopoDS import TopoDS

        for i in range(1, m.Extent() + 1):
            s = TopoDS.Face_s(m.FindKey(i)) if kind == TopAbs_FACE else TopoDS.Edge_s(m.FindKey(i))
            name = fn(s)
            out[name] = out.get(name, 0) + 1
        return dict(sorted(out.items()))

    geometry = {
        "body_count": 1,
        "solid_count": occ.index_map(shape, TopAbs_SOLID).Extent(),
        "shell_count": occ.index_map(shape, TopAbs_SHELL).Extent(),
        "face_count": F,
        "edge_count": occ.index_map(shape, TopAbs_EDGE).Extent(),
        "vertex_count": occ.index_map(shape, TopAbs_VERTEX).Extent(),
        "volume_mm3": round(volume, 6),
        "surface_area_mm2": round(area, 6),
        "bounding_box_min_mm": [round(x, 6) for x in bmin],
        "bounding_box_max_mm": [round(x, 6) for x in bmax],
        "bounding_box_mm": [round(x, 6) for x in dims],
        "center_of_mass_mm": [round(x, 6) for x in com],
        "surface_type_histogram": hist(TopAbs_FACE, occ.surface_type),
        "curve_type_histogram": hist(TopAbs_EDGE, occ.curve_type),
    }
    validation = {
        "source_valid": rep.source_valid,
        "closed_solid": rep.closed_solid,
        "orientation_ok": rep.orientation_ok,
        "repair_attempted": rep.repair_attempted,
        "repaired_valid": rep.repaired_valid,
        "repair_operations": rep.operations,
        "max_deviation_mm": rep.max_deviation_mm,
        "volume_relative_change": rep.volume_relative_change,
        "warnings": rep.warnings,
    }
    return {
        "geometry": geometry,
        "validation": validation,
        "artifacts": w.refs,
        "timings": timings,
        "observations": [o.model_dump() for o in observations],
        "features": [f.model_dump() for f in features],
        "quality": {"passed": True, "checks": checks, "details": details},
        "settings": {"tessellation": mesh.settings, "pointcloud": pc.settings},
    }
