"""Geometry tests on known primitives (in-process OCCT; no traversal-order assumptions)."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pytest

cq = pytest.importorskip("cadquery")

from cad2ml.config import (  # noqa: E402
    PointCloudConfig,
    RecognizerConfig,
    RenderConfig,
    RepairConfig,
    TessellationConfig,
)
from cad2ml.errors import PipelineError  # noqa: E402
from cad2ml.geometry import occ  # noqa: E402
from cad2ml.geometry.validation import diagnose, validate_and_repair  # noqa: E402
from cad2ml.representations.mesh import tessellate, triangle_areas  # noqa: E402
from cad2ml.representations.pointcloud import denormalize, sample_point_cloud  # noqa: E402
from cad2ml.representations.render import render_views  # noqa: E402
from cad2ml.semantics.recognizer import recognize  # noqa: E402
from cad2ml.topology.entities import extract_entities  # noqa: E402


def _model(shape):
    s = shape.wrapped if hasattr(shape, "wrapped") else shape
    return s, extract_entities(s)


def test_box_measurements_and_counts() -> None:
    s, m = _model(cq.Solid.makeBox(40, 30, 10))
    vol, com = occ.volume_props(s)
    area, _ = occ.surface_props(s)
    assert vol == pytest.approx(12000, rel=1e-9) and area == pytest.approx(2 * (1200 + 400 + 300), rel=1e-9)
    assert np.allclose(com, [20, 15, 5])
    assert (len(m.face_records), len(m.edge_records)) == (6, 12)
    assert {e.convexity for e in m.edge_records} == {"convex"}
    assert all(e.dihedral_angle_deg == pytest.approx(90.0, abs=1e-6) for e in m.edge_records)
    assert all(f.surface_type == "plane" for f in m.face_records)


def test_cylinder_hole_surface_type_and_diameter() -> None:
    s, m = _model(cq.Workplane("XY").box(40, 30, 10).faces(">Z").workplane().hole(8).val())
    cyl = [f for f in m.face_records if f.surface_type == "cylinder"]
    assert len(cyl) == 1 and cyl[0].analytic_params["radius"] == pytest.approx(4.0)
    assert cyl[0].curvature["mean_curvature_mean"] < 0  # concave: material outside the cylinder
    _, feats = recognize(m.face_records, m.edge_records, RecognizerConfig())
    holes = [f for f in feats if f.feature_type == "through_hole"]
    assert len(holes) == 1 and holes[0].parameters["diameter_mm"] == pytest.approx(8.0)
    assert holes[0].participating_faces == [cyl[0].face_id]
    assert 0 < holes[0].confidence <= 0.95


def test_boss_is_not_a_hole_and_blind_hole_detected() -> None:
    boss = cq.Workplane("XY").box(40, 40, 5).faces(">Z").workplane().circle(6).extrude(8).val()
    _, m = _model(boss)
    _, feats = recognize(m.face_records, m.edge_records)
    assert not [f for f in feats if "hole" in f.feature_type]
    blind = cq.Workplane("XY").box(40, 40, 20).faces(">Z").workplane().hole(6, depth=8).val()
    _, mb = _model(blind)
    _, fb = recognize(mb.face_records, mb.edge_records)
    bh = [f for f in fb if f.feature_type == "blind_hole"]
    assert len(bh) == 1 and bh[0].parameters["depth_mm"] == pytest.approx(8.0, abs=0.02)
    assert len(bh[0].participating_faces) == 2  # wall + floor


def test_concave_edge_on_l_bracket() -> None:
    l_shape = cq.Solid.makeBox(50, 40, 5).fuse(cq.Solid.makeBox(50, 5, 40)).clean()
    _, m = _model(l_shape)
    c = Counter(e.convexity for e in m.edge_records)
    assert c["concave"] == 1 and c["convex"] == len(m.edge_records) - 1
    concave = next(e for e in m.edge_records if e.convexity == "concave")
    assert concave.dihedral_angle_deg == pytest.approx(270.0, abs=1e-6)


def test_canonical_ids_independent_of_construction_order() -> None:
    a = cq.Solid.makeBox(30, 20, 5).fuse(cq.Solid.makeBox(10, 20, 30)).clean()
    b = cq.Solid.makeBox(10, 20, 30).fuse(cq.Solid.makeBox(30, 20, 5)).clean()
    _, ma = _model(a)
    _, mb = _model(b)
    assert [(f.face_id, f.fingerprint, f.surface_type) for f in ma.face_records] == [
        (f.face_id, f.fingerprint, f.surface_type) for f in mb.face_records
    ]
    assert [(e.edge_id, e.fingerprint) for e in ma.edge_records] == [
        (e.edge_id, e.fingerprint) for e in mb.edge_records
    ]


def test_validation_valid_solid_needs_no_repair() -> None:
    out = validate_and_repair(cq.Solid.makeBox(1, 2, 3).wrapped, RepairConfig())
    assert out.source_valid and not out.repair_attempted and out.operations == []


def test_open_shell_is_quarantined_not_forced() -> None:
    box = cq.Solid.makeBox(20, 20, 20)
    shell = cq.Shell.makeShell(box.Faces()[:-1])
    d = diagnose(shell.wrapped)
    assert d.solid_count == 0 and not d.closed_shells
    with pytest.raises(PipelineError) as e:
        validate_and_repair(shell.wrapped, RepairConfig())
    assert e.value.code == "NO_SOLID"


def test_closed_face_soup_is_sewn_with_recorded_operations() -> None:
    box = cq.Solid.makeBox(20, 20, 20)
    shell = cq.Shell.makeShell(box.Faces())
    out = validate_and_repair(shell.wrapped, RepairConfig())
    assert out.repair_attempted and out.repaired_valid
    assert any("Sewing" in op for op in out.operations) and any("MakeSolid" in op for op in out.operations)
    assert out.max_deviation_mm is not None and out.max_deviation_mm < 1e-6
    assert occ.volume_props(out.shape)[0] == pytest.approx(8000, rel=1e-6)


@pytest.fixture(scope="module")
def plate_with_tiny_hole():
    shape = cq.Workplane("XY").box(50, 50, 5).faces(">Z").workplane().hole(0.2).val().wrapped
    m = extract_entities(shape)
    bmin, bmax = occ.bbox(shape)
    diag = float(np.linalg.norm(np.subtract(bmax, bmin)))
    mesh = tessellate(m, TessellationConfig(), diag)
    return shape, m, mesh, bmin, bmax


def test_point_and_triangle_correspondence(plate_with_tiny_hole) -> None:
    _, m, mesh, bmin, bmax = plate_with_tiny_hole
    F = len(m.face_records)
    pc = sample_point_cloud(m, mesh, PointCloudConfig(num_points=2048, min_points_per_face=8), bmin, bmax)
    assert pc.face_index.min() >= 0 and pc.face_index.max() < F
    assert np.isfinite(pc.normals).all() and np.allclose(np.linalg.norm(pc.normals, axis=1), 1, atol=1e-5)
    tiny = next(i for i, f in enumerate(m.face_records) if f.surface_type == "cylinder")
    assert m.face_records[tiny].area_mm2 < 4 and pc.face_point_range[tiny, 1] >= 8  # small face represented
    s, c = pc.face_point_range[tiny]
    r = np.linalg.norm(
        pc.points_mm[s : s + c, :2] - np.asarray(m.face_records[tiny].analytic_params["axis_origin"])[:2],
        axis=1,
    )
    assert np.allclose(r, 0.1, atol=1e-6)  # points lie exactly on the 0.2 mm hole wall
    assert np.allclose(denormalize(pc.points, pc.center, pc.scale), pc.points_mm, atol=1e-4)
    for fi in range(F):
        ts, tc = mesh.face_tri_range[fi]
        assert (mesh.tri_face_index[ts : ts + tc] == fi).all() and tc > 0
        area = triangle_areas(mesh)[ts : ts + tc].sum()
        assert area == pytest.approx(m.face_records[fi].area_mm2, rel=0.05, abs=5e-3)


def test_point_cloud_deterministic(plate_with_tiny_hole) -> None:
    _, m, mesh, bmin, bmax = plate_with_tiny_hole
    a = sample_point_cloud(m, mesh, PointCloudConfig(num_points=1024, seed=5), bmin, bmax)
    b = sample_point_cloud(m, mesh, PointCloudConfig(num_points=1024, seed=5), bmin, bmax)
    c = sample_point_cloud(m, mesh, PointCloudConfig(num_points=1024, seed=6), bmin, bmax)
    assert np.array_equal(a.points_mm, b.points_mm) and np.array_equal(a.face_index, b.face_index)
    assert not np.array_equal(a.points_mm, c.points_mm)


def test_views_face_ids_are_canonical(plate_with_tiny_hole) -> None:
    _, m, mesh, bmin, bmax = plate_with_tiny_hole
    v = render_views(mesh, bmin, bmax, RenderConfig(num_views=2, resolution=96))
    ids = np.unique(v.face_id)
    assert ids[0] == 0 and ids.max() <= len(m.face_records)
    assert (v.depth[v.face_id == 0] == 0).all() and (v.depth[v.face_id > 0] > 0).all()
    top = next(
        i for i, f in enumerate(m.face_records) if f.surface_type == "plane" and f.normal_at_centroid[2] > 0.9
    )
    assert (v.face_id[0] == top + 1).sum() > 100  # elevated first camera sees the top face
    n = v.normals[0][v.face_id[0] == top + 1].astype(np.float32)
    assert np.allclose(n.mean(0), [0, 0, 1], atol=1e-2)
    assert math.isclose(float(np.linalg.norm(n, axis=1).mean()), 1.0, abs_tol=1e-2)
