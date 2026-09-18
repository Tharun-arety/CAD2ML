"""Property-based and unit tests: normalization, allocation, tools, splits, states, metrics, graph."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cad2ml.datasets.builder import assign_splits, check_leakage
from cad2ml.evaluation.metrics import classification_report
from cad2ml.jobs.states import TRANSITIONS, IllegalTransition, JobState, check_transition, is_terminal
from cad2ml.representations.graph import build_face_graph, check_graph_invariants
from cad2ml.representations.pointcloud import allocate, denormalize, normalization_transform, normalize
from cad2ml.schemas.manifest import EdgeRecord, FaceRecord
from cad2ml.synthetic.tools import ExtrudedRoundedRect

finite = st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False)


@settings(max_examples=200, deadline=None)
@given(
    pts=st.lists(st.tuples(finite, finite, finite), min_size=1, max_size=50),
    lo=st.tuples(finite, finite, finite),
    ext=st.tuples(*[st.floats(0.01, 5e3)] * 3),
)
def test_normalization_finite_and_invertible(pts: list, lo: tuple, ext: tuple) -> None:
    bmin = list(lo)
    bmax = [a + b for a, b in zip(lo, ext, strict=True)]
    c, s = normalization_transform(bmin, bmax)
    p = np.asarray(pts)
    q = normalize(p, c, s)
    assert np.isfinite(q).all()
    assert np.allclose(denormalize(q, c, s), p, atol=1e-6 * max(1.0, float(np.abs(p).max())))


@settings(max_examples=200, deadline=None)
@given(
    areas=st.lists(st.floats(0.0, 1e5), min_size=1, max_size=200),
    total=st.integers(1, 20000),
    minimum=st.integers(0, 16),
)
def test_allocation_min_per_face_and_total(areas: list, total: int, minimum: int) -> None:
    a = np.asarray(areas)
    k = allocate(a, total, minimum)
    assert (k >= minimum).all()
    assert k.sum() == max(total, minimum * len(a)) or a.sum() == 0


def test_tool_sdf_cylinder_and_stadium() -> None:
    cyl = ExtrudedRoundedRect((0, 0, 0), (1, 0, 0), (0, 0, 1), 5, 5, 5, 0, 10)
    pts = np.array([[5, 0, 3], [0, -5, 7], [0, 0, 5], [7, 0, 5], [0, 0, 10]])
    assert cyl.on_boundary(pts, 1e-9).tolist() == [True, True, False, False, True]
    assert cyl.sdf(np.array([[0, 0, 5]]))[0] == pytest.approx(-5)
    slot = ExtrudedRoundedRect((0, 0, 0), (1, 0, 0), (0, 0, 1), 20, 4, 4, -1, 6)
    assert slot.on_boundary(np.array([[20, 0, 2], [16, 4, 2], [0, 4, 0]]), 1e-9).all()


@settings(max_examples=100, deadline=None)
@given(
    n_fam=st.integers(1, 8),
    n_var=st.integers(1, 5),
    seed=st.integers(0, 10_000),
    mode=st.sampled_from(["group", "family"]),
)
def test_splits_never_leak(n_fam: int, n_var: int, seed: int, mode: str) -> None:
    units = {f"fam{f}/v{v}": f"fam{f}" for f in range(n_fam) for v in range(n_var)}
    split_unit = {u: u if mode == "group" else fam for u, fam in units.items()}
    split = assign_splits({split_unit[u]: units[u] for u in units}, seed, mode)
    samples = {
        f"{u}#{i}": {"split_unit": split_unit[u], "family": units[u], "split": split[split_unit[u]]}
        for u in units
        for i in range(3)
    }
    assert check_leakage(samples, "split_unit") == {}
    if mode == "family":
        assert check_leakage(samples, "family") == {}
    assert set(split.values()) <= {"train", "val", "test"}
    assert split == assign_splits({split_unit[u]: units[u] for u in units}, seed, mode)  # deterministic


def test_leakage_detector_flags_crossing() -> None:
    samples = {"a": {"g": "x", "split": "train"}, "b": {"g": "x", "split": "test"}}
    assert check_leakage(samples, "g") == {"x": ["test", "train"]}


def test_job_state_machine() -> None:
    path = [
        "received",
        "validated",
        "queued",
        "parsing",
        "normalizing",
        "extracting",
        "validating_outputs",
        "completed",
    ]
    for a, b in zip(path, path[1:], strict=False):
        check_transition(a, b)
    for bad in [
        ("completed", "queued"),
        ("received", "parsing"),
        ("queued", "normalizing"),
        ("quarantined", "queued"),
        ("rejected", "completed"),
    ]:
        with pytest.raises(IllegalTransition):
            check_transition(*bad)
    check_transition("extracting", "failed_retryable")
    check_transition("failed_retryable", "queued")
    assert all(not TRANSITIONS[s] for s in JobState if is_terminal(s))


def test_classification_report() -> None:
    r = classification_report([0, 0, 1, 1, 2], [0, 1, 1, 1, 0], ["a", "b", "c"])
    assert r["confusion_matrix"]["rows_true_cols_pred"] == [[1, 1, 0], [0, 2, 0], [1, 0, 0]]
    assert r["per_class"]["b"]["precision"] == pytest.approx(0.6667, abs=1e-4)
    assert r["per_class"]["c"]["recall"] == 0.0
    assert r["summary"]["accuracy"] == 0.6


def _face(fid: str, adj: list[str], edges: list[str], stype: str = "plane") -> FaceRecord:
    return FaceRecord(
        face_id=fid,
        traversal_index=0,
        surface_type=stype,
        orientation="forward",
        area_mm2=10.0,
        centroid_mm=[0, 0, 0],
        normal_at_centroid=[0, 0, 1],
        bbox_min_mm=[0, 0, 0],
        bbox_max_mm=[1, 1, 1],
        uv_bounds=[0, 1, 0, 1],
        analytic_params={},
        curvature={},
        adjacent_face_ids=adj,
        boundary_edge_ids=edges,
        outer_loop_edge_ids=edges,
        loop_count=1,
        fingerprint=fid,
    )


def _edge(eid: str, faces: list[str], conv: str = "convex") -> EdgeRecord:
    return EdgeRecord(
        edge_id=eid,
        traversal_index=0,
        curve_type="line",
        length_mm=1.0,
        start_mm=[0, 0, 0],
        end_mm=[1, 0, 0],
        closed=False,
        degenerated=False,
        adjacent_face_ids=faces,
        convexity=conv,
        dihedral_angle_deg=90.0,
        fingerprint=eid,
    )


def test_graph_invariants_on_handmade_topology() -> None:
    faces = [
        _face("F000", ["F001"], ["E000", "E002"]),
        _face("F001", ["F000", "F002"], ["E000", "E001"]),
        _face("F002", ["F001"], ["E001", "E002"], "cylinder"),
    ]
    edges = [
        _edge("E000", ["F000", "F001"]),
        _edge("E001", ["F001", "F002"], "concave"),
        _edge("E002", ["F002"], "seam"),
    ]
    g = build_face_graph(faces, edges, volume=10.0, area=30.0, bbox_dims=[1, 1, 1])
    checks = check_graph_invariants(g, faces, edges)
    assert all(checks.values()), checks
    assert g.edge_index.shape == (2, 4)
    faces[2].adjacent_face_ids.append("F000")  # corrupt: adjacency without a shared edge
    g_bad = build_face_graph(faces, edges + [_edge("E003", ["F002", "F002"], "seam")], 10.0, 30.0, [1, 1, 1])
    assert check_graph_invariants(g_bad, faces, edges)["no_self_loops"]
