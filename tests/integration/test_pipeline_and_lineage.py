"""End-to-end pipeline through isolated processes: completion, idempotency, failures, lineage (R1/R2 gates)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cad2ml.config import PIPELINE_VERSION, IngestionConfig, PipelineConfig, TessellationConfig
from cad2ml.lineage import trace_face
from cad2ml.pipeline import load_manifest, process_source
from cad2ml.schemas.manifest import Manifest
from cad2ml.storage.base import LocalFSStore


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> LocalFSStore:
    return LocalFSStore(tmp_path_factory.mktemp("pipe"))


@pytest.fixture(scope="module")
def completed(store: LocalFSStore, box_with_hole_step: Path) -> Manifest:
    return process_source(box_with_hole_step.name, box_with_hole_step.read_bytes(), store)


def test_upload_to_completed_manifest(completed: Manifest, store: LocalFSStore) -> None:
    m = completed
    assert m.status == "completed", m.rejection
    Manifest.model_validate_json(
        store.get_bytes(f"samples/{m.sample_id}/manifest.json")
    )  # schema-valid on disk
    assert m.geometry.face_count == 7 and m.geometry.solid_count == 1
    assert m.geometry.volume_mm3 == pytest.approx(40 * 30 * 10 - 3.14159265 * 16 * 10, rel=1e-6)
    assert m.source.original_units == "mm" and m.validation.source_valid and not m.validation.repair_attempted
    assert all(m.quality.checks.values())
    for ref in m.artifacts.values():  # every recorded artifact exists with the recorded size
        assert store.size(f"samples/{m.sample_id}/{ref.key}") == ref.bytes
    assert {f.feature_type for f in m.features} == {"through_hole", "planar_face"}
    assert store.exists(m.lineage["source_key"])  # immutable source retained
    assert not list(store.list("staging"))  # no partial staging leftovers


def test_idempotent_resubmission_reuses_sample(
    completed: Manifest, store: LocalFSStore, box_with_hole_step: Path
) -> None:
    events: list[str] = []
    again = process_source(
        "renamed_copy.step", box_with_hole_step.read_bytes(), store, progress=lambda s, i: events.append(s)
    )
    assert again.sample_id == completed.sample_id and events == ["reused"]
    assert again.processing.processed_at == completed.processing.processed_at


def test_trace_face_across_all_representations(completed: Manifest, store: LocalFSStore) -> None:
    """R2 gate: a selected face is traceable through every representation, with each link verified."""
    hole = next(f for f in completed.features if f.feature_type == "through_hole")
    face_id = hole.participating_faces[0]
    t = trace_face(store, completed.sample_id, face_id)
    assert t["all_links_verified"], t["verified"]
    chain = t["chain"]
    assert chain["brep_face"]["surface_type"] == "cylinder"
    assert chain["point_cloud"]["n_points"] >= 8 and chain["mesh"]["n_triangles"] > 0
    assert sum(chain["views"]["pixels_per_view"].values()) > 0
    assert chain["features (inferred)"][0]["feature_type"] == "through_hole"
    assert chain["source"]["sha256"] == completed.source.sha256
    assert chain["processing"]["pipeline_version"] == PIPELINE_VERSION


@pytest.mark.parametrize(
    "name,data,code,status",
    [
        ("empty.step", b"", "EMPTY_FILE", "rejected"),
        ("x.txt", b"ISO-10303-21;\n", "UNSUPPORTED_EXTENSION", "rejected"),
        (
            "junk.step",
            b"ISO-10303-21;\nHEADER;\nthis is not step\n",
            "STEP_PARSE_FAILED|STEP_NO_SHAPES",
            "rejected",
        ),
    ],
)
def test_invalid_inputs_rejected_with_codes(
    store: LocalFSStore, name: str, data: bytes, code: str, status: str
) -> None:
    m = process_source(name, data, store)
    assert m.status == status and m.rejection is not None and m.rejection.code in code.split("|")
    assert load_manifest(store, m.sample_id) is not None  # persisted: resubmission returns the same outcome


def test_failure_fixtures_have_expected_outcomes(mini_corpus: Path, tmp_path: Path) -> None:
    store = LocalFSStore(tmp_path)
    expected = json.loads((mini_corpus / "failures" / "expected.json").read_text())
    for name, exp in expected.items():
        path = mini_corpus / "failures" / name
        if not path.exists():
            continue
        m = process_source(name, path.read_bytes(), store)
        if exp["code"] in ("COMPLETED", "DUPLICATE"):
            assert m.status == "completed", (name, m.rejection)
        else:
            assert m.rejection is not None and m.rejection.code in exp["code"].split("|"), (name, m.rejection)
    tiny = process_source(
        "t.step", (mini_corpus / "failures" / "tiny_feature_hole_0p2mm.step").read_bytes(), store
    )
    assert any(
        f.feature_type == "through_hole" and abs(f.parameters["diameter_mm"] - 0.2) < 1e-6
        for f in tiny.features
    )


def test_parser_timeout_is_enforced(tmp_path: Path, box_with_hole_step: Path) -> None:
    cfg = PipelineConfig(ingestion=IngestionConfig(parse_timeout_s=2.0))
    store = LocalFSStore(tmp_path)
    m = process_source("slow.step", box_with_hole_step.read_bytes(), store, cfg, fault_parse_sleep_s=30)
    assert m.status == "quarantined" and m.rejection.code == "PARSER_TIMEOUT"


def test_failed_extraction_leaves_no_partial_sample(tmp_path: Path, box_with_hole_step: Path) -> None:
    cfg = PipelineConfig(tessellation=TessellationConfig(min_triangles_per_face=10**9))
    store = LocalFSStore(tmp_path)
    m = process_source("p.step", box_with_hole_step.read_bytes(), store, cfg)
    assert m.status == "quarantined" and m.rejection.code == "TESSELLATION_FAILED"
    assert list(store.list(f"samples/{m.sample_id}")) == [f"samples/{m.sample_id}/manifest.json"]
    assert not list(store.list("staging"))


def test_forced_reprocess_replaces_and_keeps_superseded(tmp_path: Path, box_with_hole_step: Path) -> None:
    store = LocalFSStore(tmp_path)
    bad_cfg = PipelineConfig(tessellation=TessellationConfig(min_triangles_per_face=10**9))
    first = process_source("p.step", box_with_hole_step.read_bytes(), store, bad_cfg)
    assert first.status == "quarantined"
    # same sample id (same config), explicit reprocess after the failure record exists
    again = process_source("p.step", box_with_hole_step.read_bytes(), store, bad_cfg, reuse_existing=False)
    assert again.status == "quarantined"
    good = process_source("p.step", box_with_hole_step.read_bytes(), store)
    good2 = process_source("p.step", box_with_hole_step.read_bytes(), store, reuse_existing=False)
    stored = load_manifest(store, good2.sample_id)
    assert stored is not None and stored.processing.processed_at == good2.processing.processed_at
    assert good.sample_id == good2.sample_id and any(
        k.startswith("superseded/") for k in store.list("superseded")
    )
