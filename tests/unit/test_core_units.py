"""Kernel-free unit tests: hashing, intake, config identity, storage safety, canonical ordering."""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

import pytest

from cad2ml.config import IngestionConfig, PipelineConfig, PointCloudConfig
from cad2ml.errors import PipelineError
from cad2ml.ingestion.intake import inspect_upload, sanitize_filename, sha256_bytes, sha256_stream
from cad2ml.parsers.step_adapter import count_entities, detect_length_unit
from cad2ml.pipeline import compute_sample_id
from cad2ml.storage.base import LocalFSStore, StorageKeyError, validate_key
from cad2ml.topology import canonical as C

STEP_HEAD = (
    b"ISO-10303-21;\nHEADER;\nFILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));\nENDSEC;\n"
)


def test_sha256_bytes_and_stream_agree() -> None:
    data = b"abc" * 1000
    assert sha256_bytes(data) == hashlib.sha256(data).hexdigest() == sha256_stream([data[:7], data[7:]])


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("../../etc/passwd", "passwd"),
        ("C:\\Users\\x\\part 1.STEP", "part_1.STEP"),
        ("", "unnamed"),
        ("héllo<script>.step", "hello_script_.step"),
        ("...", "unnamed"),
    ],
)
def test_sanitize_filename(raw: str, expected: str) -> None:
    assert sanitize_filename(raw) == expected


def test_intake_rejections() -> None:
    cfg = IngestionConfig(max_file_bytes=1000)
    cases = [
        ("a.step", b"", "EMPTY_FILE"),
        ("a.step", b"x" * 1001, "FILE_TOO_LARGE"),
        ("a.txt", STEP_HEAD, "UNSUPPORTED_EXTENSION"),
        ("a.step", b"\x89PNG....", "NOT_STEP_CONTENT"),
        ("noext", STEP_HEAD, "UNSUPPORTED_EXTENSION"),
    ]
    for name, data, code in cases:
        with pytest.raises(PipelineError) as e:
            inspect_upload(name, data, cfg)
        assert e.value.code == code


def test_intake_accepts_step_and_reads_schema() -> None:
    r = inspect_upload("Part.STP", STEP_HEAD, IngestionConfig())
    assert r.sha256 == sha256_bytes(STEP_HEAD) and r.step_schema and "214" in r.step_schema


def test_unit_detection_and_entity_count() -> None:
    assert detect_length_unit("#1=( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.) );") == "mm"
    assert detect_length_unit("#9=CONVERSION_BASED_UNIT('INCH',#10);") == "inch"
    assert detect_length_unit("#1=( SI_UNIT($,.METRE.) );") == "m"
    assert detect_length_unit("nothing") is None
    assert (
        count_entities(
            "#1=NEXT_ASSEMBLY_USAGE_OCCURRENCE('a');#2 = NEXT_ASSEMBLY_USAGE_OCCURRENCE ('b');",
            "NEXT_ASSEMBLY_USAGE_OCCURRENCE",
        )
        == 2
    )


def test_config_hash_deterministic_and_sensitive() -> None:
    a, b = PipelineConfig(), PipelineConfig()
    assert a.config_hash() == b.config_hash()
    c = PipelineConfig(pointcloud=PointCloudConfig(seed=99))
    assert c.config_hash() != a.config_hash()
    assert compute_sample_id("ab" * 32, a.config_hash()) != compute_sample_id("ab" * 32, c.config_hash())
    assert compute_sample_id("ab" * 32, a.config_hash()) == compute_sample_id("ab" * 32, b.config_hash())


@pytest.mark.parametrize("key", ["../x", "/abs", "a/../../b", "a//b", "a/./b", "a\\b", "", "a/", "a b"])
def test_storage_rejects_unsafe_keys(key: str) -> None:
    with pytest.raises(StorageKeyError):
        validate_key(key)


def test_storage_atomic_put_list_promote(tmp_path: Path) -> None:
    s = LocalFSStore(tmp_path)
    s.put_bytes("staging/x/a.json", b"1")
    s.put_bytes("staging/x/sub/b.bin", b"22")
    assert list(s.list("staging/x")) == ["staging/x/a.json", "staging/x/sub/b.bin"]
    assert not any(p.name.startswith(".tmp-") for p in tmp_path.rglob("*"))
    s.promote_prefix("staging/x", "samples/x")
    assert s.get_bytes("samples/x/sub/b.bin") == b"22" and not s.exists("staging/x")
    s.put_bytes("staging/y/a.json", b"other")
    s.promote_prefix("staging/y", "samples/x")  # existing destination is never overwritten
    assert s.get_bytes("samples/x/a.json") == b"1" and not s.exists("staging/y")


def _records(n: int, seed: int) -> list[tuple]:
    rng = random.Random(seed)
    return [
        (
            rng.choice([0, 1, 4]),
            [rng.uniform(-50, 50) for _ in range(3)],
            rng.uniform(1, 500),
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            f"fp{i}",
        )
        for i in range(n)
    ]


def test_canonical_order_is_traversal_invariant() -> None:
    recs = _records(40, 1)
    keys = [C.face_sort_key(*r) for r in recs]
    ordered = [recs[i][5] for i in C.canonical_order(keys)]
    for s in range(5):
        perm = recs[:]
        random.Random(s).shuffle(perm)
        pk = [C.face_sort_key(*r) for r in perm]
        assert [perm[i][5] for i in C.canonical_order(pk)] == ordered


def test_fingerprint_ignores_neighbour_order_and_matching_is_unambiguous() -> None:
    sig = C.intrinsic_face_signature("cylinder", 12.5, {"radius": 4.0, "axis": [0, 0, 1]}, 2, 3)
    assert C.face_fingerprint(sig, ["plane", "cylinder"]) == C.face_fingerprint(sig, ["cylinder", "plane"])
    assert C.match_entities(["a", "b", "c", "c"], ["c", "b", "x", "a"]) == {0: 3, 1: 1}


def test_exported_json_schemas_match_models() -> None:
    import json

    from cad2ml.schemas.manifest import Manifest

    root = Path(__file__).resolve().parents[2] / "schemas"
    assert json.loads((root / "manifest.schema.json").read_text()) == Manifest.model_json_schema()
    assert (
        json.loads((root / "pipeline_config.schema.json").read_text()) == PipelineConfig.model_json_schema()
    )


def test_step_files_accepts_stp_and_step_case_insensitive(tmp_path: Path) -> None:
    from cad2ml.ingestion.intake import step_files

    for n in ["a.step", "b.STP", "c.stp", "d.txt", "e.Step"]:
        (tmp_path / n).write_bytes(b"x")
    assert [p.name for p in step_files(tmp_path)] == ["a.step", "b.STP", "c.stp", "e.Step"]
