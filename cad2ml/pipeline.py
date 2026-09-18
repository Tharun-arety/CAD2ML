"""Public processing entry point shared by the CLI, the worker and the benchmark.

``process_source`` takes the bytes of an uploaded STEP file and returns a validated
``Manifest``. Identity and idempotency:

    sample_id = "s_" + sha256(source_sha256 | pipeline_version | config_hash | schema_version)[:24]

If a final (non-retryable) manifest for that id already exists, it is returned without
reprocessing. Source bytes are stored once, content-addressed, and never modified.
Artifacts are written to ``staging/`` and promoted atomically; the manifest is written
last, so a sample directory either contains a complete, validated manifest or does
not exist.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cad2ml.config import (
    EXTRACTOR_VERSION,
    PIPELINE_VERSION,
    RECOGNIZER_VERSION,
    SCHEMA_VERSION,
    PipelineConfig,
    get_settings,
)
from cad2ml.errors import ERROR_CODES, Outcome, PipelineError
from cad2ml.ingestion.intake import IntakeResult, inspect_upload
from cad2ml.observability import metrics
from cad2ml.observability.logging import bind, get_logger
from cad2ml.parsers.isolation import run_isolated
from cad2ml.parsers.step_adapter import parse_step_child
from cad2ml.schemas.manifest import Manifest, ProcessingInfo, Rejection, SampleStatus, SourceInfo
from cad2ml.storage.base import ArtifactStore

ENVIRONMENT_SENSITIVE_CODES = frozenset({"PARSER_TIMEOUT", "PROCESSING_TIMEOUT", "CHILD_CRASHED"})
ProgressFn = Callable[[str, dict[str, Any]], None]
UNAVAILABLE_V1 = [
    "parametric_feature_history (not encoded in STEP AP203/AP214 geometry)",
    "design_intent / constraints / sketches",
    "material and manufacturing process",
    "assembly context (assemblies rejected in v1)",
    "PMI / GD&T annotations (not extracted in v1)",
]


def compute_sample_id(source_sha256: str, config_hash: str) -> str:
    raw = f"{source_sha256}|{PIPELINE_VERSION}|{config_hash}|{SCHEMA_VERSION}"
    return "s_" + hashlib.sha256(raw.encode()).hexdigest()[:24]


def source_key(sha: str) -> str:
    return f"sources/{sha[:2]}/{sha}.step"


def manifest_key(sample_id: str) -> str:
    return f"samples/{sample_id}/manifest.json"


def load_manifest(store: ArtifactStore, sample_id: str) -> Manifest | None:
    key = manifest_key(sample_id)
    if not store.exists(key):
        return None
    return Manifest.model_validate_json(store.get_bytes(key))


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _status_for(outcome: Outcome) -> SampleStatus:
    return {
        Outcome.rejected: SampleStatus.rejected,
        Outcome.quarantined: SampleStatus.quarantined,
        Outcome.timed_out: SampleStatus.quarantined,
    }.get(outcome, SampleStatus.failed)


def process_source(
    filename: str,
    data: bytes,
    store: ArtifactStore,
    cfg: PipelineConfig | None = None,
    *,
    progress: ProgressFn | None = None,
    fault_parse_sleep_s: float = 0.0,
    reuse_existing: bool = True,
) -> Manifest:
    cfg = cfg or PipelineConfig()
    log = get_logger("cad2ml.pipeline")
    chash = cfg.config_hash()
    emit = progress or (lambda stage, info: None)
    timings: dict[str, float] = {}
    t_start = time.perf_counter()

    intake: IntakeResult | None = None
    try:
        intake = inspect_upload(filename, data, cfg.ingestion)
    except PipelineError as e:
        sha = hashlib.sha256(data).hexdigest()
        sample_id = compute_sample_id(sha, chash)
        bind(sample_id=sample_id)
        return _finalize_failure(
            store, e, sample_id, filename, sha, len(data), None, chash, timings, emit, log
        )

    sample_id = compute_sample_id(intake.sha256, chash)
    bind(sample_id=sample_id)
    if reuse_existing:
        existing = load_manifest(store, sample_id)
        if existing is not None:
            log.info("sample_reused", result="idempotent_hit")
            emit("reused", {"sample_id": sample_id, "status": existing.status})
            return existing

    if not store.exists(source_key(intake.sha256)):
        store.put_bytes(source_key(intake.sha256), data)
    emit("validated", {"sample_id": sample_id, "sha256": intake.sha256})

    work = Path(tempfile.mkdtemp(prefix="cad2ml-"))
    staging_key = f"staging/{sample_id}-{uuid.uuid4().hex[:8]}"
    try:
        src = work / "source.step"
        src.write_bytes(data)
        brep = work / "parsed.brep"
        emit("parsing", {})
        t0 = time.perf_counter()
        parsed = run_isolated(
            parse_step_child,
            (str(src), str(brep), fault_parse_sleep_s),
            timeout_s=cfg.ingestion.parse_timeout_s,
            stage="parsing",
            timeout_code="PARSER_TIMEOUT",
            memory_mb=get_settings().child_memory_limit_mb,
        )
        timings["parse_isolated"] = round(time.perf_counter() - t0, 6)
        metrics.STEP_PARSE_SECONDS.observe(timings["parse_isolated"])

        meta = {
            "sample_id": sample_id,
            "source": {"filename": intake.filename, "sha256": intake.sha256},
            "versions": {
                "pipeline": PIPELINE_VERSION,
                "extractor": EXTRACTOR_VERSION,
                "schema": SCHEMA_VERSION,
                "parser": parsed["parser_version"],
                "recognizer": RECOGNIZER_VERSION,
                "config_hash": chash,
            },
        }
        staging = store.local_path(staging_key)
        from cad2ml.extraction import extract_child

        t0 = time.perf_counter()
        result = run_isolated(
            extract_child,
            (str(brep), str(staging), cfg.canonical_json(), meta),
            timeout_s=max(cfg.ingestion.parse_timeout_s * 3, 180.0),
            stage="extracting",
            timeout_code="PROCESSING_TIMEOUT",
            memory_mb=get_settings().child_memory_limit_mb,
            on_progress=lambda p: emit(p["stage"], p),
        )
        timings["extract_isolated"] = round(time.perf_counter() - t0, 6)
        metrics.REPRESENTATION_SECONDS.observe(timings["extract_isolated"])
        timings.update({f"extract.{k}": v for k, v in result["timings"].items()})

        manifest = Manifest(
            sample_id=sample_id,
            status=SampleStatus.completed,
            source=SourceInfo(
                filename=intake.filename,
                sha256=intake.sha256,
                size_bytes=intake.size_bytes,
                original_units=parsed["declared_length_unit"],
                step_schema=intake.step_schema,
                originating_system=intake.originating_system,
            ),
            processing=ProcessingInfo(
                pipeline_version=PIPELINE_VERSION,
                parser_version=parsed["parser_version"],
                extractor_version=EXTRACTOR_VERSION,
                recognizer_version=RECOGNIZER_VERSION,
                schema_version=SCHEMA_VERSION,
                configuration_hash=chash,
                processed_at=_now(),
                stage_timings_s=timings,
            ),
            geometry=result["geometry"],
            validation=result["validation"],
            artifacts=result["artifacts"],
            observations=result["observations"],
            features=result["features"],
            quality=result["quality"],
            lineage={
                "source_key": source_key(intake.sha256),
                "lineage_artifact": "lineage.json",
                "derived_from": {
                    "source_sha256": intake.sha256,
                    "pipeline_version": PIPELINE_VERSION,
                    "configuration_hash": chash,
                    "schema_version": SCHEMA_VERSION,
                },
            },
            unavailable=UNAVAILABLE_V1,
        )
        manifest = Manifest.model_validate(manifest.model_dump())  # full schema validation
        timings["total"] = round(time.perf_counter() - t_start, 6)
        store.put_bytes(f"{staging_key}/config.json", cfg.canonical_json().encode())
        store.put_bytes(f"{staging_key}/manifest.json", manifest.model_dump_json(indent=1).encode())
        store.promote_prefix(staging_key, f"samples/{sample_id}")
        written = sum(r.bytes for r in manifest.artifacts.values())
        metrics.ARTIFACT_BYTES.inc(written)
        emit("completed", {"sample_id": sample_id, "artifact_bytes": written})
        log.info("sample_completed", result="completed", duration_ms=round(timings["total"] * 1000, 1))
        return manifest
    except PipelineError as e:
        store.delete_prefix(staging_key)
        return _finalize_failure(
            store,
            e,
            sample_id,
            intake.filename,
            intake.sha256,
            intake.size_bytes,
            intake,
            chash,
            timings,
            emit,
            log,
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)
        if store.exists(staging_key):
            store.delete_prefix(staging_key)


def _finalize_failure(
    store: ArtifactStore,
    e: PipelineError,
    sample_id: str,
    filename: str,
    sha: str,
    size: int,
    intake: IntakeResult | None,
    chash: str,
    timings: dict[str, float],
    emit: ProgressFn,
    log: Any,
) -> Manifest:
    spec = ERROR_CODES[e.code]
    status = _status_for(spec.outcome)
    manifest = Manifest(
        sample_id=sample_id,
        status=status,
        source=SourceInfo(
            filename=filename,
            sha256=sha,
            size_bytes=size,
            step_schema=intake.step_schema if intake else None,
            originating_system=intake.originating_system if intake else None,
        ),
        processing=ProcessingInfo(
            pipeline_version=PIPELINE_VERSION,
            parser_version="n/a",
            extractor_version=EXTRACTOR_VERSION,
            recognizer_version=RECOGNIZER_VERSION,
            schema_version=SCHEMA_VERSION,
            configuration_hash=chash,
            processed_at=_now(),
            stage_timings_s=timings,
        ),
        rejection=Rejection(code=e.code, stage=e.stage, message=e.message[:2000], retryable=spec.retryable),
        lineage={"source_key": source_key(sha) if intake else "not_stored (failed intake)"},
        unavailable=UNAVAILABLE_V1,
    )
    metrics.SAMPLES_BY_REJECTION.labels(reason=e.code).inc()
    body = manifest.model_dump_json(indent=1).encode()
    if e.code in ENVIRONMENT_SENSITIVE_CODES:
        # Audit record only: a timeout or native crash may be caused by the host (load, limits, runtime
        # libraries), so it must not permanently pin the sample's outcome. The job record keeps the state.
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
        store.put_bytes(f"quarantine/{sample_id}/{stamp}-{e.code}.json", body)
    elif not spec.retryable:
        # deterministic outcome: persisted so identical resubmissions return the same result (idempotency)
        store.put_bytes(manifest_key(sample_id), body)
    emit(spec.outcome.value, {"code": e.code, "message": e.message[:500]})
    log.warning("sample_not_completed", result=spec.outcome.value, error_code=e.code)
    return manifest


def manifest_summary(m: Manifest) -> dict[str, Any]:
    return json.loads(m.model_dump_json(include={"sample_id", "status", "rejection", "geometry"}))
