"""Job orchestration shared by API and worker: submission, claiming, leases, recovery."""

from __future__ import annotations

import hashlib
import uuid
from datetime import timedelta
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError

from cad2ml.config import PIPELINE_VERSION, PipelineConfig, Settings
from cad2ml.errors import ERROR_CODES, Outcome
from cad2ml.jobs.db import Database, FileRow, JobEventRow, JobRow, WorkerRow, new_id, utcnow
from cad2ml.jobs.queue import JobQueue
from cad2ml.jobs.states import IN_FLIGHT, TERMINAL, JobState
from cad2ml.observability import metrics
from cad2ml.observability.logging import get_logger

log = get_logger("cad2ml.jobs")


def process_idempotency_key(sha256: str, cfg: PipelineConfig) -> str:
    return f"{sha256}:{PIPELINE_VERSION}:{cfg.config_hash()}"


def register_file(
    db: Database, sha256: str, filename: str, size: int, storage_key: str
) -> tuple[FileRow, bool]:
    """Returns (row, created). Duplicate uploads resolve to the existing row by content hash."""
    with db.session() as s:
        row = s.execute(select(FileRow).where(FileRow.sha256 == sha256)).scalar_one_or_none()
        if row is not None:
            return row, False
        row = FileRow(
            file_id=new_id("f"), sha256=sha256, filename=filename, size_bytes=size, storage_key=storage_key
        )
        s.add(row)
    return row, True


def submit_job(
    db: Database,
    queue: JobQueue,
    *,
    kind: str,
    idempotency_key: str,
    file_id: str | None,
    params: dict[str, Any],
    max_attempts: int,
    correlation_id: str | None = None,
) -> tuple[JobRow, bool]:
    """Create (or return the existing) job for an idempotency key and enqueue it."""
    corr = correlation_id or uuid.uuid4().hex
    with db.session() as s:
        existing = s.execute(
            select(JobRow).where(JobRow.kind == kind, JobRow.idempotency_key == idempotency_key)
        ).scalar_one_or_none()
        if existing is not None:
            return existing, False
    job = JobRow(
        job_id=new_id("j"),
        kind=kind,
        idempotency_key=idempotency_key,
        file_id=file_id,
        state=JobState.received.value,
        attempts=0,
        max_attempts=max_attempts,
        correlation_id=corr,
        params=params,
    )
    try:
        with db.session() as s:
            s.add(job)
            s.flush()
            s.add(
                JobEventRow(
                    job_id=job.job_id,
                    state=JobState.received.value,
                    detail={"idempotency_key": idempotency_key},
                )
            )
    except IntegrityError:  # concurrent identical submission won the race
        with db.session() as s:
            return s.execute(
                select(JobRow).where(JobRow.kind == kind, JobRow.idempotency_key == idempotency_key)
            ).scalar_one(), False
    db.transition(job.job_id, JobState.received, JobState.validated, detail={"check": "file registered"})
    db.transition(job.job_id, JobState.validated, JobState.queued, detail={})
    queue.enqueue(job.job_id)
    return db.get_job(job.job_id), True  # type: ignore[return-value]


def claim(db: Database, job_id: str, worker_id: str, lease_s: int, first_state: JobState) -> JobRow | None:
    now = utcnow()
    with db.session() as s:
        res = s.execute(
            update(JobRow)
            .where(JobRow.job_id == job_id, JobRow.state == JobState.queued.value)
            .values(
                state=first_state.value,
                worker_id=worker_id,
                attempts=JobRow.attempts + 1,
                lease_expires_at=now + timedelta(seconds=lease_s),
                started_at=now,
                updated_at=now,
            )
        )
        if res.rowcount != 1:  # type: ignore[attr-defined]
            return None
        job = s.get(JobRow, job_id)
        assert job is not None
        s.add(
            JobEventRow(
                job_id=job_id,
                state=first_state.value,
                attempt=job.attempts,
                worker_id=worker_id,
                detail={"claimed": True},
            )
        )
        return job


def heartbeat(db: Database, worker_id: str, job_id: str | None, lease_s: int) -> None:
    now = utcnow()
    with db.session() as s:
        w = s.get(WorkerRow, worker_id)
        if w is None:
            s.add(WorkerRow(worker_id=worker_id, heartbeat_at=now, current_job_id=job_id))
        else:
            w.heartbeat_at, w.current_job_id = now, job_id
        if job_id:
            s.execute(
                update(JobRow)
                .where(JobRow.job_id == job_id, JobRow.worker_id == worker_id)
                .values(lease_expires_at=now + timedelta(seconds=lease_s))
            )


def reap_expired_leases(db: Database, queue: JobQueue) -> list[str]:
    """In-flight jobs whose lease expired: worker is presumed dead -> retry or terminal."""
    now = utcnow()
    moved = []
    with db.session() as s:
        rows = list(
            s.execute(
                select(JobRow).where(
                    JobRow.state.in_([x.value for x in IN_FLIGHT]), JobRow.lease_expires_at < now
                )
            ).scalars()
        )
    for job in rows:
        if not db.transition(
            job.job_id,
            job.state,
            JobState.failed_retryable,
            detail={
                "reason": "WORKER_LOST",
                "worker_id": job.worker_id,
                "lease_expired_at": job.lease_expires_at.isoformat() if job.lease_expires_at else None,
            },
            error_code="WORKER_LOST",
            error_message="lease expired",
            worker_id=None,
        ):
            continue
        if job.attempts < job.max_attempts:
            db.transition(
                job.job_id,
                JobState.failed_retryable,
                JobState.queued,
                detail={"retry": job.attempts + 1, "of": job.max_attempts},
            )
            queue.enqueue(job.job_id)
            log.warning("job_requeued_after_worker_loss", job_id=job.job_id, result="requeued")
        else:
            db.transition(
                job.job_id,
                JobState.failed_retryable,
                JobState.failed_terminal,
                detail={"reason": "max_attempts_exhausted"},
                finished_at=utcnow(),
            )
            metrics.JOBS_FAILED.labels(state="failed_terminal").inc()
        moved.append(job.job_id)
    return moved


def reconcile_queue(db: Database, queue: JobQueue, min_age_s: int = 15) -> list[str]:
    """Re-enqueue durable ``queued`` jobs missing from the transport (e.g. after a Valkey restart)."""
    cutoff = utcnow() - timedelta(seconds=min_age_s)
    present = queue.queued_ids()
    with db.session() as s:
        ids = list(
            s.execute(
                select(JobRow.job_id).where(JobRow.state == JobState.queued.value, JobRow.updated_at < cutoff)
            ).scalars()
        )
    missing = [j for j in ids if j not in present]
    for j in missing:
        queue.enqueue(j)
    if missing:
        log.warning("queue_reconciled", result="reenqueued", count=len(missing))
    return missing


def outcome_state(code: str) -> JobState:
    o = ERROR_CODES[code].outcome
    return {
        Outcome.rejected: JobState.rejected,
        Outcome.quarantined: JobState.quarantined,
        Outcome.timed_out: JobState.timed_out,
        Outcome.failed_retryable: JobState.failed_retryable,
        Outcome.failed_terminal: JobState.failed_terminal,
    }[o]


def dataset_request_key(params: dict[str, Any]) -> str:
    return hashlib.sha256(repr(sorted(params.items())).encode()).hexdigest()[:32]


def is_done(state: str) -> bool:
    return JobState(state) in TERMINAL


def settings_summary(s: Settings) -> dict[str, Any]:
    return {"lease_s": s.worker_lease_s, "heartbeat_s": s.worker_heartbeat_s, "max_attempts": s.max_attempts}
