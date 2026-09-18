"""CAD2ML worker: claims jobs, runs isolated processing, heartbeats its lease.

Recovery guarantees (tested in tests/integration/test_jobs_api_worker.py):
  * worker killed mid-job  -> lease expires -> reaper moves job to failed_retryable -> queued
  * queue emptied/restarted -> reconciler re-enqueues durable ``queued`` jobs from the DB
  * duplicate deliveries    -> DB compare-and-set claim; losers ack and skip
  * partial artifacts       -> staging dir + atomic promote; manifest written last
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from prometheus_client import start_http_server

from cad2ml.config import PipelineConfig, Settings, get_settings
from cad2ml.errors import ERROR_CODES
from cad2ml.jobs import service
from cad2ml.jobs.db import Database, FileRow, utcnow
from cad2ml.jobs.queue import JobQueue, make_redis
from cad2ml.jobs.states import JobState
from cad2ml.observability import metrics
from cad2ml.observability.logging import bind, clear, configure_logging, get_logger
from cad2ml.pipeline import process_source
from cad2ml.storage.base import LocalFSStore

PIPELINE_STAGE_TO_STATE = {
    "normalizing": JobState.normalizing,
    "extracting": JobState.extracting,
    "validating_outputs": JobState.validating_outputs,
}


class Worker:
    def __init__(
        self, settings: Settings, worker_id: str | None = None, *, create_schema: bool = False
    ) -> None:
        self.s = settings
        self.worker_id = worker_id or f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:6]}"
        self.db = Database(settings.database_url, create=create_schema)
        self.queue = JobQueue(make_redis(settings.queue_url, settings.queue_backend))
        self.store = LocalFSStore(settings.data_dir)
        self.cfg = PipelineConfig()
        self.log = get_logger("cad2ml.worker")
        self.current: str | None = None
        self.stop = threading.Event()
        self._last_maintenance = 0.0

    # ------------------------------------------------------------------ lifecycle
    def _heartbeat_loop(self) -> None:
        while not self.stop.wait(self.s.worker_heartbeat_s):
            try:
                service.heartbeat(self.db, self.worker_id, self.current, self.s.worker_lease_s)
            except Exception as exc:  # keep heartbeating through transient DB errors
                self.log.warning("heartbeat_failed", error_code=type(exc).__name__)

    def maintenance(self) -> None:
        service.reap_expired_leases(self.db, self.queue)
        service.reconcile_queue(self.db, self.queue)
        metrics.QUEUE_DEPTH.set(self.queue.depth())

    def run(self, max_jobs: int | None = None, idle_exit_s: float | None = None) -> int:
        released = self.queue.release_worker(self.worker_id)
        service.heartbeat(self.db, self.worker_id, None, self.s.worker_lease_s)
        hb = threading.Thread(target=self._heartbeat_loop, daemon=True)
        hb.start()
        metrics.ACTIVE_WORKERS.inc()
        self.log.info("worker_started", result="ok", worker=self.worker_id, released=released)
        done, idle_since = 0, time.monotonic()
        try:
            while not self.stop.is_set():
                if time.monotonic() - self._last_maintenance > max(5, self.s.worker_heartbeat_s):
                    self.maintenance()
                    self._last_maintenance = time.monotonic()
                job_id = self.queue.reserve(self.worker_id, timeout_s=2)
                if job_id is None:
                    if idle_exit_s is not None and time.monotonic() - idle_since > idle_exit_s:
                        break
                    continue
                self.handle(job_id)
                idle_since = time.monotonic()
                done += 1
                if max_jobs is not None and done >= max_jobs:
                    break
        finally:
            metrics.ACTIVE_WORKERS.dec()
            self.stop.set()
        return done

    # ------------------------------------------------------------------ job handling
    def handle(self, job_id: str) -> None:
        job = self.db.get_job(job_id)
        if job is None:
            self.queue.ack(self.worker_id, job_id)
            return
        first = JobState.training if job.kind == "train" else JobState.parsing
        claimed = service.claim(self.db, job_id, self.worker_id, self.s.worker_lease_s, first)
        if claimed is None:  # duplicate delivery or already handled
            self.queue.ack(self.worker_id, job_id)
            return
        self.current = job_id
        service.heartbeat(self.db, self.worker_id, job_id, self.s.worker_lease_s)
        bind(correlation_id=claimed.correlation_id, job_id=job_id)
        t0 = time.perf_counter()
        try:
            if claimed.kind == "train":
                self._run_training(claimed)
            else:
                self._run_processing(claimed)
        except Exception as exc:
            self.log.exception("job_internal_error", error_code="INTERNAL_ERROR")
            state = JobState(self.db.get_job(job_id).state)  # type: ignore[union-attr]
            if state not in (JobState.completed,) and not service.is_done(state):
                self.db.transition(
                    job_id,
                    state,
                    JobState.failed_terminal,
                    detail={"error": str(exc)[:500]},
                    error_code="INTERNAL_ERROR",
                    error_message=str(exc)[:2000],
                    finished_at=utcnow(),
                    worker_id=self.worker_id,
                )
                metrics.JOBS_FAILED.labels(state="failed_terminal").inc()
        finally:
            metrics.PROCESSING_SECONDS.observe(time.perf_counter() - t0)
            self.current = None
            self.queue.ack(self.worker_id, job_id)
            clear()

    def _run_processing(self, job: Any) -> None:
        with self.db.session() as s:
            f = s.get(FileRow, job.file_id)
            assert f is not None
            data = self.store.get_bytes(f.storage_key)
            filename = f.filename
        state = {"cur": JobState.parsing}

        def progress(stage: str, info: dict[str, Any]) -> None:
            target = PIPELINE_STAGE_TO_STATE.get(stage)
            if target is not None and target != state["cur"]:
                if self.db.transition(
                    job.job_id, state["cur"], target, detail=info, worker_id=self.worker_id
                ):
                    state["cur"] = target
            elif stage in ("validated", "parsing"):
                self.db.add_event(job.job_id, f"stage:{stage}", info, self.worker_id)

        fault = float(job.params.get("fault_parse_sleep_s", 0.0)) if self.s.enable_fault_injection else 0.0
        if (
            self.s.enable_fault_injection
            and job.params.get("fault_crash_worker_after_claim_s")
            and job.attempts == 1
        ):
            time.sleep(float(job.params["fault_crash_worker_after_claim_s"]))  # test hook: killed externally
        m = process_source(filename, data, self.store, self.cfg, progress=progress, fault_parse_sleep_s=fault)
        if m.status == "completed":
            ok = self.db.transition(
                job.job_id,
                state["cur"],
                JobState.completed,
                worker_id=self.worker_id,
                detail={"sample_id": m.sample_id},
                sample_id=m.sample_id,
                finished_at=utcnow(),
                result={
                    "sample_id": m.sample_id,
                    "status": "completed",
                    "face_count": m.geometry.face_count if m.geometry else None,
                },
            )
            if ok:
                metrics.JOBS_COMPLETED.inc()
            return
        code = m.rejection.code if m.rejection else "INTERNAL_ERROR"
        target = service.outcome_state(code)
        if target == JobState.failed_retryable:
            if self.db.transition(
                job.job_id,
                state["cur"],
                JobState.failed_retryable,
                detail={"code": code},
                error_code=code,
                worker_id=self.worker_id,
            ):
                refreshed = self.db.get_job(job.job_id)
                if refreshed is not None and refreshed.attempts < refreshed.max_attempts:
                    self.db.transition(
                        job.job_id,
                        JobState.failed_retryable,
                        JobState.queued,
                        detail={"retry": refreshed.attempts + 1},
                    )
                    self.queue.enqueue(job.job_id)
                else:
                    self.db.transition(
                        job.job_id,
                        JobState.failed_retryable,
                        JobState.failed_terminal,
                        detail={"reason": "max_attempts_exhausted"},
                        finished_at=utcnow(),
                    )
            return
        self.db.transition(
            job.job_id,
            state["cur"],
            target,
            worker_id=self.worker_id,
            sample_id=m.sample_id,
            detail={
                "code": code,
                "message": (m.rejection.message if m.rejection else "")[:500],
                "retryable": ERROR_CODES[code].retryable,
            },
            error_code=code,
            error_message=m.rejection.message if m.rejection else None,
            finished_at=utcnow(),
            result={"sample_id": m.sample_id, "status": m.status.value},
        )
        if target == JobState.quarantined:
            metrics.JOBS_QUARANTINED.inc()
        else:
            metrics.JOBS_FAILED.labels(state=target.value).inc()

    def _run_training(self, job: Any) -> None:
        from cad2ml.training.train import TrainConfig, run_training

        cfg = TrainConfig(**{k: v for k, v in job.params.items() if k in TrainConfig.__dataclass_fields__})
        res = run_training(self.store, job.params["dataset_id"], cfg)
        self.db.transition(
            job.job_id,
            JobState.training,
            JobState.completed,
            worker_id=self.worker_id,
            detail={"run_id": res["run_id"]},
            finished_at=utcnow(),
            result={
                "run_id": res["run_id"],
                "test": res["metrics"]["test"]["summary"],
                "best_epoch": res["best_epoch"],
            },
        )
        metrics.JOBS_COMPLETED.inc()


def main() -> int:
    ap = argparse.ArgumentParser("cad2ml-worker")
    ap.add_argument("--max-jobs", type=int, default=None)
    ap.add_argument("--idle-exit-s", type=float, default=None)
    ap.add_argument("--metrics-port", type=int, default=None)
    ap.add_argument("--create-schema", action="store_true", help="dev/test only; compose uses alembic")
    a = ap.parse_args()
    s = get_settings()
    configure_logging(s.log_level, s.log_json)
    Path(s.data_dir).mkdir(parents=True, exist_ok=True)
    port = a.metrics_port if a.metrics_port is not None else s.metrics_port
    if port:
        start_http_server(port)
    w = Worker(s, create_schema=a.create_schema)

    def _stop(*_: Any) -> None:
        w.stop.set()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    w.run(max_jobs=a.max_jobs, idle_exit_s=a.idle_exit_s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
