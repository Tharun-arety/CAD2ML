"""API + durable jobs + worker: idempotency, retries after worker death, queue loss, metrics.

Runs natively against SQLite and a fakeredis TCP server (real Redis protocol, shared across
processes). The same test runs against PostgreSQL/Valkey in compose via CAD2ML_TEST_DATABASE_URL /
CAD2ML_TEST_QUEUE_URL (use a separate database and Valkey DB index so running compose workers do not
consume test jobs).
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from cad2ml.config import Settings

ROOT = Path(__file__).resolve().parents[2]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="module")
def env(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    tmp = tmp_path_factory.mktemp("svc")
    db_url = os.environ.get("CAD2ML_TEST_DATABASE_URL", f"sqlite:///{(tmp / 'db.sqlite').as_posix()}")
    queue_url = os.environ.get("CAD2ML_TEST_QUEUE_URL")
    server = None
    if queue_url is None:
        from fakeredis import TcpFakeServer

        port = _free_port()
        TcpFakeServer.daemon_threads = True  # idle client sockets must not block interpreter exit
        TcpFakeServer.block_on_close = False
        server = TcpFakeServer(("127.0.0.1", port), server_type="valkey")
        threading.Thread(target=server.serve_forever, daemon=True).start()
        queue_url = f"redis://127.0.0.1:{port}/0"
    settings = Settings(
        data_dir=tmp / "data",
        database_url=db_url,
        queue_url=queue_url,
        queue_backend="valkey",
        worker_lease_s=4,
        worker_heartbeat_s=1,
        max_attempts=3,
        enable_fault_injection=True,
        log_json=True,
        metrics_port=0,
    )
    from apps.api.main import create_app

    app = create_app(settings, create_schema=True)
    with TestClient(app) as client:
        yield {"settings": settings, "client": client, "tmp": tmp, "app": app}
    if server is not None:
        server.shutdown()


def _worker(settings: Settings, wid: str = "test-worker") -> Any:
    from apps.worker.main import Worker

    return Worker(settings, worker_id=wid)


def _upload(client: TestClient, path: Path, name: str | None = None) -> dict[str, Any]:
    r = client.post("/v1/files", files={"file": (name or path.name, path.read_bytes(), "application/step")})
    assert r.status_code in (200, 201), r.text
    return r.json()


def _wait_state(client: TestClient, job_id: str, states: set[str], timeout: float = 120) -> dict[str, Any]:
    end = time.time() + timeout
    while time.time() < end:
        j = client.get(f"/v1/jobs/{job_id}").json()
        if j["state"] in states:
            return j
        time.sleep(0.3)
    raise AssertionError(f"job {job_id} did not reach {states}: {j}")


def test_upload_job_complete_and_idempotent(env: dict[str, Any], box_with_hole_step: Path) -> None:
    c = env["client"]
    f1 = _upload(c, box_with_hole_step)
    f2 = _upload(c, box_with_hole_step, "same_bytes_other_name.step")
    assert f1["file_id"] == f2["file_id"] and f2["duplicate"] is True
    r1 = c.post("/v1/jobs", json={"file_id": f1["file_id"]}, headers={"x-correlation-id": "corr-test-1"})
    r2 = c.post("/v1/jobs", json={"file_id": f1["file_id"]})
    assert r1.status_code == 202 and r2.status_code == 200
    job = r1.json()
    assert r2.json()["job_id"] == job["job_id"] and job["state"] == "queued"
    assert r1.headers["x-correlation-id"] == "corr-test-1" and job["correlation_id"] == "corr-test-1"
    w = _worker(env["settings"])
    assert w.run(max_jobs=1, idle_exit_s=10) == 1
    j = c.get(f"/v1/jobs/{job['job_id']}").json()
    assert j["state"] == "completed" and j["sample_id"] and j["attempts"] == 1
    states = [e["state"] for e in c.get(f"/v1/jobs/{job['job_id']}/events").json()["events"]]
    for s in [
        "received",
        "validated",
        "queued",
        "parsing",
        "normalizing",
        "extracting",
        "validating_outputs",
        "completed",
    ]:
        assert s in states
    sample = c.get(f"/v1/samples/{j['sample_id']}").json()
    assert sample["status"] == "completed" and sample["geometry"]["face_count"] == 7
    arts = c.get(f"/v1/samples/{j['sample_id']}/artifacts").json()["artifacts"]
    assert {"graph", "pointcloud", "mesh", "views", "lineage", "features"} <= set(arts)
    png = c.get(arts["view_0_rgb"]["url"])
    assert png.status_code == 200 and png.content[:4] == b"\x89PNG"
    face = next(f for f in sample["features"] if f["feature_type"] == "through_hole")["participating_faces"][
        0
    ]
    tr = c.get(f"/v1/samples/{j['sample_id']}/faces/{face}/trace").json()
    assert tr["all_links_verified"]
    assert c.get(f"/v1/samples/{j['sample_id']}/artifacts/..%2F..%2Fsecrets").status_code == 404
    assert c.get("/v1/samples/..%2Fdatasets/artifacts").status_code in (400, 404)


def test_worker_killed_mid_job_is_recovered(env: dict[str, Any], mini_corpus: Path) -> None:
    c, settings = env["client"], env["settings"]
    part = sorted((mini_corpus / "parts").glob("flange__flat*.step"))[0]
    f = _upload(c, part)
    job = c.post("/v1/jobs", json={"file_id": f["file_id"], "fault_crash_worker_after_claim_s": 120}).json()
    child_env = {
        **os.environ,
        "PYTHONPATH": str(ROOT),
        "CAD2ML_DATA_DIR": str(settings.data_dir),
        "CAD2ML_DATABASE_URL": settings.database_url,
        "CAD2ML_QUEUE_URL": settings.queue_url,
        "CAD2ML_WORKER_LEASE_S": "4",
        "CAD2ML_WORKER_HEARTBEAT_S": "1",
        "CAD2ML_ENABLE_FAULT_INJECTION": "true",
        "CAD2ML_METRICS_PORT": "0",
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "apps.worker.main", "--max-jobs", "1"],
        cwd=ROOT,
        env=child_env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        claimed = _wait_state(c, job["job_id"], {"parsing"}, timeout=90)
        assert claimed["worker_id"] and claimed["attempts"] == 1
    finally:
        proc.kill()  # hard kill: no cleanup, no ack, no state update
        proc.wait(30)
    time.sleep(settings.worker_lease_s + 1.5)  # heartbeats stopped -> lease expires
    w = _worker(settings, "recovery-worker")
    w.maintenance()  # reaper: in-flight + expired lease -> failed_retryable -> queued
    j = c.get(f"/v1/jobs/{job['job_id']}").json()
    assert j["state"] == "queued" and j["error_code"] == "WORKER_LOST"
    assert w.run(max_jobs=1, idle_exit_s=10) == 1
    j = c.get(f"/v1/jobs/{job['job_id']}").json()
    assert j["state"] == "completed" and j["attempts"] == 2
    ev = c.get(f"/v1/jobs/{job['job_id']}/events").json()["events"]
    assert [e["state"] for e in ev].count("failed_retryable") == 1
    assert any(e["detail"].get("reason") == "WORKER_LOST" for e in ev)


def test_queue_loss_is_reconciled_and_duplicates_are_harmless(env: dict[str, Any], mini_corpus: Path) -> None:
    c, settings = env["client"], env["settings"]
    part = sorted((mini_corpus / "parts").glob("clevis__plain*.step"))[0]
    f = _upload(c, part)
    job = c.post("/v1/jobs", json={"file_id": f["file_id"]}).json()
    w = _worker(settings, "queue-worker")
    w.queue.r.flushdb()  # simulate Valkey restart without persistence (this DB index only)
    assert w.queue.depth() == 0
    from cad2ml.jobs import service

    time.sleep(0.1)
    assert service.reconcile_queue(w.db, w.queue, min_age_s=0) == [job["job_id"]]
    w.queue.enqueue(job["job_id"])  # duplicate delivery
    assert w.run(max_jobs=2, idle_exit_s=5) == 2
    j = c.get(f"/v1/jobs/{job['job_id']}").json()
    assert j["state"] == "completed" and j["attempts"] == 1  # second delivery did not re-run the job


def test_invalid_geometry_job_is_rejected_and_timeout_classified(
    env: dict[str, Any], mini_corpus: Path
) -> None:
    c, settings = env["client"], env["settings"]
    bad = c.post("/v1/files", files={"file": ("x.step", b"\x89PNG\r\n", "application/step")})
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "NOT_STEP_CONTENT"
    f = _upload(c, mini_corpus / "failures" / "assembly_two_parts.step")
    job = c.post("/v1/jobs", json={"file_id": f["file_id"]}).json()
    slow = _upload(c, mini_corpus / "failures" / "_good_plate.step")
    tjob = c.post("/v1/jobs", json={"file_id": slow["file_id"], "fault_parse_sleep_s": 600}).json()
    w = _worker(settings, "reject-worker")
    w.cfg = w.cfg.model_copy(
        update={"ingestion": w.cfg.ingestion.model_copy(update={"parse_timeout_s": 3.0})}
    )
    assert w.run(max_jobs=2, idle_exit_s=5) == 2
    j = c.get(f"/v1/jobs/{job['job_id']}").json()
    assert j["state"] == "rejected" and j["error_code"] == "UNSUPPORTED_ASSEMBLY"
    t = c.get(f"/v1/jobs/{tjob['job_id']}").json()
    assert t["state"] == "timed_out" and t["error_code"] == "PARSER_TIMEOUT"


def test_health_and_metrics(env: dict[str, Any]) -> None:
    c = env["client"]
    assert c.get("/health/live").json()["status"] == "live"
    r = c.get("/health/ready")
    assert r.status_code == 200 and all(r.json()["checks"].values())
    text = c.get("/metrics").text
    for name in [
        "files_received_total",
        "jobs_completed_total",
        "jobs_failed_total",
        "jobs_quarantined_total",
        "processing_duration_seconds",
        "step_parse_duration_seconds",
        "representation_generation_seconds",
        "active_workers",
        "queue_depth",
        "artifact_bytes_written",
        "samples_by_rejection_reason",
    ]:
        assert name in text, name
