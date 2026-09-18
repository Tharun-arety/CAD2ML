"""Benchmark the asynchronous pipeline on the synthetic corpus and write a factual report.

Two modes:
  --api-url http://localhost:8000     use an already running stack (docker compose)
  --local-workers N                   start API (uvicorn), N worker processes, SQLite and a fakeredis TCP
                                      server natively (for hosts without Docker); peak worker RSS is measured

Every number in the report is computed from job records, manifests and wall-clock measurements of this run.
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # before numpy import; see DECISIONS D-009

import argparse
import json
import platform
import shutil
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TERMINAL = {"completed", "rejected", "failed_terminal", "timed_out", "quarantined"}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _pct(xs: list[float], q: float) -> float | None:
    return round(float(np.percentile(xs, q)), 3) if xs else None


class LocalStack:
    def __init__(self, workdir: Path, n_workers: int, corpus: Path) -> None:
        from fakeredis import TcpFakeServer

        shutil.rmtree(workdir, ignore_errors=True)
        workdir.mkdir(parents=True)
        qport, self.api_port = _free_port(), _free_port()
        TcpFakeServer.daemon_threads = True
        TcpFakeServer.block_on_close = False
        self.fake = TcpFakeServer(("127.0.0.1", qport), server_type="valkey")
        threading.Thread(target=self.fake.serve_forever, daemon=True).start()
        self.env = {
            **os.environ,
            "PYTHONPATH": str(ROOT),
            "CAD2ML_DATA_DIR": str(workdir / "data"),
            "CAD2ML_DATABASE_URL": f"sqlite:///{(workdir / 'bench.db').as_posix()}",
            "CAD2ML_QUEUE_URL": f"redis://127.0.0.1:{qport}/0",
            "CAD2ML_CORPUS_DIR": str(corpus),
            "CAD2ML_LOG_LEVEL": "WARNING",
            "OPENBLAS_NUM_THREADS": "1",
            "CAD2ML_METRICS_PORT": "0",
        }
        subprocess.run(
            [sys.executable, "-m", "alembic", "upgrade", "head"], cwd=ROOT, env=self.env, check=True
        )
        self.procs: list[subprocess.Popen[bytes]] = []
        self.api = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "apps.api.main:app",
                "--port",
                str(self.api_port),
                "--log-level",
                "warning",
            ],
            cwd=ROOT,
            env=self.env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.workers = [
            subprocess.Popen(
                [sys.executable, "-m", "apps.worker.main"],
                cwd=ROOT,
                env=self.env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            for _ in range(n_workers)
        ]
        self.url = f"http://127.0.0.1:{self.api_port}"
        for _ in range(120):
            try:
                if httpx.get(self.url + "/health/ready", timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        self.peak_rss = 0
        self._stop = threading.Event()
        threading.Thread(target=self._sample_memory, daemon=True).start()

    def _sample_memory(self) -> None:
        import psutil

        while not self._stop.wait(0.2):
            total = 0
            for w in self.workers:
                try:
                    p = psutil.Process(w.pid)
                    total += p.memory_info().rss + sum(
                        c.memory_info().rss for c in p.children(recursive=True)
                    )
                except psutil.Error:
                    continue
            self.peak_rss = max(self.peak_rss, total)

    def close(self) -> None:
        self._stop.set()
        for p in [*self.workers, self.api]:
            p.terminate()
        for p in [*self.workers, self.api]:
            try:
                p.wait(20)
            except subprocess.TimeoutExpired:
                p.kill()
        self.fake.shutdown()


def submit_all(client: httpx.Client, files: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for f in files:
        r = client.post("/v1/files", files={"file": (f.name, f.read_bytes(), "application/step")})
        if r.status_code >= 400:
            rows.append(
                {
                    "file": f.name,
                    "state": "rejected",
                    "error_code": r.json()["error"]["code"],
                    "stage": "upload",
                    "job_id": None,
                }
            )
            continue
        j = client.post("/v1/jobs", json={"file_id": r.json()["file_id"]}).json()
        rows.append(
            {
                "file": f.name,
                "job_id": j["job_id"],
                "created": j["created"],
                "file_duplicate": r.json()["duplicate"],
            }
        )
    return rows


def wait_all(client: httpx.Client, rows: list[dict[str, Any]], timeout_s: float) -> None:
    end = time.time() + timeout_s
    pending = {r["job_id"] for r in rows if r.get("job_id")}
    while pending and time.time() < end:
        for jid in list(pending):
            j = client.get(f"/v1/jobs/{jid}").json()
            if j["state"] in TERMINAL:
                pending.discard(jid)
        time.sleep(1.0)
    if pending:
        raise TimeoutError(f"{len(pending)} jobs unfinished")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="fixtures/corpus")
    ap.add_argument("--api-url", default=None)
    ap.add_argument("--local-workers", type=int, default=2)
    ap.add_argument("--workdir", default="data/benchmark")
    ap.add_argument("--report", default="docs/benchmark.json")
    ap.add_argument("--timeout", type=float, default=3600)
    a = ap.parse_args()
    corpus = Path(a.corpus).resolve()
    files = sorted((corpus / "parts").glob("*.step")) + sorted(
        p
        for p in (corpus / "failures").glob("*")
        if p.name not in ("expected.json",) and not p.name.startswith("_")
    )
    stack = None
    if a.api_url:
        url, mode = a.api_url, "existing stack (api-url)"
    else:
        stack = LocalStack(Path(a.workdir).resolve(), a.local_workers, corpus)
        url, mode = (
            stack.url,
            f"native local stack: uvicorn + {a.local_workers} worker processes, SQLite, fakeredis TCP",
        )
    try:
        client = httpx.Client(base_url=url, timeout=120)
        t0 = time.perf_counter()
        rows = submit_all(client, files)
        wait_all(client, rows, a.timeout)
        wall = time.perf_counter() - t0
        jobs = {}
        for r in rows:
            if r.get("job_id"):
                jobs[r["job_id"]] = client.get(f"/v1/jobs/{r['job_id']}").json()
                r["state"], r["error_code"] = jobs[r["job_id"]]["state"], jobs[r["job_id"]]["error_code"]
        latencies: list[float] = []
        stage_t: dict[str, list[float]] = {}
        art_bytes: list[int] = []
        seen_samples = set()
        for jid, j in jobs.items():
            ev = client.get(f"/v1/jobs/{jid}/events").json()["events"]
            times = [datetime.fromisoformat(e["at"].rstrip("Z")).timestamp() for e in ev]
            claimed = next((t for e, t in zip(ev, times, strict=True) if e["state"] in ("parsing",)), None)
            if claimed is not None and j["state"] == "completed":
                latencies.append(times[-1] - claimed)
            if j["state"] == "completed" and j["sample_id"] not in seen_samples:
                seen_samples.add(j["sample_id"])
                m = client.get(f"/v1/samples/{j['sample_id']}").json()
                for k, v in m["processing"]["stage_timings_s"].items():
                    stage_t.setdefault(k, []).append(v)
                art_bytes.append(sum(x["bytes"] for x in m["artifacts"].values()))
        states: dict[str, int] = {}
        for r in rows:
            states[r["state"]] = states.get(r["state"], 0) + 1
        # idempotent rerun: resubmit everything; no new jobs should be created
        t1 = time.perf_counter()
        rerun = submit_all(client, files)
        rerun_wall = time.perf_counter() - t1
        new_jobs = sum(1 for r in rerun if r.get("created"))
        metrics_text = client.get("/metrics").text
        report = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "mode": mode,
            "host": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "cpu_count": os.cpu_count(),
            },
            "total_files": len(files),
            "outcomes": states,
            "successful_files": states.get("completed", 0),
            "rejected_files": states.get("rejected", 0),
            "failed_files": sum(
                v for k, v in states.items() if k in ("failed_terminal", "timed_out", "quarantined")
            ),
            "wall_clock_s": round(wall, 2),
            "throughput_files_per_min": round(len(files) / wall * 60, 2),
            "job_latency_s (claim->terminal, completed jobs)": {
                "n": len(latencies),
                "p50": _pct(latencies, 50),
                "p95": _pct(latencies, 95),
                "max": round(max(latencies), 3) if latencies else None,
            },
            "stage_timings_s": {
                k: {"p50": _pct(v, 50), "p95": _pct(v, 95)} for k, v in sorted(stage_t.items())
            },
            "artifact_bytes_per_sample": {
                "mean": int(np.mean(art_bytes)) if art_bytes else None,
                "p95": _pct([float(b) for b in art_bytes], 95),
                "total": int(sum(art_bytes)),
            },
            "peak_worker_rss_mb (all workers + isolated children)": round(stack.peak_rss / 2**20, 1)
            if stack
            else "not measured (external stack)",
            "idempotent_rerun": {
                "resubmitted_files": len(files),
                "new_jobs_created": new_jobs,
                "wall_clock_s": round(rerun_wall, 2),
            },
            "queue_depth_after": next(
                (ln.split()[-1] for ln in metrics_text.splitlines() if ln.startswith("queue_depth ")), None
            ),
            "per_file": rows,
        }
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(json.dumps(report, indent=1))
        print(json.dumps({k: v for k, v in report.items() if k != "per_file"}, indent=1))
    finally:
        if stack:
            stack.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
