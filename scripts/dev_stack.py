"""Native development stack for hosts without Docker (Windows/macOS/Linux).

Runs: fakeredis TCP server (Redis protocol), alembic migrations on SQLite, uvicorn API, N workers.
This is for development/review only. The deployment path is docker compose (PostgreSQL + Valkey).

    python scripts/dev_stack.py --data data/dev --workers 1 --port 8000
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def main() -> int:
    from fakeredis import TcpFakeServer

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dev")
    ap.add_argument("--corpus", default="fixtures/corpus")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    data = Path(a.data).resolve()
    data.mkdir(parents=True, exist_ok=True)
    qport = _free_port()
    TcpFakeServer.daemon_threads = True
    TcpFakeServer.block_on_close = False
    server = TcpFakeServer(("127.0.0.1", qport), server_type="valkey")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT),
        "OPENBLAS_NUM_THREADS": "1",
        "CAD2ML_DATA_DIR": str(data),
        "CAD2ML_DATABASE_URL": f"sqlite:///{(data / 'cad2ml.db').as_posix()}",
        "CAD2ML_QUEUE_URL": f"redis://127.0.0.1:{qport}/0",
        "CAD2ML_CORPUS_DIR": str(Path(a.corpus).resolve()),
        "CAD2ML_METRICS_PORT": "0",
    }
    subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], cwd=ROOT, env=env, check=True)
    procs = [
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "apps.api.main:app",
                "--port",
                str(a.port),
                "--log-level",
                "warning",
            ],
            cwd=ROOT,
            env=env,
        )
    ]
    procs += [
        subprocess.Popen([sys.executable, "-m", "apps.worker.main"], cwd=ROOT, env=env)
        for _ in range(a.workers)
    ]
    print(f"CAD2ML dev stack: http://127.0.0.1:{a.port}/  (queue port {qport}, data {data})", flush=True)
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    try:
        while not stop.is_set() and all(p.poll() is None for p in procs):
            time.sleep(1)
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(20)
            except subprocess.TimeoutExpired:
                p.kill()
        server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
