"""CAD2ML HTTP API (FastAPI)."""

from __future__ import annotations

import json
import time
import uuid
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from cad2ml.config import PipelineConfig, Settings, get_settings
from cad2ml.errors import PipelineError
from cad2ml.ingestion.intake import inspect_upload
from cad2ml.jobs import service
from cad2ml.jobs.db import Database, DatasetRow, FileRow, JobRow, WorkerRow, utcnow
from cad2ml.jobs.queue import JobQueue, make_redis
from cad2ml.lineage import LineageError, trace_face
from cad2ml.observability import metrics
from cad2ml.observability.logging import bind, clear, configure_logging, get_logger
from cad2ml.pipeline import load_manifest, source_key
from cad2ml.storage.base import LocalFSStore, StorageKeyError

STATIC = Path(__file__).parent / "static"


class JobRequest(BaseModel):
    file_id: str
    fault_parse_sleep_s: float | None = Field(default=None, description="test-only; requires fault injection")
    fault_crash_worker_after_claim_s: float | None = None


class DatasetRequest(BaseModel):
    seed: int = 7
    split_mode: str = Field(default="group", pattern="^(group|family)$")


class TrainingRequest(BaseModel):
    dataset_id: str
    model: str = Field(default="gnn", pattern="^(gnn|mlp)$")
    epochs: int = Field(default=150, ge=1, le=2000)
    seed: int = 0


class State:
    settings: Settings
    db: Database
    queue: JobQueue
    store: LocalFSStore
    cfg: PipelineConfig


def create_app(settings: Settings | None = None, *, create_schema: bool = False) -> FastAPI:
    st = State()
    st.settings = settings or get_settings()
    configure_logging(st.settings.log_level, st.settings.log_json)
    log = get_logger("cad2ml.api")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> Any:
        Path(st.settings.data_dir).mkdir(parents=True, exist_ok=True)
        st.db = Database(st.settings.database_url, create=create_schema)
        st.queue = JobQueue(make_redis(st.settings.queue_url, st.settings.queue_backend))
        st.store = LocalFSStore(st.settings.data_dir)
        st.cfg = PipelineConfig()
        yield

    app = FastAPI(
        title="CAD2ML",
        version="1.0.0",
        lifespan=lifespan,
        description="STEP -> canonical B-Rep -> aligned, traceable ML representations",
    )
    app.state.cad2ml = st

    def S() -> State:
        return st

    @app.middleware("http")
    async def correlation(request: Request, call_next: Any) -> Response:
        cid = request.headers.get("x-correlation-id") or uuid.uuid4().hex
        cid = "".join(c for c in cid if c.isalnum() or c in "-_")[:64] or uuid.uuid4().hex
        request.state.correlation_id = cid
        bind(correlation_id=cid)
        t0 = time.perf_counter()
        try:
            response: Response = await call_next(request)
        finally:
            clear()
        response.headers["x-correlation-id"] = cid
        if not request.url.path.startswith(("/metrics", "/health", "/ui")):
            log.info(
                "http_request",
                path=request.url.path,
                method=request.method,
                status=response.status_code,
                duration_ms=round((time.perf_counter() - t0) * 1000, 2),
                correlation_id=cid,
            )
        return response

    # ------------------------------------------------------------------ health / metrics
    @app.get("/health/live")
    def live() -> dict[str, str]:
        return {"status": "live"}

    @app.get("/health/ready")
    def ready(s: State = Depends(S)) -> JSONResponse:
        checks: dict[str, bool] = {}
        try:
            with s.db.session() as sess:
                sess.execute(select(func.count()).select_from(JobRow))
            checks["database"] = True
        except Exception:
            checks["database"] = False
        checks["queue"] = s.queue.ping()
        try:
            s.store.put_bytes("health/ready.probe", b"ok")
            checks["storage"] = True
        except Exception:
            checks["storage"] = False
        ok = all(checks.values())
        return JSONResponse(
            {"status": "ready" if ok else "not_ready", "checks": checks}, status_code=200 if ok else 503
        )

    @app.get("/metrics")
    def prom(s: State = Depends(S)) -> Response:
        try:
            metrics.QUEUE_DEPTH.set(s.queue.depth())
            cutoff = utcnow() - timedelta(seconds=s.settings.worker_heartbeat_s * 3)
            with s.db.session() as sess:
                n = sess.execute(
                    select(func.count()).select_from(WorkerRow).where(WorkerRow.heartbeat_at >= cutoff)
                ).scalar_one()
            metrics.ACTIVE_WORKERS.set(n)
        except Exception:
            pass
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    # ------------------------------------------------------------------ files & jobs
    @app.post("/v1/files", status_code=201)
    async def upload(request: Request, file: UploadFile = File(...), s: State = Depends(S)) -> JSONResponse:
        limit = s.cfg.ingestion.max_file_bytes
        data = await file.read(limit + 1)
        try:
            intake = inspect_upload(file.filename or "upload.step", data, s.cfg.ingestion)
        except PipelineError as e:
            metrics.SAMPLES_BY_REJECTION.labels(reason=e.code).inc()
            status = 413 if e.code == "FILE_TOO_LARGE" else 422
            return JSONResponse(
                {"error": {"code": e.code, "message": e.message, "stage": e.stage}}, status_code=status
            )
        key = source_key(intake.sha256)
        if not s.store.exists(key):
            s.store.put_bytes(key, data)
        row, created = service.register_file(s.db, intake.sha256, intake.filename, intake.size_bytes, key)
        metrics.FILES_RECEIVED.inc()
        return JSONResponse(
            {
                "file_id": row.file_id,
                "sha256": row.sha256,
                "filename": row.filename,
                "size_bytes": row.size_bytes,
                "duplicate": not created,
            },
            status_code=201 if created else 200,
        )

    @app.post("/v1/jobs", status_code=202)
    def create_job(req: JobRequest, request: Request, s: State = Depends(S)) -> JSONResponse:
        with s.db.session() as sess:
            f = sess.get(FileRow, req.file_id)
        if f is None:
            raise HTTPException(404, "unknown file_id")
        params: dict[str, Any] = {}
        key = service.process_idempotency_key(f.sha256, s.cfg)
        faults = {
            k: v
            for k, v in (
                ("fault_parse_sleep_s", req.fault_parse_sleep_s),
                ("fault_crash_worker_after_claim_s", req.fault_crash_worker_after_claim_s),
            )
            if v
        }
        if faults:
            if not s.settings.enable_fault_injection:
                raise HTTPException(400, "fault injection disabled")
            params.update(faults)
            key += ":fault:" + uuid.uuid4().hex  # fault runs are never deduplicated
        job, created = service.submit_job(
            s.db,
            s.queue,
            kind="process",
            idempotency_key=key,
            file_id=f.file_id,
            params=params,
            max_attempts=s.settings.max_attempts,
            correlation_id=request.state.correlation_id,
        )
        return JSONResponse({**_job(job), "created": created}, status_code=202 if created else 200)

    @app.get("/v1/jobs/{job_id}")
    def get_job(job_id: str, s: State = Depends(S)) -> dict[str, Any]:
        job = s.db.get_job(job_id)
        if job is None:
            raise HTTPException(404, "unknown job")
        return _job(job)

    @app.get("/v1/jobs/{job_id}/events")
    def job_events(job_id: str, s: State = Depends(S)) -> dict[str, Any]:
        if s.db.get_job(job_id) is None:
            raise HTTPException(404, "unknown job")
        return {
            "job_id": job_id,
            "events": [
                {
                    "event_id": e.event_id,
                    "state": e.state,
                    "attempt": e.attempt,
                    "worker_id": e.worker_id,
                    "detail": e.detail,
                    "at": e.created_at.isoformat() + "Z",
                }
                for e in s.db.events(job_id)
            ],
        }

    # ------------------------------------------------------------------ samples
    def _manifest(s: State, sample_id: str) -> Any:
        try:
            m = load_manifest(s.store, sample_id)
        except StorageKeyError as e:
            raise HTTPException(400, "invalid sample id") from e
        if m is None:
            raise HTTPException(404, "unknown sample")
        return m

    @app.get("/v1/samples/{sample_id}")
    def get_sample(sample_id: str, s: State = Depends(S)) -> Response:
        m = _manifest(s, sample_id)
        return Response(m.model_dump_json(), media_type="application/json")

    @app.get("/v1/samples/{sample_id}/artifacts")
    def list_artifacts(sample_id: str, s: State = Depends(S)) -> dict[str, Any]:
        m = _manifest(s, sample_id)
        return {
            "sample_id": sample_id,
            "status": m.status,
            "artifacts": {
                name: {**ref.model_dump(), "url": f"/v1/samples/{sample_id}/artifacts/{name}"}
                for name, ref in m.artifacts.items()
            },
        }

    @app.get("/v1/samples/{sample_id}/artifacts/{name}")
    def get_artifact(sample_id: str, name: str, s: State = Depends(S)) -> FileResponse:
        m = _manifest(s, sample_id)
        ref = m.artifacts.get(name)  # only keys recorded in the manifest are servable
        if ref is None:
            raise HTTPException(404, "unknown artifact")
        path = s.store.local_path(f"samples/{sample_id}/{ref.key}")
        return FileResponse(path, media_type=ref.media_type)

    @app.get("/v1/samples/{sample_id}/faces/{face_id}/trace")
    def face_trace(sample_id: str, face_id: str, s: State = Depends(S)) -> dict[str, Any]:
        _manifest(s, sample_id)
        try:
            return trace_face(s.store, sample_id, face_id)
        except LineageError as e:
            raise HTTPException(404, str(e)) from e

    @app.get("/v1/samples/{sample_id}/views/{view}/pick")
    def pick_face(sample_id: str, view: int, x: int, y: int, s: State = Depends(S)) -> dict[str, Any]:
        from cad2ml.evaluation.visualize import face_at_pixel

        m = _manifest(s, sample_id)
        if m.status != "completed":
            raise HTTPException(409, f"sample is {m.status}")
        return {
            "sample_id": sample_id,
            "view": view,
            "x": x,
            "y": y,
            "face_id": face_at_pixel(s.store, sample_id, view, x, y),
        }

    @app.get("/v1/samples/{sample_id}/views/{view}/highlight/{face_id}")
    def highlight(sample_id: str, view: int, face_id: str, s: State = Depends(S)) -> Response:
        from cad2ml.evaluation.visualize import lineage_figure

        m = _manifest(s, sample_id)
        if m.status != "completed" or not face_id.startswith("F") or not face_id[1:].isdigit():
            raise HTTPException(400, "invalid request")
        if m.geometry is None or int(face_id[1:]) >= m.geometry.face_count:
            raise HTTPException(404, "unknown face")
        return Response(lineage_figure(s.store, sample_id, face_id, view), media_type="image/png")

    @app.get("/v1/samples")
    def list_samples(limit: int = 200, s: State = Depends(S)) -> dict[str, Any]:
        with s.db.session() as sess:
            rows = list(
                sess.execute(
                    select(JobRow)
                    .where(JobRow.kind == "process", JobRow.sample_id.is_not(None))
                    .order_by(JobRow.created_at.desc())
                    .limit(min(limit, 1000))
                ).scalars()
            )
            files = {f.file_id: f.filename for f in sess.execute(select(FileRow)).scalars()}
        return {
            "samples": [
                {
                    "sample_id": r.sample_id,
                    "job_id": r.job_id,
                    "state": r.state,
                    "filename": files.get(r.file_id or ""),
                    "error_code": r.error_code,
                }
                for r in rows
            ]
        }

    # ------------------------------------------------------------------ datasets
    @app.post("/v1/datasets", status_code=201)
    def create_dataset(req: DatasetRequest, s: State = Depends(S)) -> JSONResponse:
        from cad2ml.datasets.builder import build_dataset

        ds = build_dataset(
            s.store, Path(s.settings.corpus_dir), seed=req.seed, split_mode=req.split_mode, cfg=s.cfg
        )
        created = False
        with s.db.session() as sess:
            if sess.get(DatasetRow, ds["dataset_id"]) is None:
                sess.add(
                    DatasetRow(
                        dataset_id=ds["dataset_id"],
                        split_mode=req.split_mode,
                        seed=req.seed,
                        summary={"split_sizes": ds["quality"]["split_sizes"]},
                    )
                )
                created = True
        return JSONResponse(_dataset_summary(ds), status_code=201 if created else 200)

    def _dataset(s: State, dataset_id: str) -> dict[str, Any]:
        try:
            key = f"datasets/{dataset_id}/dataset.json"
            if not s.store.exists(key):
                raise HTTPException(404, "unknown dataset")
            return json.loads(s.store.get_bytes(key))  # type: ignore[no-any-return]
        except StorageKeyError as e:
            raise HTTPException(400, "invalid dataset id") from e

    @app.get("/v1/datasets/{dataset_id}")
    def get_dataset(dataset_id: str, s: State = Depends(S)) -> dict[str, Any]:
        ds = _dataset(s, dataset_id)
        return {k: v for k, v in ds.items() if k != "quality"}

    @app.get("/v1/datasets/{dataset_id}/quality")
    def dataset_quality(dataset_id: str, s: State = Depends(S)) -> dict[str, Any]:
        return {"dataset_id": dataset_id, "quality": _dataset(s, dataset_id)["quality"]}

    # ------------------------------------------------------------------ training
    @app.post("/v1/training-runs", status_code=202)
    def create_run(req: TrainingRequest, request: Request, s: State = Depends(S)) -> JSONResponse:
        from cad2ml.training.train import TrainConfig, run_id_for

        _dataset(s, req.dataset_id)
        run_id = run_id_for(req.dataset_id, TrainConfig(model=req.model, epochs=req.epochs, seed=req.seed))
        job, created = service.submit_job(
            s.db,
            s.queue,
            kind="train",
            idempotency_key=run_id,
            file_id=None,
            params={**req.model_dump(), "run_id": run_id},
            max_attempts=1,
            correlation_id=request.state.correlation_id,
        )
        return JSONResponse(
            {"run_id": run_id, **_job(job), "created": created}, status_code=202 if created else 200
        )

    @app.get("/v1/training-runs/{run_id}")
    def get_run(run_id: str, s: State = Depends(S)) -> dict[str, Any]:
        with s.db.session() as sess:
            job = sess.execute(
                select(JobRow).where(JobRow.kind == "train", JobRow.idempotency_key == run_id)
            ).scalar_one_or_none()
        key = f"training_runs/{run_id}/experiment.json"
        try:
            exp = json.loads(s.store.get_bytes(key)) if s.store.exists(key) else None
        except StorageKeyError as e:
            raise HTTPException(400, "invalid run id") from e
        if job is None and exp is None:
            raise HTTPException(404, "unknown training run")
        out: dict[str, Any] = {"run_id": run_id, "job": _job(job) if job else None}
        if exp:
            out["experiment"] = {k: v for k, v in exp.items() if k != "history"}
        return out

    @app.get("/v1/training-runs/{run_id}/predictions/{sample_id}")
    def run_predictions(run_id: str, sample_id: str, s: State = Depends(S)) -> dict[str, Any]:
        from cad2ml.training.train import infer_sample

        if not s.store.exists(f"training_runs/{run_id}/checkpoint.pt"):
            raise HTTPException(404, "unknown or unfinished training run")
        m = _manifest(s, sample_id)
        if m.status != "completed":
            raise HTTPException(409, f"sample is {m.status}")
        rows = infer_sample(s.store, run_id, sample_id)
        gt = None
        key = f"training_runs/{run_id}/experiment.json"
        ds_id = json.loads(s.store.get_bytes(key))["dataset_id"]
        labels_key = f"datasets/{ds_id}/labels.json"
        if s.store.exists(labels_key):
            lab = json.loads(s.store.get_bytes(labels_key)).get(sample_id)
            if lab:
                gt = dict(zip(lab["face_ids"], lab["labels"], strict=True))
        ds = json.loads(s.store.get_bytes(f"datasets/{ds_id}/dataset.json"))
        split = ds["samples"].get(sample_id, {}).get("split", "not_in_dataset")
        for r in rows:
            r["ground_truth"] = gt.get(r["face_id"]) if gt else None
        return {"run_id": run_id, "sample_id": sample_id, "dataset_split": split, "predictions": rows}

    @app.get("/v1/training-runs/{run_id}/predictions/{sample_id}/panel.png")
    def prediction_png(run_id: str, sample_id: str, view: int = 0, s: State = Depends(S)) -> Response:
        from cad2ml.evaluation.visualize import prediction_panel

        data = run_predictions(run_id, sample_id, s)
        return Response(
            prediction_panel(s.store, run_id, sample_id, data["predictions"], view), media_type="image/png"
        )

    @app.get("/v1/datasets")
    def list_datasets(s: State = Depends(S)) -> dict[str, Any]:
        with s.db.session() as sess:
            rows = list(sess.execute(select(DatasetRow).order_by(DatasetRow.created_at.desc())).scalars())
        return {
            "datasets": [
                {"dataset_id": r.dataset_id, "split_mode": r.split_mode, "seed": r.seed, "summary": r.summary}
                for r in rows
            ]
        }

    @app.get("/v1/training-runs")
    def list_runs(s: State = Depends(S)) -> dict[str, Any]:
        with s.db.session() as sess:
            rows = list(
                sess.execute(
                    select(JobRow).where(JobRow.kind == "train").order_by(JobRow.created_at.desc())
                ).scalars()
            )
        return {"training_runs": [{"run_id": r.idempotency_key, **_job(r)} for r in rows]}

    # ------------------------------------------------------------------ inspection UI
    if STATIC.exists():
        app.mount("/ui", StaticFiles(directory=STATIC, html=True), name="ui")

        @app.get("/")
        def root() -> FileResponse:
            return FileResponse(STATIC / "index.html")

    return app


def _job(job: JobRow) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "kind": job.kind,
        "state": job.state,
        "attempts": job.attempts,
        "max_attempts": job.max_attempts,
        "file_id": job.file_id,
        "sample_id": job.sample_id,
        "error_code": job.error_code,
        "error_message": job.error_message,
        "correlation_id": job.correlation_id,
        "worker_id": job.worker_id,
        "result": job.result,
        "created_at": job.created_at.isoformat() + "Z" if job.created_at else None,
        "finished_at": job.finished_at.isoformat() + "Z" if job.finished_at else None,
    }


def _dataset_summary(ds: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_id": ds["dataset_id"],
        "created_at": ds.get("created_at"),
        "split_sizes": ds["quality"]["split_sizes"],
        "excluded": len(ds["excluded"]),
        "leakage_passed": ds["quality"]["leakage"]["passed"],
    }


app = create_app()
