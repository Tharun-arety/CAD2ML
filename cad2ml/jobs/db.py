"""Durable metadata (PostgreSQL in compose; SQLite for native tests).

Postgres is the source of truth for job state. The queue only transports job ids.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    event,
    select,
    update,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from cad2ml.jobs.states import JobState, check_transition


def utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)  # naive UTC everywhere (portable across SQLite/Postgres)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:20]}"


class Base(DeclarativeBase):
    pass


class FileRow(Base):
    __tablename__ = "files"
    file_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    sha256: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    filename: Mapped[str] = mapped_column(String(200))
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    storage_key: Mapped[str] = mapped_column(String(300))
    created_at: Mapped[datetime] = mapped_column(DateTime(), default=utcnow)


class JobRow(Base):
    __tablename__ = "jobs"
    __table_args__ = (
        UniqueConstraint("kind", "idempotency_key", name="uq_jobs_kind_idempotency"),
        Index("ix_jobs_state_lease", "state", "lease_expires_at"),
    )
    job_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    kind: Mapped[str] = mapped_column(String(20), default="process")  # process | train
    idempotency_key: Mapped[str] = mapped_column(String(200))
    file_id: Mapped[str | None] = mapped_column(ForeignKey("files.file_id"), nullable=True)
    sample_id: Mapped[str | None] = mapped_column(String(40), nullable=True, index=True)
    state: Mapped[str] = mapped_column(String(30), index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3)
    worker_id: Mapped[str | None] = mapped_column(String(80), nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(), nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(60), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    correlation_id: Mapped[str] = mapped_column(String(64))
    params: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(), nullable=True)


class JobEventRow(Base):
    __tablename__ = "job_events"
    event_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.job_id"), index=True)
    state: Mapped[str] = mapped_column(String(30))
    attempt: Mapped[int] = mapped_column(Integer, default=0)
    worker_id: Mapped[str | None] = mapped_column(String(80), nullable=True)
    detail: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(), default=utcnow)


class DatasetRow(Base):
    __tablename__ = "datasets"
    dataset_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    split_mode: Mapped[str] = mapped_column(String(20))
    seed: Mapped[int] = mapped_column(Integer)
    summary: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(), default=utcnow)


class WorkerRow(Base):
    __tablename__ = "workers"
    worker_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    heartbeat_at: Mapped[datetime] = mapped_column(DateTime(), default=utcnow)
    current_job_id: Mapped[str | None] = mapped_column(String(40), nullable=True)


def make_engine(url: str) -> Engine:
    kwargs: dict[str, Any] = {"future": True, "pool_pre_ping": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
    engine = create_engine(url, **kwargs)
    if url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _pragma(conn: Any, _: Any) -> None:
            cur = conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()

    return engine


class Database:
    def __init__(self, url: str, create: bool = False) -> None:
        self.engine = make_engine(url)
        self.Session = sessionmaker(self.engine, expire_on_commit=False)
        if create:
            Base.metadata.create_all(self.engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        s = self.Session()
        try:
            yield s
            s.commit()
        except BaseException:
            s.rollback()
            raise
        finally:
            s.close()

    # ---- state transitions (guarded compare-and-set) -----------------------------------
    def transition(
        self,
        job_id: str,
        src: JobState | str,
        dst: JobState | str,
        *,
        detail: dict[str, Any] | None = None,
        worker_id: str | None = None,
        **fields: Any,
    ) -> bool:
        """Atomically move ``job_id`` from ``src`` to ``dst``. Returns False if the job is not in ``src``."""
        check_transition(src, dst)
        now = utcnow()
        with self.session() as s:
            res = s.execute(
                update(JobRow)
                .where(JobRow.job_id == job_id, JobRow.state == str(src))
                .values(state=str(dst), updated_at=now, **fields)
            )
            if res.rowcount != 1:  # type: ignore[attr-defined]
                return False
            attempts = s.execute(select(JobRow.attempts).where(JobRow.job_id == job_id)).scalar_one()
            s.add(
                JobEventRow(
                    job_id=job_id,
                    state=str(dst),
                    attempt=attempts,
                    worker_id=worker_id,
                    detail=detail or {},
                    created_at=now,
                )
            )
        return True

    def add_event(
        self, job_id: str, state: str, detail: dict[str, Any], worker_id: str | None = None
    ) -> None:
        with self.session() as s:
            attempts = s.execute(select(JobRow.attempts).where(JobRow.job_id == job_id)).scalar_one()
            s.add(
                JobEventRow(job_id=job_id, state=state, attempt=attempts, worker_id=worker_id, detail=detail)
            )

    def get_job(self, job_id: str) -> JobRow | None:
        with self.session() as s:
            return s.get(JobRow, job_id)

    def events(self, job_id: str) -> list[JobEventRow]:
        with self.session() as s:
            return list(
                s.execute(
                    select(JobEventRow).where(JobEventRow.job_id == job_id).order_by(JobEventRow.event_id)
                ).scalars()
            )
