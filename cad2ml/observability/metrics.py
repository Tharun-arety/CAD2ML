"""Prometheus metrics (names as specified in the operational requirements)."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

_BUCKETS = (0.25, 0.5, 1, 2, 3, 5, 8, 13, 20, 30, 60, 120, 300)

FILES_RECEIVED = Counter("files_received_total", "Uploaded files accepted by the API")
JOBS_COMPLETED = Counter("jobs_completed_total", "Jobs that reached completed")
JOBS_FAILED = Counter("jobs_failed_total", "Jobs that ended failed/rejected/timed_out", ["state"])
JOBS_QUARANTINED = Counter("jobs_quarantined_total", "Jobs quarantined")
PROCESSING_SECONDS = Histogram("processing_duration_seconds", "End-to-end job processing", buckets=_BUCKETS)
STEP_PARSE_SECONDS = Histogram(
    "step_parse_duration_seconds", "Isolated STEP parse duration", buckets=_BUCKETS
)
REPRESENTATION_SECONDS = Histogram(
    "representation_generation_seconds", "Isolated extraction duration", buckets=_BUCKETS
)
ACTIVE_WORKERS = Gauge("active_workers", "Workers with a live heartbeat")
QUEUE_DEPTH = Gauge("queue_depth", "Jobs waiting in the queue")
ARTIFACT_BYTES = Counter("artifact_bytes_written", "Bytes of derived artifacts written")
SAMPLES_BY_REJECTION = Counter(
    "samples_by_rejection_reason", "Non-completed samples by error code", ["reason"]
)
