"""Deterministic configuration.

Two kinds of configuration are kept deliberately separate:

* ``PipelineConfig`` - everything that influences derived artifacts. It is hashed
  (``config_hash``) and the hash participates in sample identity, so changing any
  value produces a different, non-colliding output.
* ``Settings`` - runtime/deployment values (URLs, paths, secrets, concurrency) that
  must never change artifact content and are therefore not hashed.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PIPELINE_VERSION = "1.1.0"  # 1.1.0: single-solid isolation from compounds, tessellated-STEP rejection
SCHEMA_VERSION = "1.0.0"
EXTRACTOR_VERSION = "1.0.0"
RECOGNIZER_VERSION = "deterministic_rule_v1"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class IngestionConfig(_Frozen):
    max_file_bytes: int = 50 * 1024 * 1024
    allowed_extensions: tuple[str, ...] = (".step", ".stp")
    parse_timeout_s: float = 120.0


class RepairConfig(_Frozen):
    enabled: bool = True
    fix_tolerance_mm: float = 1e-3
    max_deviation_mm: float = 0.05
    max_relative_volume_change: float = 1e-3


class TessellationConfig(_Frozen):
    linear_deflection_mm: float = 0.05
    relative_linear_deflection: float = 0.002  # fraction of bbox diagonal; effective = min(abs, rel*diag)
    angular_deflection_rad: float = 0.35
    min_triangles_per_face: int = 1


class PointCloudConfig(_Frozen):
    num_points: int = 4096
    min_points_per_face: int = 8
    edge_bias_fraction: float = 0.0
    compute_curvature: bool = True
    seed: int = 1234


class RenderConfig(_Frozen):
    num_views: int = 6
    resolution: int = 256
    fov_deg: float = 35.0
    elevation_deg: float = 30.0


class RecognizerConfig(_Frozen):
    angle_tol_deg: float = 2.0
    distance_tol_mm: float = 1e-3
    min_through_hole_confidence: float = 0.5


class PipelineConfig(_Frozen):
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    repair: RepairConfig = Field(default_factory=RepairConfig)
    tessellation: TessellationConfig = Field(default_factory=TessellationConfig)
    pointcloud: PointCloudConfig = Field(default_factory=PointCloudConfig)
    render: RenderConfig = Field(default_factory=RenderConfig)
    recognizer: RecognizerConfig = Field(default_factory=RecognizerConfig)

    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))

    def config_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()[:16]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CAD2ML_", env_file=".env", extra="ignore")

    data_dir: Path = Path("data")
    corpus_dir: Path = Path("fixtures/corpus")  # ground-truth sidecars for synthetic datasets
    database_url: str = "sqlite:///data/cad2ml.db"
    queue_url: str = "redis://localhost:6379/0"
    queue_backend: str = "valkey"  # "valkey" | "fake" (tests only)
    log_level: str = "INFO"
    log_json: bool = True
    worker_lease_s: int = 60
    worker_heartbeat_s: int = 10
    job_timeout_s: int = 600
    max_attempts: int = 3
    enable_fault_injection: bool = False  # test-only hooks; never enable in production
    metrics_port: int = 9100
    # RLIMIT_AS for isolated children (POSIX only). This caps *virtual address space*, which OpenBLAS/OCCT
    # reserve far beyond their resident use; 4096 crashed nearly every child on Linux (DECISIONS D-015).
    # It is a runaway guard only; real memory bounds come from the container limit (compose mem_limit).
    child_memory_limit_mb: int = 16384


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def default_pipeline_config() -> PipelineConfig:
    return PipelineConfig()
