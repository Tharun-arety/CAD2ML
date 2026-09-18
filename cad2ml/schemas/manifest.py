"""Canonical processed-part manifest and entity schemas (schema_version 1.0.0).

Provenance vocabulary used throughout (``Provenance``):

* ``observed``   - read directly from the STEP file (units, topology as encoded)
* ``computed``   - deterministic geometry computed by OCCT/NumPy from the B-Rep
* ``inferred``   - heuristic engineering-feature inference (may be wrong)
* ``ground_truth`` - synthetic generator parameters (only for generated parts)
* ``unavailable`` - explicitly not recoverable from STEP in v1
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Provenance(StrEnum):
    observed = "observed"
    computed = "computed"
    inferred = "inferred"
    ground_truth = "ground_truth"
    unavailable = "unavailable"


class SampleStatus(StrEnum):
    completed = "completed"
    rejected = "rejected"
    quarantined = "quarantined"
    failed = "failed"


class SourceInfo(_Strict):
    filename: str
    sha256: str
    size_bytes: int
    format: Literal["STEP"] = "STEP"
    original_units: str | None = None
    original_units_provenance: Provenance = Provenance.observed
    step_schema: str | None = None
    originating_system: str | None = None


class ProcessingInfo(_Strict):
    pipeline_version: str
    parser_version: str
    extractor_version: str
    recognizer_version: str
    schema_version: str
    configuration_hash: str
    processed_at: str
    stage_timings_s: dict[str, float] = Field(default_factory=dict)


class GeometrySummary(_Strict):
    provenance: Provenance = Provenance.computed
    units: Literal["mm"] = "mm"
    body_count: int
    solid_count: int
    shell_count: int
    face_count: int
    edge_count: int
    vertex_count: int
    volume_mm3: float
    surface_area_mm2: float
    bounding_box_min_mm: list[float]
    bounding_box_max_mm: list[float]
    bounding_box_mm: list[float]
    center_of_mass_mm: list[float] | None
    surface_type_histogram: dict[str, int]
    curve_type_histogram: dict[str, int]


class ValidationReport(_Strict):
    source_valid: bool
    closed_solid: bool
    orientation_ok: bool
    repair_attempted: bool
    repaired_valid: bool | None
    repair_operations: list[str] = Field(default_factory=list)
    max_deviation_mm: float | None = None
    volume_relative_change: float | None = None
    warnings: list[str] = Field(default_factory=list)


class ArtifactRef(_Strict):
    key: str
    sha256: str
    bytes: int
    media_type: str
    description: str


class FaceRecord(_Strict):
    face_id: str
    traversal_index: int
    surface_type: str
    orientation: Literal["forward", "reversed"]
    area_mm2: float
    centroid_mm: list[float]
    normal_at_centroid: list[float] | None
    bbox_min_mm: list[float]
    bbox_max_mm: list[float]
    uv_bounds: list[float]
    analytic_params: dict[str, float | list[float]]
    curvature: dict[str, float]
    adjacent_face_ids: list[str]
    boundary_edge_ids: list[str]
    outer_loop_edge_ids: list[str]
    loop_count: int
    fingerprint: str


class EdgeRecord(_Strict):
    edge_id: str
    traversal_index: int
    curve_type: str
    length_mm: float
    start_mm: list[float]
    end_mm: list[float]
    closed: bool
    degenerated: bool
    adjacent_face_ids: list[str]
    convexity: Literal["convex", "concave", "smooth", "boundary", "seam", "non_manifold", "unknown"]
    dihedral_angle_deg: float | None
    fingerprint: str


class Evidence(_Strict):
    check: str
    passed: bool
    detail: str


class GeometricObservation(_Strict):
    """What was measured (computed provenance). Never a semantic claim."""

    face_id: str
    surface_type: str
    material_side: Literal["inside", "outside", "n/a"]
    measurements: dict[str, float | list[float] | str | bool]


class FeatureRecord(_Strict):
    feature_id: str
    feature_type: Literal[
        "unknown",
        "planar_face",
        "cylindrical_hole_wall",
        "through_hole",
        "blind_hole",
        "slot",
        "pocket",
        "fillet",
    ]
    provenance: Literal[Provenance.inferred] = Provenance.inferred
    participating_faces: list[str]
    parameters: dict[str, float | list[float] | str]
    inference_method: str
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_basis: str
    evidence: list[Evidence]


class QualityReport(_Strict):
    passed: bool
    checks: dict[str, bool]
    details: dict[str, float | int | str]


class Rejection(_Strict):
    code: str
    stage: str
    message: str
    retryable: bool = False


class Manifest(_Strict):
    sample_id: str
    status: SampleStatus
    source: SourceInfo
    processing: ProcessingInfo
    geometry: GeometrySummary | None = None
    validation: ValidationReport | None = None
    rejection: Rejection | None = None
    artifacts: dict[str, ArtifactRef] = Field(default_factory=dict)
    observations: list[GeometricObservation] = Field(default_factory=list)
    features: list[FeatureRecord] = Field(default_factory=list)
    quality: QualityReport | None = None
    lineage: dict[str, str | list[str] | dict[str, str]] = Field(default_factory=dict)
    unavailable: list[str] = Field(default_factory=list)
