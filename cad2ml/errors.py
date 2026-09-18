"""Structured pipeline errors with stable codes and retry classification."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Outcome(StrEnum):
    rejected = "rejected"  # input is outside v1 contract; never retried
    quarantined = "quarantined"  # input looked plausible but is unsafe/invalid to process
    failed_terminal = "failed_terminal"
    failed_retryable = "failed_retryable"
    timed_out = "timed_out"


@dataclass(frozen=True)
class ErrorSpec:
    outcome: Outcome
    retryable: bool
    description: str


ERROR_CODES: dict[str, ErrorSpec] = {
    "EMPTY_FILE": ErrorSpec(Outcome.rejected, False, "File has zero bytes"),
    "FILE_TOO_LARGE": ErrorSpec(Outcome.rejected, False, "File exceeds configured size limit"),
    "UNSUPPORTED_EXTENSION": ErrorSpec(Outcome.rejected, False, "Extension not in allowlist"),
    "NOT_STEP_CONTENT": ErrorSpec(Outcome.rejected, False, "Content does not start with ISO-10303-21 header"),
    "STEP_PARSE_FAILED": ErrorSpec(Outcome.rejected, False, "OCCT STEP reader could not read the file"),
    "STEP_NO_SHAPES": ErrorSpec(Outcome.rejected, False, "STEP file contains no transferable shapes"),
    "UNSUPPORTED_ASSEMBLY": ErrorSpec(Outcome.rejected, False, "Assemblies are not supported in v1"),
    "MULTI_BODY": ErrorSpec(Outcome.rejected, False, "More than one solid body; v1 requires a single solid"),
    "NO_SOLID": ErrorSpec(Outcome.quarantined, False, "No closed solid (e.g. open shell or surfaces only)"),
    "INVALID_GEOMETRY": ErrorSpec(
        Outcome.quarantined, False, "Solid invalid and not repairable within tolerance"
    ),
    "REPAIR_DEVIATION_EXCEEDED": ErrorSpec(
        Outcome.quarantined, False, "Repair changed geometry beyond tolerance"
    ),
    "DEGENERATE_GEOMETRY": ErrorSpec(Outcome.quarantined, False, "Zero/negative volume or area"),
    "TESSELLATION_FAILED": ErrorSpec(Outcome.quarantined, False, "A face could not be triangulated"),
    "OUTPUT_INVARIANT_VIOLATION": ErrorSpec(
        Outcome.failed_terminal, False, "Generated artifacts failed checks"
    ),
    "PARSER_TIMEOUT": ErrorSpec(Outcome.timed_out, False, "STEP parsing exceeded hard timeout"),
    "PROCESSING_TIMEOUT": ErrorSpec(Outcome.timed_out, False, "Extraction exceeded hard timeout"),
    "CHILD_CRASHED": ErrorSpec(Outcome.quarantined, False, "Isolated geometry process terminated abnormally"),
    "STORAGE_IO_ERROR": ErrorSpec(Outcome.failed_retryable, True, "Transient storage failure"),
    "WORKER_LOST": ErrorSpec(Outcome.failed_retryable, True, "Worker lease expired during processing"),
    "INTERNAL_ERROR": ErrorSpec(Outcome.failed_terminal, False, "Unexpected internal error"),
}


class PipelineError(Exception):
    def __init__(self, code: str, message: str, stage: str) -> None:
        if code not in ERROR_CODES:
            raise ValueError(f"unknown error code {code}")
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.stage = stage

    @property
    def spec(self) -> ErrorSpec:
        return ERROR_CODES[self.code]
