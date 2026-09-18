"""Explicit job state machine.

Happy path:
    received -> validated -> queued -> parsing -> normalizing -> extracting -> validating_outputs -> completed
Training jobs: queued -> training -> completed.
Failure paths:
    rejected          input outside v1 contract (never retried)
    quarantined       unsafe/invalid geometry or crashing input (never retried automatically)
    timed_out         hard timeout in an isolated stage
    failed_retryable  transient failure; re-queued while attempts < max_attempts
    failed_terminal   retries exhausted or internal error
A lost worker (expired lease) moves an in-flight job to failed_retryable and back to queued.
"""

from __future__ import annotations

from enum import StrEnum


class JobState(StrEnum):
    received = "received"
    validated = "validated"
    queued = "queued"
    parsing = "parsing"
    normalizing = "normalizing"
    extracting = "extracting"
    validating_outputs = "validating_outputs"
    training = "training"
    completed = "completed"
    rejected = "rejected"
    failed_retryable = "failed_retryable"
    failed_terminal = "failed_terminal"
    timed_out = "timed_out"
    quarantined = "quarantined"


S = JobState
IN_FLIGHT = frozenset({S.parsing, S.normalizing, S.extracting, S.validating_outputs, S.training})
TERMINAL = frozenset({S.completed, S.rejected, S.failed_terminal, S.timed_out, S.quarantined})
_FAIL = {S.rejected, S.failed_retryable, S.failed_terminal, S.timed_out, S.quarantined}

TRANSITIONS: dict[JobState, frozenset[JobState]] = {
    S.received: frozenset({S.validated, S.rejected, S.failed_terminal}),
    S.validated: frozenset({S.queued, S.failed_terminal}),
    S.queued: frozenset({S.parsing, S.training, S.completed}) | _FAIL,
    S.training: frozenset({S.completed}) | _FAIL,
    S.parsing: frozenset({S.normalizing, S.completed}) | _FAIL,
    S.normalizing: frozenset({S.extracting}) | _FAIL,
    S.extracting: frozenset({S.validating_outputs}) | _FAIL,
    S.validating_outputs: frozenset({S.completed}) | _FAIL,
    S.failed_retryable: frozenset({S.queued, S.failed_terminal}),
    S.completed: frozenset(),
    S.rejected: frozenset(),
    S.failed_terminal: frozenset(),
    S.timed_out: frozenset(),
    S.quarantined: frozenset(),
}


class IllegalTransition(ValueError):
    pass


def check_transition(src: JobState | str, dst: JobState | str) -> None:
    s, d = JobState(src), JobState(dst)
    if d not in TRANSITIONS[s]:
        raise IllegalTransition(f"{s} -> {d} is not allowed")


def is_terminal(state: JobState | str) -> bool:
    return JobState(state) in TERMINAL
