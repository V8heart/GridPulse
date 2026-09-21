"""Deterministic progress-log availability policy for synth and real eval."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass


POLICY_VERSION = "v1_eligible_uniform_drop"
DEFAULT_DROP_PROB = 0.4


@dataclass(frozen=True)
class ProgressLogDecision:
    native_progress_available: bool
    progress_log_masked: bool
    write_progress_log: bool
    policy_version: str = POLICY_VERSION
    drop_prob: float = DEFAULT_DROP_PROB


def _unit_interval(seed: int, session_id: str) -> float:
    """Stable [0, 1) draw from (seed, session_id); platform-independent."""
    digest = hashlib.sha256(f"{seed}:{session_id}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0x100000000


def decide_progress_log(
    session_id: str,
    *,
    native_events: list | None,
    seed: int,
    drop_prob: float = DEFAULT_DROP_PROB,
) -> ProgressLogDecision:
    """Apply class-independent drop only when native events exist.

    Native-empty workloads (SWMA base, crypto, coordinated) stay unavailable.
    Eligible sessions (including LTMA / piggyback / mimicry) are masked with
    the same probability; this decision does not mutate original event lists.
    """
    native = bool(native_events)
    if not native:
        return ProgressLogDecision(
            native_progress_available=False,
            progress_log_masked=False,
            write_progress_log=False,
            drop_prob=float(drop_prob),
        )
    masked = _unit_interval(int(seed), str(session_id)) < float(drop_prob)
    return ProgressLogDecision(
        native_progress_available=True,
        progress_log_masked=masked,
        write_progress_log=not masked,
        drop_prob=float(drop_prob),
    )


def eval_mask_hides_log(
    session_id: str,
    *,
    seed: int,
    drop_prob: float = DEFAULT_DROP_PROB,
    native_progress_available: bool = True,
) -> bool:
    """Eval-time mask: hide a preserved original log without deleting it."""
    if not native_progress_available:
        return False
    return decide_progress_log(
        session_id,
        native_events=[{"placeholder": True}],
        seed=seed,
        drop_prob=drop_prob,
    ).progress_log_masked
