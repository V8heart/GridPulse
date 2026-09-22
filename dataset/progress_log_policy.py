"""Deterministic progress-log visibility policy (gp-telemetry/1.2)."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass


POLICY_VERSION = "v1_1_uniform_all_sessions"
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


def _validate_drop_prob(drop_prob: float) -> float:
    value = float(drop_prob)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"drop_prob must be in [0, 1], got {value}")
    return value


def decide_progress_log(
    session_id: str,
    *,
    native_events: list | None = None,
    seed: int,
    drop_prob: float = DEFAULT_DROP_PROB,
    idle_exception: bool = False,
) -> ProgressLogDecision:
    """Apply class-independent drop to every non-idle session with a raw log.

    ``idle_exception=True`` is reserved for interactive ``normal_idle`` which
    naturally has no progress events. Empty ``native_events`` no longer skips
    the mask: callers must supply decoy/host events for all other classes.
    """
    drop = _validate_drop_prob(drop_prob)
    if idle_exception:
        return ProgressLogDecision(
            native_progress_available=False,
            progress_log_masked=False,
            write_progress_log=False,
            drop_prob=drop,
        )
    # native_events is retained for call-site clarity / future checks; presence
    # of a raw log is the eligibility signal under v1.2.
    _ = native_events
    masked = _unit_interval(int(seed), str(session_id)) < drop
    return ProgressLogDecision(
        native_progress_available=True,
        progress_log_masked=masked,
        write_progress_log=not masked,
        drop_prob=drop,
    )


def eval_mask_hides_log(
    session_id: str,
    *,
    seed: int,
    drop_prob: float = DEFAULT_DROP_PROB,
    native_progress_available: bool = True,
    idle_exception: bool = False,
) -> bool:
    """Eval-time mask: hide a preserved original log without deleting it."""
    if idle_exception or not native_progress_available:
        return False
    return decide_progress_log(
        session_id,
        native_events=[{"placeholder": True}],
        seed=seed,
        drop_prob=drop_prob,
        idle_exception=False,
    ).progress_log_masked
