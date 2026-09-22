"""Streaming progress.jsonl writer (gp-telemetry/1.2 §5)."""
from __future__ import annotations

import fcntl
import json
import os
import time
from pathlib import Path
from typing import Any


ALLOWED_EVENTS = frozenset(
    {
        "workload_start",
        "workload_end",
        "step_end",
        "eval_start",
        "eval_end",
        "checkpoint_start",
        "checkpoint_end",
        "request_in",
        "request_out",
        "phase_change",
    }
)


class ProgressLog:
    """표준 §5 progress.jsonl 기록기. path가 None이면 아무것도 쓰지 않는다."""

    ALLOWED = ALLOWED_EVENTS

    def __init__(self, path: str | Path | None, gpu_id: int, *, fsync: bool = False):
        self.path = None if path is None else Path(path)
        self.gpu_id = int(gpu_id)
        self.fsync = bool(fsync)
        self._fh = None
        self._lock_fh = None
        self._closed = False
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # line-buffered text isn't enough for multi-rank; use binary append + flock
            self._fh = open(self.path, "ab", buffering=0)
            lock_path = self.path.with_suffix(self.path.suffix + ".lock")
            self._lock_fh = open(lock_path, "a+", encoding="utf-8")

    def emit(self, event: str, *, step: int | None = None, t: Any = None, **extra: Any) -> None:
        if self._closed:
            raise RuntimeError("ProgressLog is closed")
        if t is not None:
            raise ValueError("workload must not record 't'; finalize computes t from t_epoch")
        if event not in self.ALLOWED:
            raise ValueError(f"unknown progress event {event!r}; allowed={sorted(self.ALLOWED)}")
        payload: dict[str, Any] = {
            "t_epoch": float(time.time()),
            "gpu_id": self.gpu_id,
            "event": event,
        }
        if step is not None:
            payload["step"] = int(step)
        if extra:
            payload["extra"] = extra
        if self._fh is None:
            return
        line = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        assert self._lock_fh is not None
        fcntl.flock(self._lock_fh.fileno(), fcntl.LOCK_EX)
        try:
            os.write(self._fh.fileno(), line)
            if self.fsync:
                os.fsync(self._fh.fileno())
        finally:
            fcntl.flock(self._lock_fh.fileno(), fcntl.LOCK_UN)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._fh is not None:
            try:
                self._fh.flush()
            finally:
                self._fh.close()
                self._fh = None
        if self._lock_fh is not None:
            self._lock_fh.close()
            self._lock_fh = None

    def __enter__(self) -> "ProgressLog":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class DecoyStepper:
    """Main-loop decoy step_end emitter; does not sleep or spawn threads."""

    def __init__(
        self,
        log: ProgressLog,
        period_s: float,
        *,
        jitter_frac: float = 0.05,
        rng=None,
    ):
        if period_s <= 0:
            raise ValueError("period_s must be positive")
        if not 0.0 <= float(jitter_frac) < 1.0:
            raise ValueError("jitter_frac must be in [0, 1)")
        self.log = log
        self.period_s = float(period_s)
        self.jitter_frac = float(jitter_frac)
        self._rng = rng
        self._step = 0
        self._next_mono: float | None = None

    def _next_period(self) -> float:
        if self.jitter_frac <= 0 or self._rng is None:
            return self.period_s
        delta = self._rng.uniform(-self.jitter_frac, self.jitter_frac)
        return max(1e-3, self.period_s * (1.0 + float(delta)))

    def tick(self, *, now_mono: float | None = None) -> bool:
        """Emit at most one decoy step_end if due. Returns True if emitted."""
        now = time.monotonic() if now_mono is None else float(now_mono)
        if self._next_mono is None:
            self._next_mono = now
        if now < self._next_mono:
            return False
        self.log.emit("step_end", step=self._step)
        self._step += 1
        self._next_mono = now + self._next_period()
        return True
