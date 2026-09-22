"""Opaque session / run / group identifiers for gp-telemetry/1.2."""
from __future__ import annotations

import hashlib
import re
import secrets
from datetime import datetime, timezone

SESSION_ID_RE = re.compile(r"^s-[0-9a-f]{16}$")
RUN_ID_RE = re.compile(r"^r-\d{8}-[0-9a-f]{6}$")
GROUP_ID_RE = re.compile(r"^g-[0-9a-f]{16}$")


def new_session_id() -> str:
    """Cryptographically random opaque session id."""
    return f"s-{secrets.token_hex(8)}"


def synthetic_session_id(*, seed: int, key: str) -> str:
    """Deterministic opaque session id derived from seed and an internal key."""
    digest = hashlib.sha256(f"{int(seed)}:{key}".encode("utf-8")).hexdigest()
    return f"s-{digest[:16]}"


def new_run_id(*, when: datetime | None = None) -> str:
    """Batch id with date prefix; not a split group."""
    stamp = (when or datetime.now(timezone.utc)).strftime("%Y%m%d")
    return f"r-{stamp}-{secrets.token_hex(3)}"


def group_id_from_parts(*parts: object) -> str:
    """Opaque group id from matrix entry / parameter cluster (repeat-independent)."""
    payload = "|".join("" if p is None else str(p) for p in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"g-{digest[:16]}"


def is_session_id(value: str) -> bool:
    return bool(SESSION_ID_RE.fullmatch(str(value)))


def is_run_id(value: str) -> bool:
    return bool(RUN_ID_RE.fullmatch(str(value)))


def is_group_id(value: str) -> bool:
    return bool(GROUP_ID_RE.fullmatch(str(value)))


def require_session_id(value: str) -> str:
    text = str(value)
    if not is_session_id(text):
        raise ValueError(f"session_id must match ^s-[0-9a-f]{{16}}$, got {text!r}")
    return text
