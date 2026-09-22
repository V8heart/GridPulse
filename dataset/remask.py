"""Re-apply progress-log visibility mask without re-capturing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

from dataset.finalize_session import _rewrite_progress
from dataset.progress_log_policy import DEFAULT_DROP_PROB, decide_progress_log


def remask_session(
    session_id: str,
    *,
    root: Path = ROOT,
    drop_prob: float = DEFAULT_DROP_PROB,
    seed: int = 7,
) -> dict:
    session_dir = root / "dataset" / "real" / "sessions" / session_id
    session_json_path = session_dir / "session.json"
    raw = session_dir / "progress.raw.jsonl"
    visible = session_dir / "progress.jsonl"
    if not session_json_path.exists():
        raise FileNotFoundError(session_json_path)
    meta = json.loads(session_json_path.read_text(encoding="utf-8"))
    t0_epoch = float(meta["t0_epoch"])
    idle = not raw.exists()
    decision = decide_progress_log(
        session_id,
        native_events=[{"ok": True}] if raw.exists() else [],
        seed=seed,
        drop_prob=drop_prob,
        idle_exception=idle,
    )
    raw_hash_before = raw.read_bytes() if raw.exists() else b""
    _rewrite_progress(raw, visible, t0_epoch=t0_epoch, masked=decision.progress_log_masked)
    raw_hash_after = raw.read_bytes() if raw.exists() else b""
    if raw_hash_before != raw_hash_after:
        raise RuntimeError("remask must not mutate progress.raw.jsonl")
    history = meta.setdefault("progress_log_history", [])
    history.append(
        {
            "drop_prob": float(drop_prob),
            "seed": int(seed),
            "masked": bool(decision.progress_log_masked),
            "policy_version": decision.policy_version,
        }
    )
    meta["progress_log"] = {
        "present": bool(visible.exists()),
        "masked": bool(decision.progress_log_masked),
        "policy_version": decision.policy_version,
        "drop_prob": float(drop_prob),
        "seed": int(seed),
    }
    session_json_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"session_id": session_id, "masked": decision.progress_log_masked, "present": visible.exists()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--drop-prob", type=float, default=DEFAULT_DROP_PROB)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(remask_session(args.session_id, root=args.root, drop_prob=args.drop_prob, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
