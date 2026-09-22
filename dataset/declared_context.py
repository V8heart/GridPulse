"""Leakage-safe declared context for synthetic and real telemetry sessions."""
from __future__ import annotations

from typing import Literal

import numpy as np

DECLARED_POOL = {
    "llm_pretrain": {"family": "training", "process": "python train.py"},
    "llm_finetune": {"family": "training", "process": "python finetune.py"},
    "ddp_training": {"family": "training", "process": "torchrun train.py"},
    "hpo_sweep": {"family": "training", "process": "python sweep.py"},
    "batch_inference": {"family": "inference", "process": "python batch_infer.py"},
    "online_inference": {"family": "inference", "process": "python serve.py"},
    "evaluation": {"family": "evaluation", "process": "python eval.py"},
    "notebook": {"family": "interactive", "process": "jupyter-kernel"},
}

FAMILY_TO_DECLARED = {
    "training": ("llm_pretrain", "llm_finetune", "ddp_training", "hpo_sweep"),
    "inference": ("batch_inference", "online_inference"),
    "evaluation": ("evaluation",),
    "interactive": ("notebook",),
}

DeclaredPolicy = Literal["honest", "pool_random", "host_family_matched", "host_inherited"]
SPLIT_DECLARED_ATTEMPTS = 16


def _job_choices(
    true_family: str | None,
    rng: np.random.Generator,
    *,
    disguise: bool | None,
    policy: DeclaredPolicy | None,
    mismatch_rate: float,
    allowed_families: tuple[str, ...] | None,
) -> list[str]:
    resolved = _resolve_policy(disguise=disguise, policy=policy)
    if resolved == "host_inherited":
        raise ValueError("host_inherited has no job pool")
    if resolved == "pool_random" or true_family is None:
        choices = list(DECLARED_POOL)
    elif resolved == "host_family_matched":
        choices = list(FAMILY_TO_DECLARED["training"])
    elif rng.random() < mismatch_rate:
        choices = list(DECLARED_POOL)
    else:
        choices = list(FAMILY_TO_DECLARED.get(true_family, DECLARED_POOL))
    if allowed_families is not None:
        allowed = set(allowed_families)
        unknown = allowed - set(FAMILY_TO_DECLARED)
        if unknown:
            raise ValueError(f"unknown declared families: {sorted(unknown)}")
        choices = [job for job in choices if DECLARED_POOL[job]["family"] in allowed]
        if not choices:
            raise ValueError("allowed_families removed every declared choice")
    return choices


def _declared_from_job(job_type: str, user: str) -> dict[str, str]:
    spec = DECLARED_POOL[job_type]
    return {
        "declared_job_type": job_type,
        "declared_job_family": spec["family"],
        "declared_process_name": spec["process"],
        "declared_user": user,
    }


def sample_declared(
    true_family: str | None,
    rng: np.random.Generator,
    *,
    disguise: bool | None = None,
    policy: DeclaredPolicy | None = None,
    mismatch_rate: float = 0.1,
    host_declared: dict[str, str] | None = None,
    allowed_families: tuple[str, ...] | None = None,
) -> dict[str, str]:
    """Sample declared scheduler context without exposing the true class label.

    Prefer ``policy``. Legacy ``disguise=True`` maps to ``pool_random``.
    ``host_inherited`` copies ``host_declared`` (piggyback / ltma / mimicry).
    """
    resolved = _resolve_policy(disguise=disguise, policy=policy)
    if resolved == "host_inherited":
        if not host_declared:
            raise ValueError("host_inherited requires host_declared")
        required = (
            "declared_job_type",
            "declared_job_family",
            "declared_process_name",
            "declared_user",
        )
        missing = [k for k in required if k not in host_declared]
        if missing:
            raise ValueError(f"host_declared missing keys: {missing}")
        out = {k: str(host_declared[k]) for k in required}
        if "declared_gres" in host_declared:
            out["declared_gres"] = str(host_declared["declared_gres"])
        return out

    choices = _job_choices(
        true_family,
        rng,
        disguise=disguise,
        policy=policy,
        mismatch_rate=mismatch_rate,
        allowed_families=allowed_families,
    )
    job_type = str(rng.choice(choices))
    user = f"user_{int(rng.integers(0, 50)):03d}"
    return _declared_from_job(job_type, user)


def _split_from_anchor(left: dict[str, str], right: dict[str, str]) -> bool:
    return (
        right["declared_user"] != left["declared_user"]
        and right["declared_job_type"] != left["declared_job_type"]
    )


def sample_split_partner(
    anchor: dict[str, str],
    rng: np.random.Generator,
    *,
    true_family: str | None,
    policy: DeclaredPolicy | None,
    mismatch_rate: float = 0.0,
    allowed_families: tuple[str, ...] | None = None,
    attempts: int = SPLIT_DECLARED_ATTEMPTS,
) -> dict[str, str]:
    """Sample a second GPU declaration whose user and job type differ from ``anchor``.

    Retries up to ``attempts`` times, then picks a different job and user from the pool.
    """
    for _ in range(attempts):
        candidate = sample_declared(
            true_family,
            rng,
            policy=policy,
            mismatch_rate=mismatch_rate,
            allowed_families=allowed_families,
        )
        if _split_from_anchor(anchor, candidate):
            return candidate
    choices = _job_choices(
        true_family,
        rng,
        disguise=None,
        policy=policy,
        mismatch_rate=mismatch_rate,
        allowed_families=allowed_families,
    )
    others = [job for job in choices if job != anchor["declared_job_type"]]
    if not others:
        raise ValueError("declared pool has no job type distinct from the anchor GPU")
    job_type = str(rng.choice(others))
    users = [f"user_{index:03d}" for index in range(50) if f"user_{index:03d}" != anchor["declared_user"]]
    return _declared_from_job(job_type, str(rng.choice(users)))


def _resolve_policy(
    *,
    disguise: bool | None,
    policy: DeclaredPolicy | None,
) -> DeclaredPolicy:
    if policy is not None:
        if policy not in {"honest", "pool_random", "host_family_matched", "host_inherited"}:
            raise ValueError(f"unknown declared policy: {policy!r}")
        return policy
    if disguise is True:
        return "pool_random"
    return "honest"
