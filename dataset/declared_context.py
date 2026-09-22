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


def sample_declared(
    true_family: str | None,
    rng: np.random.Generator,
    *,
    disguise: bool | None = None,
    policy: DeclaredPolicy | None = None,
    mismatch_rate: float = 0.1,
    host_declared: dict[str, str] | None = None,
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

    if resolved == "pool_random" or true_family is None:
        choices = list(DECLARED_POOL)
    elif resolved == "host_family_matched":
        choices = list(FAMILY_TO_DECLARED["training"])
    else:  # honest
        if rng.random() < mismatch_rate:
            choices = list(DECLARED_POOL)
        else:
            choices = list(FAMILY_TO_DECLARED.get(true_family, DECLARED_POOL))

    job_type = str(rng.choice(choices))
    spec = DECLARED_POOL[job_type]
    user = f"user_{int(rng.integers(0, 50)):03d}"
    return {
        "declared_job_type": job_type,
        "declared_job_family": spec["family"],
        "declared_process_name": spec["process"],
        "declared_user": user,
    }


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
