"""Leakage-safe declared context for synthetic telemetry sessions."""
from __future__ import annotations

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


def sample_declared(
    true_family: str | None,
    rng: np.random.Generator,
    *,
    disguise: bool,
    mismatch_rate: float = 0.1,
) -> dict[str, str]:
    """Sample declared scheduler context without exposing the true class label."""
    if disguise or true_family is None or rng.random() < mismatch_rate:
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
