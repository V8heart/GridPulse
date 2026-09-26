"""Leakage-safe declared context for synthetic and real telemetry sessions."""
from __future__ import annotations

from typing import Literal

import numpy as np

DECLARED_POOL = {
    "llm_pretrain": {"family": "training", "process": "python train.py"},
    "llm_finetune": {"family": "training", "process": "python finetune.py"},
    "ddp_training": {"family": "training", "process": "torchrun train.py"},
    "hpo_sweep": {"family": "training", "process": "python sweep.py"},
    "vision_training": {"family": "training", "process": "python train_vision.py"},
    "dataloader_bound": {"family": "training", "process": "python dataloader.py"},
    "batch_inference": {"family": "inference", "process": "python batch_infer.py"},
    "online_inference": {"family": "inference", "process": "python serve.py"},
    "evaluation": {"family": "evaluation", "process": "python eval.py"},
    "notebook": {"family": "interactive", "process": "jupyter-kernel"},
}

FAMILY_TO_DECLARED = {
    "training": (
        "llm_pretrain",
        "llm_finetune",
        "ddp_training",
        "hpo_sweep",
        "vision_training",
        "dataloader_bound",
    ),
    "inference": ("batch_inference", "online_inference"),
    "evaluation": ("evaluation",),
    "interactive": ("notebook",),
}

# Capture --workload / gt_variant / gt_params.workload → honest job_type.
HONEST_WORKLOAD_TO_JOB_TYPE = {
    "llm_finetune": "llm_finetune",
    "gpt_tiny_finetune": "llm_finetune",
    "finetune": "llm_finetune",
    "llm_pretrain_ddp": "ddp_training",
    "pretrain_ddp": "ddp_training",
    "llm_pretrain_fsdp": "ddp_training",
    "pretrain_fsdp": "ddp_training",
    "resnet_ddp": "ddp_training",
    "ddp": "ddp_training",
    "distributed": "ddp_training",
    "llm_flat_pretrain": "llm_pretrain",
    "flat_pretrain": "llm_pretrain",
    "hpo": "hpo_sweep",
    "dataloader_stall": "dataloader_bound",
    "resnet_single": "vision_training",
    "single": "vision_training",
    "llm_inference_serving": "online_inference",
    "serving": "online_inference",
    "llm_inference_batch": "batch_inference",
    "batch": "batch_inference",
    "baseline": "notebook",
    "checkpoint": "notebook",
    "eval_train_switch": "notebook",
}

ATTACK_DECLARED_POLICIES = frozenset({"pool_random", "host_family_matched", "host_inherited"})

# Optional mid-group keys (used only after inventory decision).
MID_GROUP_FOR_JOB_TYPE = {
    "llm_pretrain": "llm_train",
    "llm_finetune": "llm_train",
    "hpo_sweep": "llm_train",
    "ddp_training": "distributed_train",
    "vision_training": "vision_train",
    "dataloader_bound": "vision_train",
    "online_inference": "inference_online",
    "batch_inference": "inference_batch",
    "notebook": "interactive",
    "evaluation": "interactive",
}


def mid_group_for_job_type(job_type: object) -> str:
    key = str(job_type or "").strip()
    return MID_GROUP_FOR_JOB_TYPE.get(key, key or "unknown")


def apply_mid_group_column(frame):
    """Add declared_mid_group. Used only after the inventory mid-group choice."""
    out = frame.copy()
    out["declared_mid_group"] = out["declared_job_type"].map(mid_group_for_job_type)
    return out


# Declared expected power band. Never derived from observed mean_w.
# evaluation: low is unused on the real capture; revisit before applying to synth.
DECLARED_EXPECTED_BAND = {
    "llm_pretrain": "high",
    "llm_finetune": "high",
    "ddp_training": "high",
    "vision_training": "mid",
    "hpo_sweep": "low",
    "dataloader_bound": "low",
    "batch_inference": "low",
    "online_inference": "low",
    "notebook": "low",
    "evaluation": "low",
}


def expected_band_for_job_type(job_type: object) -> str:
    key = str(job_type or "").strip()
    return DECLARED_EXPECTED_BAND.get(key, "low")


def family_for_job_type(job_type: object) -> str:
    key = str(job_type or "").strip()
    spec = DECLARED_POOL.get(key)
    return str(spec["family"]) if spec else "unknown"


def apply_expected_band_column(frame):
    """Add expected_band from declared_job_type only."""
    out = frame.copy()
    out["expected_band"] = out["declared_job_type"].map(expected_band_for_job_type)
    return out


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


def resolve_honest_workload_key(*candidates: object) -> str | None:
    """Return the first candidate that maps to an honest job type."""
    for raw in candidates:
        if raw is None:
            continue
        key = str(raw).strip()
        if not key or key.lower() in {"nan", "none"}:
            continue
        if key in HONEST_WORKLOAD_TO_JOB_TYPE:
            return key
    return None


def honest_job_type_for_workload(*candidates: object) -> str:
    key = resolve_honest_workload_key(*candidates)
    if key is None:
        raise ValueError(f"no honest job_type mapping for {candidates!r}")
    return HONEST_WORKLOAD_TO_JOB_TYPE[key]


def honest_declared_for_workload(*candidates: object, user: str) -> dict[str, str]:
    return _declared_from_job(honest_job_type_for_workload(*candidates), user)


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
