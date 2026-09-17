"""Explicit gt_label -> expected corpus document mapping (no special cases)."""
from __future__ import annotations

# Longest-prefix / exact map. Normals map to normal_workloads only
# (benign_periodic is NOT an automatic accept for all normals).
LABEL_TO_DOC: dict[str, str] = {
    "swma": "swma",
    "swma_shallow": "swma",
    "swma_jitter": "swma",
    "swma_piggyback": "swma",
    "swma_mimicry": "swma",
    "swma_coordinated": "swma",
    "swma_holdout": "swma",
    "ltma": "ltma",
    "cryptojacking": "cryptojacking",
    "coordinated_multi_gpu": "coordinated_multi_gpu",
    "burst_ramp_attack": "burst_ramp_attack",
    "hidden_ml_training": "hidden_ml_training",
    "llmjacking": "llmjacking",
    "sponge_attack": "sponge_attack",
    "normal_workloads": "normal_workloads",
    "normal_baseline": "normal_workloads",
    "normal_checkpoint": "normal_workloads",
    "normal_dataloader_stall": "normal_workloads",
    "normal_distributed_training": "normal_workloads",
    "normal_eval_train_switch": "normal_workloads",
    "normal_hpo_search": "normal_workloads",
    "normal_sync_ddp": "normal_workloads",
    "normal_fsdp_deep_trough": "normal_workloads",
    "normal_flat_pretrain": "normal_workloads",
    "normal_inference_bursty": "normal_workloads",
    "normal_mixed_tenants": "normal_workloads",
    "benign_periodic": "benign_periodic",
}


def expected_document(label: str | None) -> str | None:
    if label is None:
        return None
    text = str(label)
    if text in LABEL_TO_DOC:
        return LABEL_TO_DOC[text]
    if text.startswith("swma"):
        return "swma"
    if text.startswith("normal"):
        return "normal_workloads"
    return text
