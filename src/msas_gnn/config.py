"""配置加载与深度合并工具。"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

BASE_MODEL_BY_ABLATION = {
    "b0": "sdgnn",
    "sdgnn_pure": "sdgnn_pure",
    "b1": "msas_gnn_b5",
    "b2": "msas_gnn_b5",
    "b3": "msas_gnn_b5",
    "b4": "msas_gnn_b5",
    "b5": "msas_gnn_b5",
    "b5_frozen": "msas_gnn_b5",
    "b2_rnd": "msas_gnn_b5",
    "hop_uniform": "msas_gnn_b5",
    "hop_near_engineering": "msas_gnn_b5",
    "hop_spectral_gap_reference": "msas_gnn_b5",
    "hop_reverse": "msas_gnn_b5",
}

ABLATION_CONFIGS = {
    "b0": "configs/ablations/b0_sdgnn.yaml",
    "sdgnn_pure": "configs/ablations/sdgnn_pure.yaml",
    "b1": "configs/ablations/b1_plus_spectral.yaml",
    "b2": "configs/ablations/b2_plus_centrality.yaml",
    "b3": "configs/ablations/b3_plus_kcore.yaml",
    "b4": "configs/ablations/b4_plus_entropy.yaml",
    "b5": "configs/ablations/b5_full_main.yaml",
    "b5_frozen": "configs/ablations/b5_frozen.yaml",
    "b2_rnd": "configs/ablations/b2_rnd_control.yaml",
    "hop_uniform": "configs/ablations/hop_uniform.yaml",
    "hop_near_engineering": "configs/ablations/hop_near_engineering.yaml",
    "hop_spectral_gap_reference": "configs/ablations/hop_spectral_gap_reference.yaml",
    "hop_reverse": "configs/ablations/hop_reverse.yaml",
}

GLOBAL_CONFIGS = (
    "configs/global/default.yaml",
    "configs/global/paths.yaml",
    "configs/global/runtime.yaml",
    "configs/global/seeds.yaml",
)


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """递归合并dict；标量/列表采用后者覆盖。"""
    merged = deepcopy(base)
    for key, value in (update or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_yaml(path: str | Path) -> dict[str, Any]:
    if not path:
        return {}
    full_path = _repo_path(path)
    if not full_path.exists() or full_path.is_dir():
        return {}
    with full_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def normalize_teacher_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """将 teacher: gcn 规范化为完整teacher配置。"""
    teacher_cfg = cfg.get("teacher", {})
    if isinstance(teacher_cfg, str):
        teacher_name = teacher_cfg
        cfg["teacher"] = load_yaml(f"configs/teachers/{teacher_name}.yaml")
    elif not isinstance(teacher_cfg, dict):
        cfg["teacher"] = {}
    elif "model" in teacher_cfg and "name" not in teacher_cfg:
        cfg["teacher"] = deepcopy(teacher_cfg)
        cfg["teacher"]["name"] = teacher_cfg["model"]
    teacher_name = cfg.get("teacher", {}).get("name", "gcn")
    cfg["teacher_name"] = teacher_name
    return cfg


def normalize_config_aliases(cfg: dict[str, Any]) -> dict[str, Any]:
    """将顶层工程别名归位到源码实际消费的嵌套字段。"""
    cfg = deepcopy(cfg)
    train = cfg.setdefault("train", {})
    lars = cfg.setdefault("lars", {})
    hop_dim = cfg.setdefault("hop_dim", {})
    node_dim = cfg.setdefault("node_dim", {})

    alias_pairs = [
        ("lr", train, "lr"),
        ("dropout", train, "dropout"),
        ("weight_decay", train, "weight_decay"),
        ("epochs", train, "epochs"),
        ("early_stopping_patience", train, "early_stopping_patience"),
        ("batch_size", train, "batch_size"),
        ("k", lars, "k"),
        ("L", hop_dim, "L"),
        ("tau_base", node_dim, "tau_base"),
    ]
    for source_key, target_dict, target_key in alias_pairs:
        if source_key in cfg:
            target_dict[target_key] = cfg[source_key]
    return cfg


def load_experiment_config(
    dataset: str,
    ablation_id: str = "b5",
    override_path: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """按论文协议装配一份完整实验配置。"""
    cfg: dict[str, Any] = {}
    for path in GLOBAL_CONFIGS:
        cfg = deep_merge(cfg, load_yaml(path))

    base_model = BASE_MODEL_BY_ABLATION.get(ablation_id, "msas_gnn_b5")
    cfg = deep_merge(cfg, load_yaml(f"configs/models/{base_model}.yaml"))
    cfg = deep_merge(cfg, load_yaml(f"configs/datasets/{dataset}.yaml"))
    cfg = deep_merge(cfg, load_yaml(ABLATION_CONFIGS.get(ablation_id, "")))

    if override_path:
        cfg = deep_merge(cfg, load_yaml(override_path))
    if overrides:
        cfg = deep_merge(cfg, overrides)

    cfg["dataset"] = dataset
    cfg["ablation_id"] = ablation_id
    cfg = normalize_teacher_config(cfg)
    return normalize_config_aliases(cfg)
