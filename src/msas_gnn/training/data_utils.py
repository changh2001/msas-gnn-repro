"""训练前的数据准备辅助。"""
from __future__ import annotations

import logging

from torch_geometric.data import Data

logger = logging.getLogger(__name__)


def prepare_supervised_data(cfg, seed, device="cpu") -> tuple[Data, int, int]:
    """加载数据、应用论文协议划分，并按需施加噪声扰动。"""
    from msas_gnn.data.dataset_factory import load_dataset
    from msas_gnn.data.split_manager import load_or_create_split
    from msas_gnn.data.transforms import add_self_loops_transform, apply_edge_noise

    dataset = cfg.get("dataset", "cora")
    data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data", {}), dict) else {}
    data_root = data_cfg.get("raw", "data/raw")
    split_dir = data_cfg.get("splits", "data/splits")

    data, num_classes, num_features = load_dataset(dataset, root=data_root)
    noise_cfg = cfg.get("noise", {}) if isinstance(cfg.get("noise", {}), dict) else {}
    if noise_cfg.get("enabled", False):
        mode = noise_cfg.get("mode", "flip")
        ratio = float(noise_cfg.get("ratio", 0.0))
        seed_offset = int(noise_cfg.get("seed_offset", 10000))
        data = apply_edge_noise(data, mode=mode, ratio=ratio, seed=seed + seed_offset)
        logger.info(
            "[%s][seed=%s] 应用图噪声 mode=%s ratio=%.2f",
            dataset,
            seed,
            mode,
            ratio,
        )
    data = add_self_loops_transform(data)

    split_ratio = cfg.get("split_ratio", [0.6, 0.2, 0.2])
    use_time_split = bool(cfg.get("use_time_split", False))
    use_original_split = bool(cfg.get("use_original_split", False))
    if not use_time_split and not use_original_split:
        train_mask, val_mask, test_mask = load_or_create_split(
            dataset,
            data.y,
            seed,
            split_dir=split_dir,
            train_ratio=split_ratio[0],
            val_ratio=split_ratio[1],
        )
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

    return data.to(device), num_classes, num_features
