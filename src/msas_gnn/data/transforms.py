"""图变换。"""
import logging
from copy import copy

import torch
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_geometric.transforms import NormalizeFeatures

logger = logging.getLogger(__name__)


def add_self_loops_transform(data):
    """添加自环（对应论文§3.1.1 Ã=A+I）。"""
    ei, _ = remove_self_loops(data.edge_index)
    ei, _ = add_self_loops(ei, num_nodes=data.num_nodes)
    data.edge_index = ei
    return data


def apply_edge_noise(data, mode="flip", ratio=0.0, seed=42):
    """对边集施加简单随机扰动，用于噪声鲁棒性对照。"""
    if float(ratio) <= 0.0:
        return data
    if mode not in {"add", "delete", "flip"}:
        raise ValueError(f"Unknown noise mode: {mode}")

    edge_index, _ = remove_self_loops(data.edge_index)
    if edge_index.numel() == 0:
        return data

    num_nodes = int(data.num_nodes)
    edge_ids = (
        edge_index[0].cpu().to(torch.int64) * num_nodes
        + edge_index[1].cpu().to(torch.int64)
    )
    existing_ids = set(int(edge_id) for edge_id in edge_ids.tolist())
    n_changes = max(int(round(edge_ids.numel() * float(ratio))), 1)
    generator = torch.Generator()
    generator.manual_seed(int(seed))

    if mode in {"delete", "flip"} and existing_ids:
        perm = torch.randperm(edge_ids.numel(), generator=generator)[: min(n_changes, edge_ids.numel())]
        for edge_id in edge_ids[perm].tolist():
            existing_ids.discard(int(edge_id))

    if mode in {"add", "flip"}:
        added = 0
        attempts = 0
        max_attempts = max(n_changes * 25, 1000)
        while added < n_changes and attempts < max_attempts:
            src = int(torch.randint(num_nodes, (1,), generator=generator).item())
            dst = int(torch.randint(num_nodes - 1, (1,), generator=generator).item())
            if dst >= src:
                dst += 1
            edge_id = src * num_nodes + dst
            if edge_id not in existing_ids:
                existing_ids.add(edge_id)
                added += 1
            attempts += 1
        if added < n_changes:
            logger.warning(
                "边噪声添加未达到目标：mode=%s requested=%s added=%s",
                mode,
                n_changes,
                added,
            )

    out = copy(data)
    if not existing_ids:
        out.edge_index = edge_index.new_empty((2, 0))
        return out

    ids = torch.tensor(sorted(existing_ids), dtype=torch.long, device=edge_index.device)
    src = (ids // num_nodes).to(edge_index.dtype)
    dst = (ids % num_nodes).to(edge_index.dtype)
    out.edge_index = torch.stack([src, dst], dim=0)
    return out


def normalize_features(data):
    return NormalizeFeatures()(data)
