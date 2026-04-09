"""预处理摊销分析：Q_be = t_pre/(t_dense-t_sparse)。对应论文§6.4.2。"""
from __future__ import annotations

import json
import logging
import math
import os

logger = logging.getLogger(__name__)


def compute_break_even(t_pre, t_dense, t_sparse, dataset=""):
    sp = t_dense / (t_sparse + 1e-10)
    d = t_dense - t_sparse
    if d <= 0:
        logger.warning("%s: t_dense<=t_sparse，Q_be无意义", dataset)
        return {"dataset": dataset, "Q_be": float("inf"), "speedup": sp}
    q_be = t_pre / d
    logger.info("[break_even] %s: Q_be=%.0f speedup=%.1fx", dataset, q_be, sp)
    return {
        "dataset": dataset,
        "Q_be": q_be,
        "t_pre": t_pre,
        "t_dense": t_dense,
        "t_sparse": t_sparse,
        "speedup": sp,
    }


def load_json(path):
    with open(path, encoding="utf-8") as fp:
        return json.load(fp)


def find_latest_efficiency_result(results_dir):
    candidates = []
    for root, _, files in os.walk(results_dir):
        for name in files:
            if name.startswith("efficiency_") and name.endswith(".json"):
                candidates.append(os.path.join(root, name))
    if not candidates:
        raise FileNotFoundError(f"未找到 efficiency_*.json: {results_dir}")
    return max(candidates, key=os.path.getmtime)


def latency_payload_to_seconds(payload, num_nodes=None):
    if "median_ms" in payload:
        return float(payload["median_ms"]) / 1000.0
    if "full_graph_ms" in payload:
        return float(payload["full_graph_ms"]) / 1000.0
    if "median_ms_per_batch" in payload:
        batch_size = int(payload.get("batch_size", 1024))
        if num_nodes is None:
            raise ValueError("大图批次口径换算缺少 num_nodes")
        num_batches = max(int(math.ceil(float(num_nodes) / float(batch_size))), 1)
        return float(payload["median_ms_per_batch"]) * num_batches / 1000.0
    raise KeyError(f"未找到可识别的时延字段: {sorted(payload.keys())}")


def extract_break_even_inputs(
    efficiency_payload,
    dataset,
    dense_method="gcn",
    sparse_method="msas_gnn",
    num_nodes=None,
):
    per_dataset = efficiency_payload.get("per_dataset", {})
    dataset_payload = per_dataset.get(dataset)
    if dataset_payload is None:
        raise KeyError(f"效率结果中缺少数据集: {dataset}")
    dense_payload = dataset_payload.get(dense_method)
    sparse_payload = dataset_payload.get(sparse_method)
    if dense_payload is None:
        raise KeyError(f"{dataset} 缺少稠密方法结果: {dense_method}")
    if sparse_payload is None:
        raise KeyError(f"{dataset} 缺少稀疏方法结果: {sparse_method}")

    t_dense = latency_payload_to_seconds(dense_payload, num_nodes=num_nodes)
    t_sparse = latency_payload_to_seconds(sparse_payload, num_nodes=num_nodes)
    if "preprocess_seconds" not in sparse_payload:
        raise KeyError(f"{dataset}/{sparse_method} 缺少 preprocess_seconds，无法计算 break-even")
    return {
        "t_pre": float(sparse_payload["preprocess_seconds"]),
        "t_dense": t_dense,
        "t_sparse": t_sparse,
        "dense_payload": dense_payload,
        "sparse_payload": sparse_payload,
    }
