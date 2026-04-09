"""Θ^fixed封装：固化、序列化、稀疏统计。稀疏格式torch.sparse_csr。"""
import logging, os
import torch
from msas_gnn.typing import ThetaFixed
logger = logging.getLogger(__name__)

def save_theta_fixed(tf: ThetaFixed, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "theta": tf.theta,
            "k_bar": tf.k_bar,
            "sparsity": tf.sparsity,
            "support_total": tf.support_total,
            "candidate_total": tf.candidate_total,
        },
        path,
    )
    logger.info(f"Θ^fixed已保存：{path}")

def load_theta_fixed(path: str) -> ThetaFixed:
    d = torch.load(path, map_location="cpu")
    return ThetaFixed(**d)

def get_sparsity_stats(theta) -> dict:
    t = theta.to_dense() if theta.is_sparse else theta
    nnz = (t.abs() > 1e-8).float().sum(dim=0)
    k_bar = nnz.mean().item()
    return {"k_bar":k_bar, "sparsity":1.0-k_bar/t.shape[0], "nnz":int(nnz.sum().item())}
