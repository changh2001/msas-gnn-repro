"""归一化局部图熵 H_norm(i)。对应论文§3.2.2式(3.9)-(3.10)。"""
import logging, math
import torch
logger = logging.getLogger(__name__)

def _compute_gaussian_gamma(data) -> float:
    src, dst = data.edge_index
    mask = src != dst
    if mask.sum().item() == 0:
        return 1.0
    diffs = data.x[src[mask]] - data.x[dst[mask]]
    sq_dist = diffs.pow(2).sum(dim=1)
    sigma_sq = float(torch.median(sq_dist).item())
    if sigma_sq <= 1e-12:
        return 1.0
    return 1.0 / (2.0 * sigma_sq)


def compute_local_entropy(data, weight_mode="gaussian", gamma_x=None) -> torch.Tensor:
    """基于邻域边权分布的局部图熵，默认采用论文中的高斯核权重。"""
    n = data.num_nodes
    src, dst = data.edge_index
    mask = src != dst
    src, dst = src[mask], dst[mask]
    H = torch.zeros(n, dtype=torch.float32)
    if src.numel() == 0:
        return H

    if weight_mode == "gaussian" and getattr(data, "x", None) is not None:
        gamma = _compute_gaussian_gamma(data) if gamma_x is None else float(gamma_x)
        weights = torch.exp(-gamma * (data.x[src] - data.x[dst]).pow(2).sum(dim=1))
    else:
        weights = torch.ones(src.shape[0], dtype=torch.float32)

    for i in range(n):
        nb_mask = src == i
        d_i = int(nb_mask.sum().item())
        if d_i <= 1:
            continue
        w = weights[nb_mask]
        z = float(w.sum().item())
        if z <= 1e-12:
            continue
        p = w / z
        H[i] = (-(p * (p + 1e-10).log()).sum() / math.log(d_i)).clamp(0.0, 1.0)
    logger.debug(f"H_norm mean={H.mean():.4f}")
    return H
