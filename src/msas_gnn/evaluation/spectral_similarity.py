"""谱相似性代理量与工程参考值计算。"""
from __future__ import annotations

import logging
import math

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _torch_sparse_to_scipy(matrix):
    import scipy.sparse as sp

    coo = matrix.detach().cpu().to_sparse_coo().coalesce()
    indices = coo.indices()
    values = coo.values().numpy()
    row = indices[0].numpy()
    col = indices[1].numpy()
    return sp.coo_matrix((values, (row, col)), shape=tuple(coo.shape)).tocsr()


def build_proxy_adjacency(theta_fixed):
    """按补充谱代理实验的口径，由 Θ^fixed 构造对称非负代理邻接矩阵。"""
    theta = theta_fixed.theta
    if not theta.is_sparse:
        theta = theta.to_sparse()
    theta_abs = torch.abs(theta.detach().cpu())
    adjacency = _torch_sparse_to_scipy(theta_abs)
    return ((adjacency + adjacency.T) * 0.5).tocsr()


def normalized_laplacian_from_adjacency(adjacency):
    import scipy.sparse as sp

    adjacency = adjacency.tocsr().copy()
    adjacency.setdiag(0.0)
    adjacency.eliminate_zeros()
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    with np.errstate(divide="ignore"):
        inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)
    d_inv = sp.diags(inv_sqrt)
    identity = sp.eye(adjacency.shape[0], format="csr")
    return (identity - d_inv @ adjacency @ d_inv).tocsr()


def compute_sigma_proxy(data, theta_fixed, num_eigs=50, tol=1e-6):
    """基于原图/代理图低频特征值比值近似估计 σ̃_proxy。"""
    from msas_gnn.spectral.laplacian import compute_normalized_laplacian_scipy

    lap_orig = compute_normalized_laplacian_scipy(data)
    lap_proxy = normalized_laplacian_from_adjacency(build_proxy_adjacency(theta_fixed))
    orig = _low_frequency_eigenvalues(lap_orig, num_eigs=min(int(num_eigs), max(int(data.num_nodes) - 2, 1)))
    proxy = _low_frequency_eigenvalues(lap_proxy, num_eigs=min(int(num_eigs), max(int(data.num_nodes) - 2, 1)))
    positive_orig = orig[orig > tol]
    positive_proxy = proxy[proxy > tol]
    count = min(len(positive_orig), len(positive_proxy))
    ratios = np.maximum(positive_proxy[:count] / positive_orig[:count], tol) if count else np.asarray([])
    if ratios.size == 0:
        logger.warning("有效谱模态不足，σ̃_proxy 回退为 1.0")
        return {"sigma_proxy": 1.0, "rayleigh_min": 1.0, "rayleigh_max": 1.0, "num_modes": 0}

    rayleigh_min = min(ratios)
    rayleigh_max = max(ratios)
    sigma_proxy = max(rayleigh_max, 1.0 / max(rayleigh_min, tol))
    return {
        "sigma_proxy": float(sigma_proxy),
        "rayleigh_min": float(rayleigh_min),
        "rayleigh_max": float(rayleigh_max),
        "num_modes": int(len(ratios)),
    }


def _low_frequency_eigenvalues(laplacian, num_eigs, exact_threshold=512):
    if laplacian.shape[0] <= exact_threshold:
        import scipy.linalg as sla

        return np.asarray(sla.eigvalsh(laplacian.toarray()), dtype=np.float64)

    import scipy.sparse.linalg as spla

    k = min(int(num_eigs), max(int(laplacian.shape[0]) - 2, 1))
    try:
        values = spla.eigsh(
            laplacian,
            k=k,
            which="SM",
            tol=1e-4,
            maxiter=5 * laplacian.shape[0],
            return_eigenvectors=False,
        )
    except spla.ArpackNoConvergence as exc:
        values = exc.eigenvalues
        if values is None or len(values) == 0:
            logger.warning("ARPACK未返回已收敛特征值，退化为全谱")
            import scipy.linalg as sla

            values = sla.eigvalsh(laplacian.toarray())
        else:
            logger.warning("ARPACK部分收敛，仅使用%s个已收敛特征值", len(values))
    except Exception as exc:
        logger.warning("ARPACK失败(%s)，退化为全谱", exc)
        import scipy.linalg as sla

        values = sla.eigvalsh(laplacian.toarray())
    return np.sort(np.asarray(values, dtype=np.float64))


def compute_engineering_reference(sigma_proxy, features, poly_coeff_sum=1.0):
    """将式(5.9)按 ε_approx 口径做工程化归一，供补充实验图表对照。"""
    n = max(int(features.shape[0]), 1)
    feature_scale = float(torch.norm(features, p="fro").item() / math.sqrt(n))
    return float(2.0 * max(float(poly_coeff_sum), 0.0) * max(float(sigma_proxy) - 1.0, 0.0) * feature_scale)
