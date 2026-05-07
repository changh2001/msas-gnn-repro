"""谱相似性代理量与工程参考值计算。"""
from __future__ import annotations

import logging
import math

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _torch_matrix_to_scipy(matrix, *, use_weighted_support=False):
    import scipy.sparse as sp

    matrix = matrix.detach().cpu()
    if matrix.layout == torch.sparse_coo:
        coo = matrix.coalesce()
    elif matrix.layout == torch.sparse_csr:
        coo = matrix.to_sparse_coo().coalesce()
    elif matrix.layout == torch.sparse_csc:
        coo = matrix.to_sparse_coo().coalesce()
    else:
        coo = matrix.to_sparse_coo().coalesce()
    indices = coo.indices()
    raw_values = coo.values().numpy()
    if use_weighted_support:
        values = np.abs(raw_values).astype(np.float64)
    else:
        values = (np.abs(raw_values) > 0.0).astype(np.float64)
    row = indices[0].numpy()
    col = indices[1].numpy()
    return sp.coo_matrix((values, (row, col)), shape=tuple(coo.shape)).tocsr()


def _symmetrize_support(adjacency, *, use_weighted_support=False):
    adjacency = adjacency.tocsr().copy()
    adjacency.setdiag(0.0)
    adjacency.eliminate_zeros()
    if use_weighted_support:
        symmetric = adjacency.maximum(adjacency.T)
    else:
        symmetric = (adjacency + adjacency.T).astype(bool).astype(np.float64)
    symmetric.setdiag(0.0)
    symmetric.eliminate_zeros()
    return symmetric.tocsr()


def _edge_index_to_adjacency(edge_index, num_nodes):
    import scipy.sparse as sp

    edge_index = edge_index.detach().cpu() if isinstance(edge_index, torch.Tensor) else torch.as_tensor(edge_index)
    if edge_index.numel() == 0:
        return sp.csr_matrix((int(num_nodes), int(num_nodes)), dtype=np.float64)
    if edge_index.dim() != 2:
        raise ValueError("edge_index/support must be a rank-2 tensor or array")
    if edge_index.shape[0] == 2:
        row = edge_index[0].numpy()
        col = edge_index[1].numpy()
    elif edge_index.shape[1] == 2:
        row = edge_index[:, 0].numpy()
        col = edge_index[:, 1].numpy()
    else:
        raise ValueError("edge_index/support must have shape [2, m] or [m, 2]")
    values = np.ones_like(row, dtype=np.float64)
    return sp.coo_matrix((values, (row, col)), shape=(int(num_nodes), int(num_nodes))).tocsr()


def _extract_original_graph(edge_index_original_or_data, num_nodes):
    if hasattr(edge_index_original_or_data, "edge_index"):
        data = edge_index_original_or_data
        edge_index = data.edge_index
        if num_nodes is None:
            num_nodes = int(data.num_nodes)
    else:
        edge_index = edge_index_original_or_data
    if num_nodes is None:
        if isinstance(edge_index, torch.Tensor) and edge_index.numel() > 0:
            num_nodes = int(edge_index.max().item()) + 1
        else:
            edge_array = np.asarray(edge_index)
            num_nodes = int(edge_array.max()) + 1 if edge_array.size else 0
    return edge_index, int(num_nodes)


def _theta_or_support_to_adjacency(theta_or_support, num_nodes, *, use_weighted_support=False):
    theta = theta_or_support.theta if hasattr(theta_or_support, "theta") else theta_or_support
    if isinstance(theta, torch.Tensor):
        if theta.dim() == 2 and theta.shape[0] == int(num_nodes) and theta.shape[1] == int(num_nodes):
            return _torch_matrix_to_scipy(theta, use_weighted_support=use_weighted_support)
        return _edge_index_to_adjacency(theta, num_nodes)
    array = np.asarray(theta)
    if array.ndim == 2 and array.shape == (int(num_nodes), int(num_nodes)):
        import scipy.sparse as sp

        values = np.abs(array) if use_weighted_support else (np.abs(array) > 0.0).astype(np.float64)
        return sp.csr_matrix(values)
    return _edge_index_to_adjacency(torch.as_tensor(array, dtype=torch.long), num_nodes)


def build_proxy_adjacency(theta_fixed, use_weighted_support=False):
    """由 Θ^fixed/support 构造对称代理邻接矩阵；默认采用无权支撑集口径。"""
    num_nodes = int(theta_fixed.theta.shape[0]) if hasattr(theta_fixed, "theta") else int(theta_fixed.shape[0])
    adjacency = _theta_or_support_to_adjacency(
        theta_fixed,
        num_nodes,
        use_weighted_support=use_weighted_support,
    )
    return _symmetrize_support(adjacency, use_weighted_support=use_weighted_support)


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


def compute_sigma_proxy(
    edge_index_original_or_data,
    theta_or_support,
    num_nodes=None,
    k_lanczos=50,
    eps=1e-8,
    use_weighted_support=False,
    seed=42,
    num_eigs=None,
    tol=1e-6,
):
    """估计低频谱相似性代理量 σ̃_proxy。

    兼容旧调用 `compute_sigma_proxy(data, theta_fixed, num_eigs=50)`；新口径
    在原图低频特征向量上分别计算原图与稀疏图归一化拉普拉斯的 Rayleigh 商。
    """

    if num_eigs is not None:
        k_lanczos = int(num_eigs)
    edge_index, num_nodes = _extract_original_graph(edge_index_original_or_data, num_nodes)
    adjacency_orig = _symmetrize_support(
        _edge_index_to_adjacency(edge_index, num_nodes),
        use_weighted_support=False,
    )
    adjacency_proxy = _symmetrize_support(
        _theta_or_support_to_adjacency(
            theta_or_support,
            num_nodes,
            use_weighted_support=use_weighted_support,
        ),
        use_weighted_support=use_weighted_support,
    )
    lap_orig = normalized_laplacian_from_adjacency(adjacency_orig)
    lap_proxy = normalized_laplacian_from_adjacency(adjacency_proxy)
    evals, evecs = _low_frequency_eigenpairs(
        lap_orig,
        num_eigs=min(int(k_lanczos) + 1, max(int(num_nodes) - 1, 1)),
        seed=seed,
    )
    valid = np.flatnonzero(evals > float(tol))
    if valid.size:
        valid = valid[: int(k_lanczos)]
    if valid.size == 0:
        logger.warning("有效谱模态不足，σ̃_proxy 回退为 1.0")
        return {"sigma_proxy": 1.0, "rayleigh_min": 1.0, "rayleigh_max": 1.0, "num_modes": 0}

    ratios = []
    deviations = []
    for idx in valid:
        u = np.asarray(evecs[:, idx], dtype=np.float64).reshape(-1)
        q_orig = float(u @ (lap_orig @ u))
        q_proxy = float(u @ (lap_proxy @ u))
        ratio = q_proxy / (q_orig + float(eps))
        inv_ratio = (q_orig + float(eps)) / (q_proxy + float(eps))
        ratios.append(ratio)
        deviations.append(max(ratio, inv_ratio))

    ratios = np.asarray(ratios, dtype=np.float64)
    deviations = np.asarray(deviations, dtype=np.float64)
    rayleigh_min = float(np.min(ratios))
    rayleigh_max = float(np.max(ratios))
    sigma_proxy = float(np.max(deviations))
    return {
        "sigma_proxy": sigma_proxy,
        "rayleigh_min": rayleigh_min,
        "rayleigh_max": rayleigh_max,
        "num_modes": int(valid.size),
    }


def _low_frequency_eigenvalues(laplacian, num_eigs, exact_threshold=512):
    values, _ = _low_frequency_eigenpairs(laplacian, num_eigs, exact_threshold=exact_threshold)
    return values


def _low_frequency_eigenpairs(laplacian, num_eigs, exact_threshold=512, seed=42):
    if laplacian.shape[0] <= exact_threshold:
        import scipy.linalg as sla

        values, vectors = sla.eigh(laplacian.toarray())
        return np.asarray(values, dtype=np.float64), np.asarray(vectors, dtype=np.float64)

    import scipy.sparse.linalg as spla

    k = min(int(num_eigs), max(int(laplacian.shape[0]) - 2, 1))
    rng = np.random.default_rng(int(seed))
    v0 = rng.standard_normal(laplacian.shape[0])
    try:
        values, vectors = spla.eigsh(
            laplacian,
            k=k,
            which="SM",
            tol=1e-4,
            maxiter=5 * laplacian.shape[0],
            v0=v0,
            return_eigenvectors=True,
        )
    except spla.ArpackNoConvergence as exc:
        values = exc.eigenvalues
        vectors = exc.eigenvectors
        if values is None or len(values) == 0 or vectors is None:
            logger.warning("ARPACK未返回已收敛特征值，退化为全谱")
            import scipy.linalg as sla

            values, vectors = sla.eigh(laplacian.toarray())
        else:
            logger.warning("ARPACK部分收敛，仅使用%s个已收敛特征值", len(values))
    except Exception as exc:
        logger.warning("ARPACK失败(%s)，退化为全谱", exc)
        import scipy.linalg as sla

        values, vectors = sla.eigh(laplacian.toarray())
    order = np.argsort(np.asarray(values, dtype=np.float64))
    return (
        np.asarray(values, dtype=np.float64)[order],
        np.asarray(vectors, dtype=np.float64)[:, order],
    )


def compute_engineering_reference(sigma_proxy, features, poly_coeff_sum=1.0):
    """将式(5.9)按 ε_approx 口径做工程化归一，供补充实验图表对照。"""
    n = max(int(features.shape[0]), 1)
    feature_scale = float(torch.norm(features, p="fro").item() / math.sqrt(n))
    return float(2.0 * max(float(poly_coeff_sum), 0.0) * max(float(sigma_proxy) - 1.0, 0.0) * feature_scale)
