"""节点谱能量 E_spectral(i) = Σ_k w_k·λ_k·u_k(i)²。
对应论文§3.2.2定义3.2。默认采用论文中的等权实例化：
对所有非平凡特征值分量取相同权重，零特征值分量不参与归一化。
"""
import logging
import torch
logger = logging.getLogger(__name__)

def compute_spectral_energy(eigenvalues, eigenvectors, weights=None):
    """等权实例化节点谱能量。复杂度O(n·K_eig)。
    边界：孤立节点E=0（自然处理）；极小特征值忽略（防浮点累积）。
    """
    valid = eigenvalues > 1e-8
    if weights is None:
        weights = valid.to(dtype=eigenvalues.dtype)
        denom = weights.sum().clamp_min(1.0)
        weights = weights / denom
    wl = weights * valid.to(dtype=eigenvalues.dtype) * eigenvalues
    E = (eigenvectors ** 2) @ wl
    logger.debug(f"E_spectral min={E.min():.4e} max={E.max():.4e} mean={E.mean():.4e}")
    return E
