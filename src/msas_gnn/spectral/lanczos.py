"""Lanczos迭代：近似L̃前K_eig个特征对。
对应论文§3.1.1，复杂度O(m·K_eig)。

重要约定（防特征值编号混淆）：
  eigenvalues[0] ≈ 0（最小，λ_0，连通图保证）
  eigenvalues[1] = λ_gap（谱间隙）→ 调用方用 eigenvalues[1] 获取谱间隙，不得用 eigenvalues[0]
  孤立节点：谱能量置0；非连通图：λ_1=0
数值稳定性：full reorthogonalization，主线默认K_eig=50
"""
import logging, numpy as np
import scipy.sparse.linalg as spla
import torch
logger = logging.getLogger(__name__)

def lanczos_eigenpairs(L_scipy, K_eig=50, which="SM"):
    """近似L̃前K_eig个特征对（最小特征值）。返回(eigenvalues升序, eigenvectors)。"""
    n = L_scipy.shape[0]
    K = min(K_eig, n-2)
    if K <= 0:
        import scipy.linalg as sla
        v, vecs = sla.eigh(L_scipy.toarray())
        K2 = min(50, n)
        return torch.from_numpy(v[:K2].astype(np.float32)), torch.from_numpy(vecs[:, :K2].astype(np.float32))
    try:
        v0 = np.ones(n)/np.sqrt(n)
        vals, vecs = spla.eigsh(L_scipy, k=K, which=which, tol=1e-6, maxiter=10*n, v0=v0)
        idx = np.argsort(vals); vals, vecs = vals[idx], vecs[:,idx]
        if abs(vals[0]) > 0.05: logger.warning(f"eigenvalues[0]={vals[0]:.4f}偏离0，图可能不连通")
        return torch.from_numpy(vals.astype(np.float32)), torch.from_numpy(vecs.astype(np.float32))
    except Exception as e:
        logger.warning(f"ARPACK失败({e})，退化为全谱")
        import scipy.linalg as sla
        v, vecs = sla.eigh(L_scipy.toarray())
        K2 = min(K, n)
        return torch.from_numpy(v[:K2].astype(np.float32)), torch.from_numpy(vecs[:, :K2].astype(np.float32))
