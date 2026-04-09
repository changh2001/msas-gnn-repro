"""单节点单层 LARS/Lasso 求解器。对应论文§5.1.1算法5.2（方案2：分层残差级联）。
min_θ (1/2)‖r-Φ̃·θ‖²+τ(i)·‖θ‖₁  s.t. ‖θ‖₀≤k_i^(l)

方案1（整体优化）vs 方案2（分层残差级联）：
  方案1：跨层统一优化，全局最优但计算代价高
  方案2：逐层优化，上层残差传递给下层→本仓库实现方案2

实现说明：
- 采用 scikit-learn 的 LassoLars，真实走 LARS/Lasso 路径；
- 由于 sklearn 的目标函数是 (1 / (2n))‖y-Xw‖² + alpha‖w‖₁，
  这里按 n=d 做尺度换算，使用 alpha = τ / d；
- max_iter 直接对应论文中的预算上界 k_i^(l)。

数值稳定性：Cholesky eps 由 reg_eps 控制。
边界：候选集空/budget=0返回零向量；不收敛时回退并记录警告。
"""
import logging
import warnings

import numpy as np
import torch

logger = logging.getLogger(__name__)

def lars_lasso_single(r, phi_candidates, tau, budget, reg_eps=1e-8):
    """r:[d] phi_candidates:[m,d] → theta:[m] 稀疏权重，非零数≤budget。"""
    m = phi_candidates.shape[0]
    if m == 0:
        return torch.zeros(0, dtype=r.dtype)
    if budget == 0:
        return torch.zeros(m, dtype=r.dtype)
    budget = min(budget, m)
    d = int(r.shape[0])
    if d == 0:
        return torch.zeros(m, dtype=r.dtype)

    y = r.detach().cpu().numpy().astype(np.float64)
    x = phi_candidates.detach().cpu().numpy().astype(np.float64).T
    alpha = max(float(tau) / max(d, 1), 0.0)

    try:
        from sklearn.linear_model import LassoLars

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = LassoLars(
                alpha=alpha,
                fit_intercept=False,
                precompute="auto",
                max_iter=budget,
                fit_path=False,
                eps=float(reg_eps),
            )
            model.fit(x, y)
        theta = np.asarray(model.coef_, dtype=np.float32).reshape(-1)
    except Exception as exc:
        logger.warning("LassoLars 失败(%s)，回退为零向量", exc)
        return torch.zeros(m, dtype=r.dtype)

    if theta.shape[0] != m:
        logger.warning("LassoLars 返回维度异常=%s，回退为零向量", theta.shape)
        return torch.zeros(m, dtype=r.dtype)

    nonzero = np.flatnonzero(np.abs(theta) > 1e-8)
    if nonzero.size > budget:
        keep = np.argsort(np.abs(theta))[::-1][:budget]
        pruned = np.zeros_like(theta)
        pruned[keep] = theta[keep]
        theta = pruned

    return torch.from_numpy(theta).to(dtype=r.dtype)
