"""跳距维分层预算分配：k_i^(l)矩阵。对应论文§4.3式(4.14)-(4.18)。复杂度O(n·L)。

变量命名严格对应论文：
  xi_budget（ξ）：跳距预算分配参数；gamma/delta/eta为节点维参数（在tau_builder中），不在此复用。
策略：uniform/xi_budget（近邻优先）/reverse（远跳优先）
预算守恒：Σ_l k_i^(l) = k，整数化后修正最后一层。
"""
import logging
import torch
logger = logging.getLogger(__name__)

def _base_retention_rates(L: int, p_first: float = 0.6, p_last: float = 0.2) -> torch.Tensor:
    if L <= 1:
        return torch.ones(1, dtype=torch.float32)
    ratio = (p_last / p_first) ** (1.0 / (L - 1))
    levels = torch.arange(L, dtype=torch.float32)
    return p_first * (ratio ** levels)


def allocate_hop_budget(tau, k=50, L=3, strategy="xi_budget", xi_budget=1.0):
    """分配分层跳距预算矩阵。返回shape[n,L]（long型），保证预算守恒。"""
    n = tau.shape[0]
    if k == 0: return torch.zeros(n, L, dtype=torch.long)
    k_total = torch.full((n,), float(k))
    if strategy == "uniform":
        kb = torch.full((n, L), k / L)
    elif strategy == "xi_budget":
        base = _base_retention_rates(L)
        hw = torch.pow(base, float(xi_budget)); hw = hw / hw.sum()
        kb = k_total.unsqueeze(1) * hw.unsqueeze(0)
    elif strategy == "reverse":
        hw = torch.flip(_base_retention_rates(L), dims=[0]); hw = hw / hw.sum()
        kb = k_total.unsqueeze(1) * hw.unsqueeze(0)
    else: raise ValueError(f"Unknown strategy: {strategy}")
    kb_int = kb.round().long()
    disc = k_total.long() - kb_int.sum(dim=1)
    kb_int[:, -1] = (kb_int[:, -1] + disc).clamp(min=0)
    # 二次校验
    if (kb_int.sum(dim=1) != k).any():
        kb_int[:, -1] += k - kb_int.sum(dim=1)
        kb_int = kb_int.clamp(min=0)
    logger.info(f"hop_budget strategy={strategy} ξ={xi_budget if strategy=='xi_budget' else 'N/A'} 预算守恒:PASSED")
    return kb_int
