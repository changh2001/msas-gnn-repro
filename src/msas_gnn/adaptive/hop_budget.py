"""跳距维分层预算分配。

论文第4章当前口径不再预设每个节点固定总预算 K_i，而是先构造
BFS 环层 R_l(i)，再由保留率 p_l 与环层容量 |R_l(i)| 诱导 LARS
路径截断上界 k_i^(l)。
"""
from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def retention_rates(
    L: int,
    strategy: str = "spectral_gap_reference",
    *,
    lambda_gap: float = 0.0,
    p_base: float = 0.6,
    p_min: float = 0.1,
    kappa: float = 2.0,
    varrho: float = 0.7,
) -> torch.Tensor:
    """Return p_l for l=1..L under the paper strategies."""

    L = int(L)
    if L <= 0:
        return torch.zeros(0, dtype=torch.float32)
    levels = torch.arange(L, dtype=torch.float32)
    strategy = str(strategy)
    if strategy == "uniform":
        rates = torch.full((L,), float(p_base), dtype=torch.float32)
    elif strategy == "near_engineering":
        rates = float(p_base) * (float(varrho) ** levels)
    elif strategy == "spectral_gap_reference":
        rates = float(p_base) * torch.exp(-float(kappa) * float(lambda_gap) * levels)
    elif strategy == "reverse":
        forward = float(p_base) * (float(varrho) ** levels)
        rates = torch.flip(forward, dims=[0])
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    return rates.clamp(min=float(p_min), max=1.0)


def _candidate_ring_sizes(candidate_sets, n: int, L: int) -> torch.Tensor:
    sizes = torch.zeros((n, L), dtype=torch.long)
    for l in range(L):
        hop = candidate_sets[l] if l < len(candidate_sets) else {}
        for i in range(n):
            sizes[i, l] = len(hop.get(i, []))
    return sizes


def allocate_hop_budget_for_candidates(
    candidate_sets,
    tau: torch.Tensor,
    *,
    cfg: dict | None = None,
    strategy: str | None = None,
    lambda_gap: float = 0.0,
    p_base: float | None = None,
    p_min: float | None = None,
    kappa: float | None = None,
    varrho: float | None = None,
    rho: float | None = None,
    budget_scale: float | None = None,
) -> torch.Tensor:
    """Allocate k_i^(l) from actual BFS ring capacities."""

    cfg = cfg or {}
    hop_cfg = cfg.get("hop_dim", {}) if isinstance(cfg.get("hop_dim", {}), dict) else {}
    n = int(tau.shape[0])
    L = len(candidate_sets)
    capacities = _candidate_ring_sizes(candidate_sets, n=n, L=L)
    if str(cfg.get("ablation_id", "")) in {"b1", "b2", "b3", "b4", "b2_rnd"}:
        logger.info("hop_budget ablation=%s: 未启用分层跳距预算，使用环层容量上界", cfg.get("ablation_id"))
        return capacities
    strategy = strategy or hop_cfg.get("strategy", "spectral_gap_reference")
    rates = retention_rates(
        L,
        strategy=strategy,
        lambda_gap=lambda_gap,
        p_base=hop_cfg.get("p_base", 0.6) if p_base is None else p_base,
        p_min=hop_cfg.get("p_min", 0.1) if p_min is None else p_min,
        kappa=hop_cfg.get("kappa", 2.0) if kappa is None else kappa,
        varrho=hop_cfg.get("varrho", 0.7) if varrho is None else varrho,
    )
    scale = float(hop_cfg.get("rho", 1.0) if rho is None else rho)
    if budget_scale is not None:
        scale = float(budget_scale)
    raw = capacities.float() * rates.unsqueeze(0) * scale
    budgets = torch.round(raw).long()
    budgets = torch.where((capacities > 0) & (budgets < 1), torch.ones_like(budgets), budgets)
    budgets = torch.minimum(budgets, capacities)
    logger.info(
        "hop_budget strategy=%s rates=%s rho_scale=%.3f capacity_mean=%.2f budget_mean=%.2f",
        strategy,
        ",".join(f"{x:.3f}" for x in rates.tolist()),
        scale,
        float(capacities.float().mean().item()) if capacities.numel() else 0.0,
        float(budgets.float().mean().item()) if budgets.numel() else 0.0,
    )
    return budgets


def allocate_hop_budget(tau, k=50, L=3, strategy="spectral_gap_reference", **kwargs):
    """Compatibility fallback used by tests or legacy callers without candidate rings.

    The real training path calls :func:`allocate_hop_budget_for_candidates` after BFS
    rings are known. This fallback assumes each hop has a nominal capacity of k/L.
    """

    n = int(tau.shape[0])
    nominal = max(int(round(float(k) / max(int(L), 1))), 1)
    candidate_sets = [
        {i: list(range(nominal)) for i in range(n)}
        for _ in range(int(L))
    ]
    lambda_gap = float(kwargs.pop("lambda_gap", 0.0) or 0.0)
    return allocate_hop_budget_for_candidates(
        candidate_sets,
        tau,
        strategy=strategy,
        cfg={"hop_dim": kwargs},
        lambda_gap=lambda_gap,
    )
