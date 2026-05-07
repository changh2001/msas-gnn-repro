"""Phase-Θ完整实现（分层残差级联LARS主循环）。对应论文§5.1.2算法5.3。
残差逐层更新顺序须严格保持（方案2核心约束）。
"""
import logging
import torch
from msas_gnn.typing import AdaptiveParamSet, ThetaFixed
logger = logging.getLogger(__name__)


def _solve_self_channel(target, phi_self, tau):
    """单变量 Lasso 自特征通道：min 1/2||a*phi_i-target||^2 + tau|a|."""
    dot = float(torch.dot(phi_self, target).item())
    denom = float(torch.dot(phi_self, phi_self).item())
    if denom <= 1e-12:
        return 0.0
    shrink = max(abs(dot) - float(tau), 0.0)
    if shrink == 0.0:
        return 0.0
    return float(torch.sign(torch.tensor(dot)).item() * shrink / denom)


def _build_theta_fixed(n, phi_dtype, row_idx, col_idx, values, selected_counts, total_candidates, node_indices=None):
    if values:
        indices = torch.tensor([row_idx, col_idx], dtype=torch.long)
        values_t = torch.tensor(values, dtype=phi_dtype)
        theta_sparse = torch.sparse_coo_tensor(indices, values_t, size=(n, n)).coalesce().to_sparse_csr()
    else:
        theta_sparse = torch.sparse_coo_tensor(
            torch.zeros((2, 0), dtype=torch.long),
            torch.zeros((0,), dtype=phi_dtype),
            size=(n, n),
        ).to_sparse_csr()
    denom = float(len(node_indices)) if node_indices is not None else float(n)
    selected = selected_counts if node_indices is None else selected_counts[node_indices]
    total_selected = float(selected.sum().item())
    k_bar = float(total_selected / max(denom, 1.0))
    sparsity = 1.0 - total_selected / max(float(total_candidates), 1.0)
    return ThetaFixed(
        theta=theta_sparse,
        k_bar=k_bar,
        sparsity=sparsity,
        support_total=int(total_selected),
        candidate_total=int(total_candidates),
    )


def _resolve_theta_solver_mode(cfg):
    lars_cfg = cfg.get("lars", {}) if isinstance(cfg, dict) else {}
    mode = lars_cfg.get("theta_solver_mode", lars_cfg.get("scheme", "residual_cascade"))
    mode = str(mode or "residual_cascade")
    if mode not in {"residual_cascade", "shared_target"}:
        raise ValueError(
            "Unknown theta_solver_mode=%s; expected residual_cascade or shared_target" % mode
        )
    return mode


def run_phase_theta(
    h_star,
    phi_tilde,
    params: AdaptiveParamSet,
    candidate_sets,
    cfg,
    node_indices=None,
) -> ThetaFixed:
    """Phase-Θ：固定W，优化Θ（LARS）。
    复杂度(推理主项)：训练O(n·k·d·L)；推理O(n·k̄·d)（Φ̃已离线缓存）
    """
    n, d = h_star.shape; L = params.k_budget.shape[1]
    solver_mode = _resolve_theta_solver_mode(cfg)
    nodes = range(n) if node_indices is None else [int(i) for i in node_indices]
    row_idx = []
    col_idx = []
    values = []
    nnz_counts = torch.zeros(n, dtype=torch.float32)
    total_candidates = 0
    for i in nodes:
        recon_i = torch.zeros_like(h_star[i])
        tau_i = float(params.tau[i].item())
        total_candidates += 1
        self_value = _solve_self_channel(h_star[i], phi_tilde[i], tau_i)
        if abs(self_value) > 1e-8:
            row_idx.append(int(i))
            col_idx.append(int(i))
            values.append(self_value)
            recon_i = recon_i + self_value * phi_tilde[i]
            nnz_counts[i] += 1.0
        for l in range(L):
            hop_cands = candidate_sets[l]
            cands = hop_cands.get(i, [])
            total_candidates += len(cands)
            if not cands:
                continue
            budget = int(params.k_budget[i, l].item())
            if budget == 0:
                continue
            if solver_mode == "shared_target":
                r_il = h_star[i]
            else:
                r_il = h_star[i] - recon_i
            phi_c = phi_tilde[cands]
            from msas_gnn.decomposition.lars_solver import lars_lasso_single
            th = lars_lasso_single(r_il, phi_c, tau_i, budget)
            non_zero = 0
            for idx, j in enumerate(cands):
                if idx >= len(th):
                    break
                value = float(th[idx].item())
                if abs(value) <= 1e-8:
                    continue
                row_idx.append(int(j))
                col_idx.append(int(i))
                values.append(value)
                recon_i = recon_i + value * phi_tilde[j]
                non_zero += 1
            nnz_counts[i] += float(non_zero)
    logger.info(
        "Phase-Θ完成节点求解 mode=%s total_nodes=%s total_hops=%s",
        solver_mode,
        len(nodes),
        L,
    )
    theta_fixed = _build_theta_fixed(
        n=n,
        phi_dtype=phi_tilde.dtype,
        row_idx=row_idx,
        col_idx=col_idx,
        values=values,
        selected_counts=nnz_counts,
        total_candidates=total_candidates,
        node_indices=nodes if node_indices is not None else None,
    )
    logger.info(
        "Phase-Θ完成 k̄=%.1f 候选剪枝率=%.4f (selected=%s candidates=%s)",
        theta_fixed.k_bar,
        theta_fixed.sparsity,
        theta_fixed.support_total,
        int(total_candidates),
    )
    return theta_fixed


def _flatten_bfs_candidates(i, candidate_sets):
    cands = [int(i)]
    seen = {int(i)}
    for hop in candidate_sets:
        for j in hop.get(i, []):
            j = int(j)
            if j in seen:
                continue
            seen.add(j)
            cands.append(j)
    return cands


def run_phase_theta_bfs_flat(
    h_star,
    phi_tilde,
    params: AdaptiveParamSet,
    candidate_sets,
    cfg,
    node_indices=None,
) -> ThetaFixed:
    """B0 Phase-Θ：分层 BFS 候选池展平后求解单个 LARS/Lasso 子问题。"""

    n, _ = h_star.shape
    nodes = range(n) if node_indices is None else [int(i) for i in node_indices]
    row_idx = []
    col_idx = []
    values = []
    nnz_counts = torch.zeros(n, dtype=torch.float32)
    total_candidates = 0
    max_lars_iter = int(cfg.get("lars", {}).get("k", 50) or 50)

    from msas_gnn.decomposition.lars_solver import lars_lasso_single

    for i in nodes:
        cands = _flatten_bfs_candidates(i, candidate_sets)
        total_candidates += len(cands)
        if not cands:
            continue
        tau_i = float(params.tau[i].item())
        budget = min(max_lars_iter, len(cands))
        th = lars_lasso_single(h_star[i], phi_tilde[cands], tau_i, budget)
        non_zero = 0
        for idx, j in enumerate(cands):
            if idx >= len(th):
                break
            value = float(th[idx].item())
            if abs(value) <= 1e-8:
                continue
            row_idx.append(int(j))
            col_idx.append(int(i))
            values.append(value)
            non_zero += 1
        nnz_counts[i] = float(non_zero)

    theta_fixed = _build_theta_fixed(
        n=n,
        phi_dtype=phi_tilde.dtype,
        row_idx=row_idx,
        col_idx=col_idx,
        values=values,
        selected_counts=nnz_counts,
        total_candidates=total_candidates,
        node_indices=nodes if node_indices is not None else None,
    )
    logger.info(
        "Phase-Θ[B0-flat]完成 k̄=%.1f 候选剪枝率=%.4f (selected=%s candidates=%s)",
        theta_fixed.k_bar,
        theta_fixed.sparsity,
        theta_fixed.support_total,
        theta_fixed.candidate_total,
    )
    return theta_fixed


def run_phase_theta_sdgnn(
    h_star,
    phi_tilde,
    params: AdaptiveParamSet,
    candidate_set,
    cfg,
    node_indices=None,
) -> ThetaFixed:
    """原始 SDGNN 风格的 Phase-Θ：平坦候选池上的单次 LARS/Lasso。"""

    n, _ = h_star.shape
    nodes = range(n) if node_indices is None else [int(i) for i in node_indices]
    row_idx = []
    col_idx = []
    values = []
    nnz_counts = torch.zeros(n, dtype=torch.float32)
    total_candidates = 0
    max_lars_iter = int(cfg.get("sdgnn_pure", {}).get("max_lars_iter", 0) or 0)

    from msas_gnn.decomposition.lars_solver import lars_lasso_single

    for i in nodes:
        cands = candidate_set.get(i, [])
        total_candidates += len(cands)
        if not cands:
            continue
        tau_i = float(params.tau[i].item())
        phi_c = phi_tilde[cands]
        budget = len(cands)
        if max_lars_iter > 0:
            budget = min(budget, max_lars_iter)
        th = lars_lasso_single(h_star[i], phi_c, tau_i, budget)
        non_zero = 0
        for idx, j in enumerate(cands):
            if idx >= len(th):
                break
            value = float(th[idx].item())
            if abs(value) <= 1e-8:
                continue
            row_idx.append(int(j))
            col_idx.append(int(i))
            values.append(value)
            non_zero += 1
        nnz_counts[i] = float(non_zero)

    theta_fixed = _build_theta_fixed(
        n=n,
        phi_dtype=phi_tilde.dtype,
        row_idx=row_idx,
        col_idx=col_idx,
        values=values,
        selected_counts=nnz_counts,
        total_candidates=total_candidates,
        node_indices=nodes if node_indices is not None else None,
    )
    logger.info(
        "Phase-Θ[sdgnn_orig]完成 k̄=%.1f 候选剪枝率=%.4f (selected=%s candidates=%s)",
        theta_fixed.k_bar,
        theta_fixed.sparsity,
        theta_fixed.support_total,
        theta_fixed.candidate_total,
    )
    return theta_fixed
