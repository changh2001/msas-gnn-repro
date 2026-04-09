"""实验协议元信息。

用于把“新版实现口径”显式写入结果日志，避免旧结果和新结果在出表阶段混用。
"""

from __future__ import annotations


def build_protocol_metadata(cfg: dict) -> dict[str, object]:
    """提炼当前实现的重要协议字段。"""

    dataset = str(cfg.get("dataset", "unknown"))
    feature_cfg = cfg.get("feature_transform", {})
    ablation_id = str(cfg.get("ablation_id", "b5"))

    if dataset == "ogbn_arxiv":
        batch_protocol = "random_node_minibatch_recompute_phi"
    else:
        batch_protocol = "full_batch"

    if ablation_id == "sdgnn_pure":
        return {
            "table_protocol_version": 4,
            "phi_tilde_init": "l2_row_normalize_h_star",
            "w_phi_parameterization": "explicit_linear_xw",
            "w_phi_init": str(feature_cfg.get("init_protocol", "ridge_x_to_h_star")),
            "phase_w_protocol": "optimize_explicit_w_phi_then_recompute_phi_tilde",
            "phase_theta_protocol": "sdgnn_flat_lars_lasso",
            "candidate_protocol": "khop_plus_recursive_fanout_sampling",
            "sparsity_protocol": "candidate_pruning_rate",
            "frequency_mode": "none",
            "batch_protocol": batch_protocol,
        }

    return {
        "table_protocol_version": 4,
        "phi_tilde_init": "l2_row_normalize_h_star",
        "w_phi_parameterization": "explicit_linear_xw",
        "w_phi_init": str(feature_cfg.get("init_protocol", "ridge_x_to_h_star")),
        "phase_w_protocol": "optimize_explicit_w_phi_then_recompute_phi_tilde",
        "phase_theta_protocol": "layered_residual_cascade_lars_lasso",
        "candidate_protocol": "layered_bfs_hop_rings",
        "sparsity_protocol": "candidate_pruning_rate",
        "frequency_mode": "equal_weight",
        "batch_protocol": batch_protocol,
    }
