"""三维参数统一封装（正文 B0~B5 与对照变体共享）。对应论文§4.4。"""
import logging
import torch
from msas_gnn.typing import MetricBundle, AdaptiveParamSet
from msas_gnn.adaptive.frequency_correction import apply_frequency_correction
from msas_gnn.adaptive.tau_builder import (
    build_tau,
    perturb_tau,
    resolve_energy_threshold,
)
from msas_gnn.adaptive.hop_budget import allocate_hop_budget
from msas_gnn.adaptive.validators import validate_adaptive_params
logger = logging.getLogger(__name__)

def build_adaptive_params(bundle: MetricBundle | None, cfg: dict, data=None) -> AdaptiveParamSet:
    """整合第3→第4章参数链，输出AdaptiveParamSet。消融由cfg["ablation_id"]控制。"""
    abl = cfg.get("ablation_id", "b5")
    if abl == "sdgnn_pure":
        if data is None:
            raise ValueError("sdgnn_pure 需要 data 以构造全局 λ 与占位预算")
        n = int(data.num_nodes)
        lambda_reg = float(cfg.get("lambda_reg", 1e-3))
        tau = torch.full((n,), lambda_reg, dtype=torch.float32)
        k = int(cfg.get("lars", {}).get("k", 50))
        kb = torch.full((n, 1), k, dtype=torch.long)
        freq_weights = torch.ones(1, dtype=torch.float32)
        logger.info("[sdgnn_pure] 使用原始口径全局 λ=%.2e", lambda_reg)
        return AdaptiveParamSet(tau=tau, k_budget=kb, freq_weights=freq_weights)
    nc = cfg.get("node_dim", {})
    hc = cfg.get("hop_dim", {})
    fc = cfg.get("frequency", {})
    freq_weights, corrected_energy = apply_frequency_correction(bundle, mode=fc.get("mode", "equal_weight"))
    tau_bundle = bundle._replace(spectral_energy=corrected_energy)
    explicit_threshold = nc.get("e_threshold")
    energy_threshold = resolve_energy_threshold(
        tau_bundle,
        c_e=nc.get("c_e", 1.0),
        explicit_threshold=explicit_threshold,
    )
    if abl == "b0":
        tau = build_tau(tau_bundle, global_tau=cfg.get("lambda_reg", 1e-3))
    else:
        use_spectral = nc.get("use_spectral_energy", abl != "b0")
        use_cent = nc.get("use_centrality", abl not in ("b0", "b1"))
        use_kc = nc.get("use_kcore", abl not in ("b0", "b1", "b2", "b2_rnd"))
        use_ent = nc.get("use_entropy", abl not in ("b0", "b1", "b2", "b3", "b2_rnd"))
        tau = build_tau(
            tau_bundle,
            tau_base=nc.get("tau_base", 1e-3),
            tau_min=nc.get("tau_min", 1e-7),
            e_threshold=energy_threshold,
            beta_tau=nc.get("beta_tau", 1.0),
            gamma=nc.get("gamma", 0.5) if use_cent else 0.0,
            delta=nc.get("delta", 0.3) if use_kc else 0.0,
            omega_h=nc.get("omega_h", 0.2) if use_ent else 0.0,
            use_spectral_energy=use_spectral,
            use_centrality=use_cent,
            use_kcore=use_kc,
            use_entropy=use_ent,
        )
        if abl == "b2_rnd" or nc.get("random_tau_perturbation", False):
            tau = perturb_tau(
                tau,
                tau_min=nc.get("tau_min", 1e-7),
                tau_base=nc.get("tau_base", 1e-3),
                random_seed=int(cfg.get("seed", 42)),
                log_scale=float(nc.get("random_tau_log_scale", 0.5)),
            )
    k = cfg.get("lars", {}).get("k", 50)
    L = hc.get("L",3)
    kb = allocate_hop_budget(tau, k=k, L=L,
                              strategy=hc.get("strategy","spectral_gap_reference"),
                              p_base=hc.get("p_base", 0.6),
                              p_min=hc.get("p_min", 0.1),
                              kappa=hc.get("kappa", 2.0),
                              varrho=hc.get("varrho", 0.7),
                              rho=hc.get("rho", 1.0),
                              lambda_gap=cfg.get("lambda_gap", 0.0))
    params = AdaptiveParamSet(tau=tau, k_budget=kb, freq_weights=freq_weights)
    validate_adaptive_params(params, cfg)
    return params
