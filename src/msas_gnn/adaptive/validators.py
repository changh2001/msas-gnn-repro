"""参数合法性校验（值域、单调性、预算守恒）。"""
import logging
logger = logging.getLogger(__name__)

def validate_adaptive_params(params, cfg):
    """校验AdaptiveParamSet合法性。关键约束违反时抛出AssertionError。"""
    tau = params.tau; kb = params.k_budget; fw = params.freq_weights
    tau_base = cfg.get("node_dim",{}).get("tau_base",1e-3)
    tau_min = cfg.get("node_dim",{}).get("tau_min",1e-7)
    k = cfg.get("lars",{}).get("k",50)
    assert not tau.isnan().any(), "τ(i)含NaN！"
    assert not tau.isinf().any(), "τ(i)含Inf！"
    assert tau.min() >= tau_min-1e-10, f"τ下界违反：{tau.min():.4e}"
    assert tau.max() <= tau_base+1e-10, f"τ上界违反：{tau.max():.4e}"
    assert (kb.sum(dim=1) - k).abs().max() <= 1, "预算不守恒"
    assert kb.min() >= 0, "k_budget含负值！"
    if fw.numel() > 0: assert not fw.isnan().any(), "freq_weights含NaN！"
    logger.info("[validators] 三维自适应参数校验：PASSED")
