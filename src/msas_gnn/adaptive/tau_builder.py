"""节点级自适应正则系数τ(i)构造。对应论文§4.2式(4.7)-(4.13)算法4.1。复杂度O(n)。

多因子融合公式：
  τ(i) = clip(τ_base·exp(-β_τ·E_spectral(i)/E_threshold) / (1+γ·C_deg(i)+δ·core(i)/k_max+η·H_norm(i)), τ_min, τ_base)

消融：B0全局统一λ；B1仅谱能量；B2+度中心性；B3+k-core；B4/B5+局部熵
B2-RND：沿B2路径先构造τ(i)，再对τ(i)施加随机扰动（验证并非“任意随机修正都有效”）
"""
import logging
import torch
logger = logging.getLogger(__name__)

def resolve_energy_threshold(bundle, c_e=1.0, explicit_threshold=None):
    """按论文公式 E_threshold = c_E * lambda_gap / bar_lambda 计算。"""
    if explicit_threshold is not None:
        return float(explicit_threshold)
    evals = bundle.eigenvalues
    valid = evals[evals > 1e-8]
    if valid.numel() == 0:
        logger.warning("无有效非平凡特征值，E_threshold回退为1.0")
        return 1.0
    lambda_gap = float(valid[0].item())
    bar_lambda = float(valid.mean().item())
    if bar_lambda <= 1e-12:
        logger.warning("bar_lambda≈0，E_threshold回退为1.0")
        return 1.0
    threshold = float(c_e) * lambda_gap / bar_lambda
    if threshold <= 1e-12:
        logger.warning("E_threshold≈0，回退为1.0")
        return 1.0
    return threshold


def build_tau(bundle, tau_base=1e-3, tau_min=1e-7, e_threshold=1.0,
              beta_tau=1.0, gamma=0.5, delta=0.3, eta=0.2,
              use_spectral_energy=True, use_centrality=True, use_kcore=True, use_entropy=True,
              global_tau=None):
    """构造τ(i)向量。返回shape[n]，∈[tau_min,tau_base]，无NaN/Inf。"""
    n = bundle.spectral_energy.shape[0]
    if global_tau is not None:
        logger.info(f"B0全局统一τ={global_tau:.2e}")
        return torch.full((n,), global_tau, dtype=bundle.spectral_energy.dtype)
    if use_spectral_energy:
        E = bundle.spectral_energy
    else:
        E = torch.zeros(n, dtype=bundle.spectral_energy.dtype)
    E_norm = E / (float(e_threshold) + 1e-8)
    exp_term = torch.exp(-beta_tau * E_norm).clamp(1e-10, 1.0)
    denom = torch.ones(n, dtype=bundle.spectral_energy.dtype)
    if use_centrality:
        denom = denom + gamma * bundle.c_deg
    if use_kcore:
        km = bundle.core.max().item()
        if km > 0:
            denom = denom + delta * (bundle.core / km)
    if use_entropy:
        denom = denom + eta * bundle.h_norm
    tau = (tau_base * exp_term / denom).clamp(tau_min, tau_base)
    logger.info(f"τ(i): min={tau.min():.4e} max={tau.max():.4e} mean={tau.mean():.4e}")
    return tau


def build_tau_feature_matrix(bundle) -> torch.Tensor:
    """构造附录B.1所需的四维手工特征 [E, C_deg, core_norm, H_norm]。"""
    core = bundle.core.float()
    core_norm = core / core.max().clamp_min(1.0)
    return torch.stack(
        [
            bundle.spectral_energy.float(),
            bundle.c_deg.float(),
            core_norm,
            bundle.h_norm.float(),
        ],
        dim=1,
    )


def perturb_tau(
    tau: torch.Tensor,
    tau_min: float,
    tau_base: float,
    random_seed: int = 42,
    log_scale: float = 0.5,
) -> torch.Tensor:
    """对已有τ(i)施加有界乘法随机扰动，用于 B2-RND 对照。"""
    rng = torch.Generator()
    rng.manual_seed(int(random_seed))
    delta = torch.empty_like(tau).uniform_(-float(log_scale), float(log_scale), generator=rng)
    perturbed = torch.clamp(tau * torch.exp(delta), min=tau_min, max=tau_base)
    logger.info(
        "B2-RND：对B2路径τ(i)施加随机扰动 | log_scale=%.3f | mean=%.4e",
        log_scale,
        perturbed.mean().item(),
    )
    return perturbed
