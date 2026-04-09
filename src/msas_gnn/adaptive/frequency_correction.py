"""频率维：等权实例化→E_spectral(i)。对应论文§4.1。"""
import logging
import torch
logger = logging.getLogger(__name__)

def compute_frequency_weights(eigenvalues, h_edge=0.5, mode="equal_weight"):
    """频率接口的权重计算；当前主线仅保留等权实例化。"""
    K = eigenvalues.shape[0]
    if K == 0:
        return torch.tensor([])
    valid = eigenvalues > 1e-8
    if valid.sum().item() == 0:
        return torch.zeros(K, dtype=eigenvalues.dtype)
    if mode != "equal_weight":
        raise ValueError(f"Unknown freq mode: {mode}")
    weights = valid.to(dtype=eigenvalues.dtype)
    return weights / weights.sum().clamp_min(1.0)

def apply_frequency_correction(bundle, mode="equal_weight"):
    from msas_gnn.spectral.spectral_energy import compute_spectral_energy
    weights = compute_frequency_weights(bundle.eigenvalues, bundle.h_edge, mode)
    corrected_energy = compute_spectral_energy(bundle.eigenvalues, bundle.eigenvectors, weights)
    return weights, corrected_energy
