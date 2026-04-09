"""推理接口：Ĥ = Θ^{fixed,⊤}·Φ̃。对应论文§5.3，复杂度O(n·k̄·d)。
口径警告：本函数=热路径诊断口径（Φ̃已离线缓存，只计矩阵乘）。
论文表6.6使用evaluation/efficiency.py::infer_latency_paper_protocol()。
"""
import torch
from msas_gnn.typing import ThetaFixed

def infer_h_hat(theta_fixed: ThetaFixed, phi_tilde):
    """Ĥ = Θ^{fixed,⊤}·Φ̃。theta_fixed.theta可为sparse或dense。"""
    theta = theta_fixed.theta
    if theta.device != phi_tilde.device:
        theta = theta.to(phi_tilde.device)
    if theta.is_sparse: return torch.sparse.mm(theta.t(), phi_tilde)
    else: return theta.t() @ phi_tilde
