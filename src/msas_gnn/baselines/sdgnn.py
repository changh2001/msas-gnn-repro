"""SDGNN 推理骨架。

说明：
- `b0`：当前仓库中的 SDGNN 兼容基线，沿用 MSAS 的分层候选集/Phase-Θ 口径；
- `sdgnn_pure`：更贴近原始论文的训练协议，使用独立的候选集与平坦 LARS/Lasso Phase-Θ。

两者在推理阶段都共享 `H_hat = Θ^{fixed,T} Φ_tilde` 这一骨架，因此复用同一个模型容器。
"""
import logging
import torch, torch.nn as nn
logger = logging.getLogger(__name__)

class SDGNN(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=64, lambda_reg=1e-3, k=50, L=3, **kwargs):
        super().__init__()
        self.lambda_reg = lambda_reg  # 全局统一正则系数（原始 SDGNN 的核心）
        self.k = k; self.L = L
        self.classifier = nn.Linear(hidden_channels, out_channels)
        self.theta_fixed = None

    def set_theta_fixed(self, theta):
        self.theta_fixed = theta
        logger.info(f"[SDGNN] Θ^fixed已固化 shape={theta.shape}")

    def infer(self, phi_tilde):
        """推理：Ĥ = Θ^{fixed,⊤}·Φ̃。口径：O(n·k̄·d)，Φ̃已离线缓存。"""
        if self.theta_fixed is None: raise RuntimeError("Θ^fixed未固化")
        if self.theta_fixed.is_sparse: h = torch.sparse.mm(self.theta_fixed.t(), phi_tilde)
        else: h = self.theta_fixed.t() @ phi_tilde
        return self.classifier(h)

    def forward(self, x, edge_index, phi_tilde=None):
        if phi_tilde is not None and self.theta_fixed is not None: return self.infer(phi_tilde)
        return self.classifier(x)
