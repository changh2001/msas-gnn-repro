"""特征变换 phi(X; W_phi)。

对应论文第3章/第5章中的线性特征变换参数 W_phi，
并提供一个基于岭回归的工程化初始化，用于后续 Phase-W 的热启动。
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class LinearFeatureTransform(nn.Module):
    """线性特征变换 phi(X; W_phi) = X W_phi。"""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def compute_phi_tilde(transform: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """按论文口径计算 Phi_tilde = l2-row-normalize(phi(X; W_phi))."""

    return F.normalize(transform(x), p=2, dim=1)


def initialize_linear_feature_transform(
    x: torch.Tensor,
    h_star: torch.Tensor,
    ridge: float = 1e-4,
    device: str | torch.device = "cpu",
    exact_threshold: int = 512,
) -> LinearFeatureTransform:
    """用闭式岭回归近似初始化 W_phi。

    该初始化只用于生成 W_phi^* 的工程近似，真正的论文主线优化由 Phase-W 完成。
    """

    x_cpu = x.detach().to(dtype=torch.float64, device="cpu")
    h_cpu = h_star.detach().to(dtype=torch.float64, device="cpu")
    in_dim = int(x_cpu.shape[1])
    out_dim = int(h_cpu.shape[1])

    if min(int(x_cpu.shape[0]), in_dim) <= int(exact_threshold):
        if in_dim <= int(x_cpu.shape[0]):
            gram = x_cpu.T @ x_cpu
            gram = gram + float(ridge) * torch.eye(in_dim, dtype=torch.float64)
            rhs = x_cpu.T @ h_cpu
            weight = torch.linalg.solve(gram, rhs)
        else:
            gram = x_cpu @ x_cpu.T
            gram = gram + float(ridge) * torch.eye(int(x_cpu.shape[0]), dtype=torch.float64)
            dual = torch.linalg.solve(gram, h_cpu)
            weight = x_cpu.T @ dual
        init_mode = "exact_ridge"
    else:
        denom = x_cpu.pow(2).sum(dim=0, keepdim=True).T.clamp_min(float(ridge))
        weight = (x_cpu.T @ h_cpu) / denom
        init_mode = "diag_ridge_approx"
    weight = weight.to(dtype=torch.float32)

    transform = LinearFeatureTransform(in_dim=in_dim, out_dim=out_dim)
    with torch.no_grad():
        transform.linear.weight.copy_(weight.T.contiguous())
    transform = transform.to(device)
    logger.info(
        "W_phi 初始化完成 mode=%s in_dim=%s out_dim=%s ridge=%.2e",
        init_mode,
        in_dim,
        out_dim,
        ridge,
    )
    return transform
