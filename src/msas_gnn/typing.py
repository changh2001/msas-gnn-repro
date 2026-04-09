"""全局类型别名。"""
from typing import NamedTuple, Tuple, List
import torch
from torch import Tensor

class MetricBundle(NamedTuple):
    """第3章图复杂度指标包。
    spectral_energy: [n] 节点谱能量 E_spectral(i)
    h_norm: [n] 归一化局部图熵 H_norm(i)
    c_deg: [n] 度中心性 C_deg(i)
    core: [n] k-core指数
    h_edge: float 描述性边一致性统计（正文主线不使用）
    eigenvalues: [K_eig] 升序特征值，eigenvalues[1]=λ_gap
    eigenvectors: [n, K_eig] 特征向量矩阵
    """
    spectral_energy: Tensor
    h_norm: Tensor
    c_deg: Tensor
    core: Tensor
    h_edge: float
    eigenvalues: Tensor
    eigenvectors: Tensor

class AdaptiveParamSet(NamedTuple):
    """第4章三维自适应参数集合。"""
    tau: Tensor          # [n] 节点级正则系数 τ(i)
    k_budget: Tensor     # [n, L] 分层跳距预算矩阵（long）
    freq_weights: Tensor # [K_eig] 频率维权重

class ThetaFixed(NamedTuple):
    """固化稀疏权重矩阵 Θ^fixed。"""
    theta: Tensor           # torch.sparse_csr, [n,n]
    k_bar: float            # 平均非零数
    sparsity: float         # 候选集剪枝率（论文表6.4口径）
    support_total: int = 0  # 全图保留的非零支撑数
    candidate_total: int = 0  # 全图候选池总规模

EigenPair = Tuple[Tensor, Tensor]  # (eigenvalues[K], eigenvectors[n,K])
BudgetMatrix = Tensor              # [n, L] long
