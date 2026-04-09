"""热启动：Φ̃^(0) = ℓ2-row-normalize(H*)。对应论文§5.1.2。
动机：从教师表示H*出发初始化Φ̃，加速交替优化收敛。
"""
import logging
import torch.nn.functional as F
logger = logging.getLogger(__name__)

def warm_start_phi_tilde(h_star):
    """Φ̃^(0) = ℓ2-row-normalize(H*)。零行归一化后仍为零。"""
    phi = F.normalize(h_star, p=2, dim=1)
    logger.info(f"热启动初始化完成 shape={phi.shape}")
    return phi
