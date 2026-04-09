"""谱间隙 λ_gap = eigenvalues[1]（第二小特征值）。对应论文§3.2.1定义3.1。"""
import logging
logger = logging.getLogger(__name__)
def compute_spectral_gap(eigenvalues):
    if eigenvalues.shape[0] < 2: logger.warning("K_eig<2，返回0"); return 0.0
    lg = float(eigenvalues[1].item())
    if lg < 1e-8: logger.warning(f"λ_gap={lg:.2e}≈0，图可能不连通")
    return lg
