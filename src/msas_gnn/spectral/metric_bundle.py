"""统一预处理流水线→MetricBundle。对应论文§3.2算法3.1。"""
import logging, os, time
import torch
from msas_gnn.typing import MetricBundle
logger = logging.getLogger(__name__)

def compute_metric_bundle(
    data,
    K_eig=50,
    cache_path=None,
    force_recompute=False,
    observed_mask=None,
):
    """计算全套图复杂度指标。复杂度O(m·K_eig)（Lanczos主项）。
    步骤：归一化拉普拉斯→Lanczos→谱能量→局部图熵→度中心性→k-core→描述性边一致性统计
    """
    if cache_path and os.path.exists(cache_path) and not force_recompute:
        cached = torch.load(cache_path, map_location="cpu")
        expected_k = min(K_eig, max(data.num_nodes - 2, 0))
        cached_k = int(cached.get("eigenvalues", torch.empty(0)).shape[0])
        if expected_k == 0 or cached_k == expected_k:
            logger.info(f"加载缓存：{cache_path}")
            return MetricBundle(**cached)
        logger.info(
            "缓存 %s 的 K_eig=%s 与当前请求 K_eig=%s 不一致，重新计算",
            cache_path,
            cached_k,
            expected_k,
        )
    n = data.num_nodes; t0 = time.time()
    logger.info(f"计算图指标 n={n} K_eig={K_eig}...")
    from msas_gnn.spectral.laplacian import compute_normalized_laplacian_scipy
    from msas_gnn.spectral.lanczos import lanczos_eigenpairs
    from msas_gnn.spectral.spectral_energy import compute_spectral_energy
    from msas_gnn.spectral.entropy import compute_local_entropy
    from msas_gnn.spectral.centrality import compute_degree_centrality
    from msas_gnn.spectral.kcore import compute_kcore
    from msas_gnn.spectral.homophily import compute_edge_homophily
    L = compute_normalized_laplacian_scipy(data)
    evals, evecs = lanczos_eigenpairs(L, K_eig=min(K_eig, n-2))
    logger.info(f"Lanczos完成 λ_gap={evals[1].item():.6f}")
    E = compute_spectral_energy(evals, evecs)
    H = compute_local_entropy(data); C = compute_degree_centrality(data)
    core = compute_kcore(data)
    he = compute_edge_homophily(data, observed_mask=observed_mask)
    label_scope = "train-observed" if observed_mask is not None else "descriptive-only"
    logger.info(f"指标计算完成 {time.time()-t0:.1f}s | descriptive_h_edge={he:.4f} | scope={label_scope}")
    bundle = MetricBundle(spectral_energy=E, h_norm=H, c_deg=C, core=core, h_edge=he,
                          eigenvalues=evals, eigenvectors=evecs)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        torch.save(bundle._asdict(), cache_path)
    return bundle
