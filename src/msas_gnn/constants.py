"""全局常量定义。"""
DATASET_NAMES = ["cora","citeseer","pubmed","ogbn_arxiv","chameleon","squirrel"]
ABLATION_IDS = ["b0","sdgnn_pure","b1","b2","b3","b4","b5_shared","b5_frozen","b5","b2_rnd"]
ABLATION_NAMES = {
    "b0":"SDGNN-compatible (B0)","sdgnn_pure":"SDGNN-pure",
    "b1":"B1 (+谱能量)","b2":"B2 (+度中心性)",
    "b3":"B3 (+k-core)","b4":"B4 (+局部图熵)","b5_shared":"B5-shared",
    "b5_frozen":"B5-frozen","b5":"MSAS-GNN (B5-full)","b2_rnd":"B2-RND",
}
HOP_STRATEGIES = ["uniform","near_engineering","spectral_gap_reference","reverse"]
DEFAULT_SEEDS = [42,123,456,789,2021,2022,2023,2024,2025,2026]
LATENCY_PROTOCOLS = ["sparse","paper","end_to_end"]
CONFIG_SYMBOL_MAP = {
    "rho":"ρ","varrho":"ϱ","kappa":"κ","p_base":"p_base","p_min":"p_min","beta_tau":"β_τ","gamma":"γ","delta":"δ","omega_h":"ω_H",
    "tau_base":"τ_base","tau_min":"τ_min","lambda_reg":"λ","k":"k","L":"L",
}
