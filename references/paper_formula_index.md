# 论文公式–代码对照索引

## 第3章
| 公式/算法 | 代码文件 | 函数 |
|-----------|---------|------|
| 式(3.1)-(3.3) L̃ | spectral/laplacian.py | compute_normalized_laplacian_scipy |
| 定义3.1 λ_gap | spectral/spectral_gap.py | compute_spectral_gap |
| 定义3.2 E_spectral | spectral/spectral_energy.py | compute_spectral_energy |
| 式(3.4) h_edge | spectral/homophily.py | compute_edge_homophily |
| 式(3.9) H_norm | spectral/entropy.py | compute_local_entropy |
| 式(3.11) C_deg | spectral/centrality.py | compute_degree_centrality |
| 算法3.2 k-core | spectral/kcore.py | compute_kcore |
| 算法3.1 流水线 | spectral/metric_bundle.py | compute_metric_bundle |

## 第4章
| 公式/算法 | 代码文件 | 函数 |
|-----------|---------|------|
| 式(4.1)-(4.3) 频率维 | adaptive/frequency_correction.py | compute_frequency_weights |
| 式(4.7)-(4.13) τ(i) | adaptive/tau_builder.py | build_tau |
| 式(4.14)-(4.18) 跳距维 | adaptive/hop_budget.py | allocate_hop_budget_for_candidates |
| §4.4 三维封装 | adaptive/joint_budget.py | build_adaptive_params |

## 第5章
| 公式/算法 | 代码文件 | 函数 |
|-----------|---------|------|
| 算法5.1 候选集 | decomposition/candidate_builder.py | build_bfs_candidate_sets |
| 算法5.2 LARS | decomposition/lars_solver.py | lars_lasso_single |
| 算法5.3 Phase-Θ | decomposition/theta_optimizer.py | run_phase_theta |
| 算法5.4 交替优化 | training/alternating_opt.py | AlternatingOptimizer.run |
| §5.1.2 W_phi 岭回归热启动 | training/feature_transform.py | initialize_linear_feature_transform |
| §5.3 推理 | decomposition/inference.py | infer_h_hat |

## 第6章
| 指标 | 代码文件 | 函数 |
|------|---------|------|
| ε_approx | evaluation/metrics.py | compute_epsilon_approx |
| Wilcoxon | evaluation/significance.py | run_wilcoxon |
| Q_be | evaluation/break_even.py | compute_break_even |
| 近期补充基线 | baselines/graphsaint.py, baselines/graph_transformers.py | GraphSAINT / NodeFormer / DIFFormer / SGFormer / NAGphormer |
