# 配置字段说明

| 论文符号 | 代码变量 |
|---------|--------|
| ρ | hop_dim.rho（全局预算缩放系数） |
| \varrho | hop_dim.varrho（跳距几何衰减公比） |
| κ | hop_dim.kappa |
| p_base | hop_dim.p_base |
| p_min | hop_dim.p_min |
| β_τ | beta_tau |
| γ | gamma |
| δ | delta |
| ω_H | omega_h |
| τ_base | tau_base |
| τ_min | tau_min |
| c_E | c_e |
| λ | lambda_reg |
| L | hop_dim.L |
| K_eig | spectral.K_eig |
| T_W | alternating_opt.t_w |
| η_W | alternating_opt.eta_w |

历史别名：

- `e_threshold`：旧版配置中曾直接写入节点维阈值；当前主线按论文公式运行时计算 `E_threshold = c_E * median(E_spectral)`，因此建议统一改用 `c_e`

第6章大图训练工程开关：

- `alternating_opt.batch_mode`：`auto | full_batch | mini_batch`，控制是否在 Phase-W 启用大图 mini-batch 主循环
- `alternating_opt.node_threshold_for_minibatch`：`batch_mode=auto` 时的节点数阈值，默认 `5e4`
- `candidate_sampling.keep_complete_hops`：前多少跳完整保留
- `candidate_sampling.sampled_max_candidates`：后续跳的单层候选采样上限
- `train.batch_size`：mini-batch 目标节点批大小；论文 ogbn-arxiv 口径为 `1024`
