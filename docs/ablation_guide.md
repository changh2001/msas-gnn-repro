# 消融实验指南

## 快速运行

```bash
python scripts/experiments/run_ablation_modular.py --datasets cora --ablations b0 b1 b2 b3 b4 b5_shared b5_frozen b5 b2_rnd
python scripts/experiments/run_ablation_hop_strategy.py --datasets cora chameleon --compute_sigma_proxy
```

## B0~B5 累加逻辑

- SDGNN: 原始协议纯基线（`K-hop + D-hop fanout` 平坦候选池）
- B0: SDGNN 兼容基线（分层 BFS 候选池 + 全局统一 λ + 单个平坦 LARS/Lasso）
- B1: +谱能量驱动 τ(i)
- B2: +度中心性
- B3: +k-core
- B4: +局部图熵
- B5-shared: B4 + 跳距预算 + 共享目标 LARS + 冻结 W
- B5-frozen: B4 + 跳距预算 + 残差级联 LARS + 冻结 W
- B5: +跳距预算 + 残差级联 LARS + 完整交替优化（正文主方法）
- B2-RND: 沿 B2 路径对 τ(i) 做随机扰动的对照

正文实验中，凡依赖教师表示 `H*` 的分解式方法（`B0`~`B5`、`B2-RND` 以及 `sdgnn_pure`）
统一采用三层 `GCN` 作为教师模型，以保持可比性。

## 原始 SDGNN

- 如需更贴近原论文的复现实验，请使用 `sdgnn_pure`
- 该入口采用 `K-hop` 全保留 + `D-hop` fanout 采样候选集，以及平坦候选池上的单次 LARS/Lasso `Phase-Θ`
- 当前默认仍统一挂接 `GCN` 教师；若需研究教师骨干差异，可再单独覆写 `teacher`
