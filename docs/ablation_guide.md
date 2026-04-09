# 消融实验指南

## 快速运行

```bash
python scripts/experiments/run_ablation_modular.py --datasets cora --ablations b0 b1 b2 b3 b4 b5 b2_rnd b5_frozen
python scripts/experiments/run_ablation_hop_strategy.py --datasets cora chameleon
```

## B0~B5 累加逻辑

- B0: SDGNN 兼容基线（全局统一 λ，但仍沿用本仓库分层候选/Phase-Θ 口径）
- B1: +谱能量驱动 τ(i)
- B2: +度中心性
- B3: +k-core
- B4: +局部图熵
- B5: +跳距预算（正文主方法）
- B2-RND: 沿 B2 路径对 τ(i) 做随机扰动的对照
- B5-frozen: 冻结 W，不执行 Phase-W

正文实验中，凡依赖教师表示 `H*` 的分解式方法（`B0`~`B5`、`B2-RND` 以及 `sdgnn_pure`）
统一采用两层 `GCN` 作为教师模型，以保持可比性。

## 原始 SDGNN

- 如需更贴近原论文的复现实验，请使用 `sdgnn_pure`
- 该入口采用 `K-hop` 全保留 + `D-hop` fanout 采样候选集，以及平坦候选池上的单次 LARS/Lasso `Phase-Θ`
- 当前默认仍统一挂接 `GCN` 教师；若需研究教师骨干差异，可再单独覆写 `teacher`
