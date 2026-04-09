# MSAS-GNN：谱驱动自适应稀疏图神经网络的鲁棒高效推理

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch 2.0.1](https://img.shields.io/badge/pytorch-2.0.1-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **论文**：MSAS-GNN: Spectrally-Driven Adaptive Sparse GNN for Robust and Efficient Inference  
> **作者**：常昊，北京师范大学应用统计专业  
> **基础框架**：SDGNN（Hu et al., arXiv:2410.19723）

---

## 项目简介

MSAS-GNN 针对 SDGNN 三个结构性局限展开改进：

1. **全局统一正则系数 λ₁** → 节点级自适应正则系数 τ(i)（第3、4章）
2. **静态冻结推理参数** → Phase-Θ / Phase-W 交替优化（第5章）
3. **缺乏图结构质量控制** → 分层跳距预算 k_i^(l)（第4章 §4.3）

---

## 论文–代码映射表

| 论文章节 | 核心内容 | 对应代码路径 |
|---------|---------|------------|
| 第3章 §3.1 | 拉普拉斯 + Lanczos | `src/msas_gnn/spectral/laplacian.py`, `lanczos.py` |
| 第3章 §3.2 | 四类图复杂度指标 | `src/msas_gnn/spectral/` |
| 第4章 §4.1 | 频率维（等权谱能量） | `adaptive/frequency_correction.py` |
| 第4章 §4.2 | 节点维 τ(i) | `adaptive/tau_builder.py` |
| 第4章 §4.3 | 跳距维 k_i^(l) | `adaptive/hop_budget.py` |
| 第5章 §5.1 | LARS 稀疏分解 | `decomposition/lars_solver.py`, `theta_optimizer.py` |
| 第5章 §5.2 | 交替优化 | `training/alternating_opt.py` |
| 第6章 §6.2 | 主实验 | `scripts/experiments/run_main_benchmarks.py` |
| 第6章 §6.3 | 消融实验 | `scripts/experiments/run_ablation_modular.py` |
| 第6章 §6.4 | 效率分析 | `scripts/experiments/run_efficiency.py` |
| 第6章 实验设置 | ogbn-arxiv 大图 mini-batch 训练口径 | `training/alternating_opt.py` |
| 附录 C.3 | 补充实验 | `scripts/experiments/appendix/` |
| 第6章/附录C 图表 | LaTeX 表格与 PDF 图 | `scripts/visualization/build_paper_tables.py`, `build_paper_figures.py` |

---

## 快速开始

```bash
# 1. 创建环境
conda env create -f environment.yml
conda activate msas-gnn
pip install -e .

# 2. 验证环境
python scripts/setup/verify_env.py

# 3. 下载数据
make download-data

# 4. Cora 冒烟测试（约5分钟）
make smoke-test-cora
```

---

## 完整复现

```bash
make reproduce-main       # 主实验（≈27h，双GPU并行）
make reproduce-ablation   # 消融实验 + 表6.4/6.5（≈9h）
make reproduce-efficiency # 效率分析 + 表6.6/6.7（≈1h）
make reproduce-appendix   # 附录实验 + 附录表/图（≈6.5h）
make reproduce-all        # 全量复现 + 全部表图（约2–3天）
make export               # 结果打包
```

---

## 自动生成产物

- `outputs/results/`：实验原始 JSON 结果
- `outputs/figures/`：正文图（如敏感性、`τ(i)` 分布、t-SNE）
- `outputs/figures/appendix/`：附录图（如 `ξ` 扫描、谱代理量曲线）
- `outputs/tables/`：正文 LaTeX 表格（表 6.2–6.7）
- `outputs/tables/appendix/`：附录 LaTeX 表格（当前覆盖 `tab:appC-spectral`）

生成入口：

```bash
python scripts/visualization/build_paper_figures.py --main
python scripts/visualization/build_paper_figures.py --ablation
python scripts/visualization/build_paper_figures.py --efficiency
python scripts/visualization/build_paper_figures.py --appendix
python scripts/visualization/build_paper_figures.py --all
```

---

## 当前默认论文口径

- Cora 主方法默认配置已对齐附录 C.1：`B5/B0` 使用 `lr=0.005`、`dropout=0.3`
- 正文实验中所有依赖教师表示 `H*` 的分解式方法（`SDGNN-compatible`、`sdgnn_pure`、`MSAS-GNN`）统一采用两层 `GCN` 教师模型
- 正文消融表中同时区分两条 SDGNN 口径：`sdgnn_pure` 表示更贴近原论文的原始协议基线，`B0` 表示与 `B1-B5` 共用分层求解主干的兼容基线
- `GCN` 教师默认配置为 `hidden_dim=128`、`lr=0.005`、`dropout=0.3`
- 正文敏感性分析默认扫描 `τ_base / k / ξ`
- 附录谱代理量实验默认按 10 个随机种子聚合，结果可直接生成 `tab:appC-spectral`

---

## 预期主实验结果（论文表6.2/6.3）

| 方法 | Cora | Citeseer | PubMed | ogbn-arxiv |
|------|------|----------|--------|------------|
| SDGNN-compatible (B0) | 86.6±0.9 | 80.3±1.1 | 88.7±0.5 | 74.27±0.21 |
| **MSAS-GNN (B5)** | **88.3±0.7** | **82.1±0.9** | **89.4±0.4** | **75.13±0.23** |
| p 值 | 0.002 | 0.001 | 0.048 | 0.009 |

| 方法 | Chameleon | Squirrel |
|------|-----------|----------|
| SDGNN-compatible (B0) | 63.5±1.1 | 54.2±1.4 |
| **MSAS-GNN (B5)** | **67.2±0.9** | **56.9±1.2** |

---

## 注意事项

1. **Chameleon/Squirrel** 使用 60/20/20 重划分，**不可**与官方固定划分文献直接比较
2. **SDGNN 口径**：`B0` 为兼容基线，它与 `B1-B5` 共用分层 BFS 候选集、残差级联 `Phase-Θ` 和交替优化主干，只是将自适应机制退化为全局统一 `lambda` 与均匀跳距预算；若需更贴近原论文的实现，请使用 `--ablation sdgnn_pure`，其采用 `K-hop + D-hop fanout` 平坦候选池。两者在当前默认实验中都统一采用 `GCN` 教师模型（见 `references/sdgnn_impl_notes.md`）
3. **推理口径**：小图 ms/次全图前向，ogbn-arxiv ms/batch（bs=1024），不得混用
4. **参数量口径**：稠密基线按模型参数数统计；SDGNN/MSAS-GNN 按固化稀疏权重 `Θ^{fixed}` 非零项加线性头统计
5. **收敛性**：交替优化为非凸问题，代码仅保证训练稳定性
6. **大图训练口径**：ogbn-arxiv 按第6章实验设置使用 mini-batch 交替优化，训练结束后会补做一次全图 Phase-Θ 重对齐
