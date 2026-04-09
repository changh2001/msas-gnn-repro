# MSAS-GNN：谱驱动自适应稀疏图神经网络的鲁棒高效推理

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch 2.0.1](https://img.shields.io/badge/pytorch-2.0.1-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **论文**：MSAS-GNN: Spectrally-Driven Adaptive Sparse GNN for Robust and Efficient Inference  
> **作者**：常昊，北京师范大学应用统计专业  
> **基础框架**：SDGNN（Hu et al., arXiv:2410.19723）

---

## 目录

- [1. 项目简介](#1-项目简介)
- [2. 仓库里有什么](#2-仓库里有什么)
- [3. 第一次拿到仓库应该怎么开始](#3-第一次拿到仓库应该怎么开始)
- [4. 环境与依赖](#4-环境与依赖)
- [5. 数据准备](#5-数据准备)
- [6. 最常用的运行入口](#6-最常用的运行入口)
- [7. 完整复现流程](#7-完整复现流程)
- [8. 实验分组与脚本说明](#8-实验分组与脚本说明)
- [9. 配置系统说明](#9-配置系统说明)
- [10. 代码层次结构](#10-代码层次结构)
- [11. 论文-代码映射表](#11-论文代码映射表)
- [12. 结果产物与目录约定](#12-结果产物与目录约定)
- [13. 文档索引](#13-文档索引)
- [14. 测试与验证](#14-测试与验证)
- [15. 当前默认论文口径](#15-当前默认论文口径)
- [16. 预期主实验结果](#16-预期主实验结果)
- [17. 注意事项与常见问题](#17-注意事项与常见问题)

---

## 1. 项目简介

MSAS-GNN 是一个围绕论文复现组织的研究代码仓库，目标是对 SDGNN 的推理路径做更稳健、更高效的改进。仓库里不仅包含主方法实现，还包含：

- 主实验与对照基线
- 模块消融与跳距策略消融
- 推理效率与摊销分析
- 补充实验脚本、图表生成与导出脚本

MSAS-GNN 针对 SDGNN 三个结构性局限展开改进：

1. **全局统一正则系数 λ1** -> 节点级自适应正则系数 `tau(i)`（第 3、4 章）
2. **静态冻结推理参数** -> `Phase-Theta / Phase-W` 交替优化（第 5 章）
3. **缺乏图结构质量控制** -> 分层跳距预算 `k_i^(l)`（第 4 章 §4.3）

如果你第一次接触这个仓库，可以把它理解成三层：

- `configs/`：论文协议层，定义实验口径
- `scripts/`：实验调度层，负责批量运行与出图出表
- `src/msas_gnn/`：算法实现层，负责真正的数据、训练、分解与评估逻辑

---

## 2. 仓库里有什么

这个仓库既可以当作“论文复现包”使用，也可以当作“算法实现代码库”来阅读。

如果你的目标是：

- **先跑起来**：先看“[3. 第一次拿到仓库应该怎么开始](#3-第一次拿到仓库应该怎么开始)”
- **跑完整论文结果**：先看“[7. 完整复现流程](#7-完整复现流程)”
- **读主方法实现**：先看“[10. 代码层次结构](#10-代码层次结构)”
- **改实验配置**：先看“[9. 配置系统说明](#9-配置系统说明)”

仓库当前覆盖的主要实验板块：

- **主实验**：6 个数据集，含同配图与异配图
- **消融实验**：`B0` 到 `B5`、`B2-RND`、`B5-frozen`
- **跳距策略消融**：`uniform / xi05 / xi10 / reverse`
- **效率实验**：时延、显存、参数量、相对加速比
- **摊销实验**：break-even 查询次数 `Q_be`
- **补充实验脚本**：`xi` 细粒度扫描、补充敏感性、谱代理量验证

---

## 3. 第一次拿到仓库应该怎么开始

推荐按下面顺序进行。

### 路线 A：只想确认仓库能不能跑

```bash
conda env create -f environment.yml
conda activate msas-gnn
pip install -e .
python scripts/setup/verify_env.py
make download-data
make smoke-test-cora
```

这个流程会完成：

- 创建论文环境
- 安装包到当前仓库
- 检查 PyTorch / PyG / OGB / CUDA
- 下载并准备数据
- 跑一个 `Cora + B5` 的冒烟实验

### 路线 B：想快速看懂代码主线

建议按下面顺序阅读：

1. `run.py`
2. `src/msas_gnn/api.py`
3. `src/msas_gnn/config.py`
4. `src/msas_gnn/training/msas_trainer.py`
5. `src/msas_gnn/adaptive/joint_budget.py`
6. `src/msas_gnn/training/alternating_opt.py`
7. `src/msas_gnn/decomposition/`
8. `src/msas_gnn/evaluation/`

### 路线 C：想复现论文结果

建议按这个顺序：

1. 跑环境检查
2. 下载数据并生成划分
3. 预处理图指标和教师表示
4. 跑主实验
5. 跑消融
6. 跑效率
7. 视需要运行补充实验脚本
8. 统一生成图表与导出包

完整命令见“[7. 完整复现流程](#7-完整复现流程)”。

---

## 4. 环境与依赖

推荐环境直接使用仓库内的 [environment.yml](environment.yml)。

核心依赖版本：

- Python `3.10`
- PyTorch `2.0.1`
- PyG `2.3.1`
- OGB `1.3.6`
- CUDA 口径：`pytorch-cuda=11.8`

创建环境：

```bash
conda env create -f environment.yml
conda activate msas-gnn
pip install -e .
```

安装完成后建议立即验证：

```bash
python scripts/setup/verify_env.py
```

该脚本会检查：

- Python 版本
- PyTorch 是否可导入
- CUDA 是否可用
- GPU 型号与显存
- `torch_geometric / ogb / scipy / sklearn / optuna / yaml / networkx` 是否安装完整

说明：

- 没有 GPU 也可以运行仓库中的大部分逻辑，但**效率实验的数字不会与论文完全对齐**
- `ogbn-arxiv` 的训练与效率评测更推荐在 GPU 环境中完成

---

## 5. 数据准备

一键准备数据：

```bash
make download-data
```

这个命令实际做了两件事：

```bash
python scripts/data/download_all.py
python scripts/data/build_splits.py --seeds 42 123 456 789 2021 2022 2023 2024 2025 2026
```

准备完成后，数据相关目录通常包括：

- `data/raw/`：原始下载数据
- `data/splits/`：按论文协议生成的 10 组划分
- `data/cache/`：图指标缓存、教师表示缓存等
- `data/processed/`：部分中间处理产物

当前仓库覆盖的数据集：

- `cora`
- `citeseer`
- `pubmed`
- `ogbn_arxiv`
- `chameleon`
- `squirrel`

其中：

- `cora / citeseer / pubmed`：Planetoid 类数据集
- `ogbn_arxiv`：OGB 大图
- `chameleon / squirrel`：WikipediaNetwork 异配图

---

## 6. 最常用的运行入口

### 6.1 统一入口 `run.py`

适合做单任务试跑。

```bash
python run.py --task smoke --dataset cora --ablation b5
python run.py --task train --dataset cora --ablation b5
python run.py --task ablation --dataset cora --ablation b0
python run.py --task efficiency --dataset cora --methods gcn msas_gnn
python run.py --task visualize --dataset cora --vis_type tsne
```

`run.py` 目前支持的任务：

- `smoke`
- `train`
- `ablation`
- `efficiency`
- `visualize`

### 6.2 Makefile 入口

适合复现整批实验。

```bash
make check-env
make download-data
make smoke-test-cora
make reproduce-main
make reproduce-ablation
make reproduce-efficiency
make reproduce-supplemental  # 补充实验脚本，可选
make reproduce-all           # 包含补充实验脚本
make export
```

### 6.3 直接运行脚本

适合研究过程中局部重跑某个板块。

例如：

```bash
python scripts/experiments/run_main_benchmarks.py --datasets cora citeseer
python scripts/experiments/run_ablation_modular.py --datasets cora
python scripts/experiments/run_efficiency.py --all
python scripts/experiments/run_tsne.py --dataset cora
```

---

## 7. 完整复现流程

仓库已经把论文复现分成三条正文主线和一条可选补充实验线，对应 [Makefile](Makefile) 中的目标。

```bash
make reproduce-main       # 主实验（约 27h，双 GPU 并行）
make reproduce-ablation   # 消融实验 + 表 6.4 / 6.5（约 9h）
make reproduce-efficiency # 效率分析 + 表 6.6 / 6.7（约 1h）
make reproduce-supplemental # 补充实验脚本（可选，约 6.5h）
make reproduce-all          # 全量复现 + 补充实验脚本（约 2-3 天）
make export               # 结果打包
```

### 7.1 主实验 `make reproduce-main`

包括：

1. 图指标预处理
2. 教师训练与 `H*` 缓存
3. 主实验并行执行
4. 显著性检验与图表生成

具体顺序：

```bash
python scripts/preprocess/compute_graph_metrics.py --all
python scripts/preprocess/cache_teacher_reprs.py --all
bash scripts/run_parallel_main.sh
python scripts/experiments/run_significance.py --exp main
python scripts/visualization/build_paper_figures.py --main
```

其中 `scripts/run_parallel_main.sh` 会将任务拆成：

- GPU 0：`cora / citeseer / pubmed`
- GPU 1：`chameleon / squirrel`，然后继续跑 `ogbn_arxiv`

### 7.2 消融 `make reproduce-ablation`

包括两类：

- 模块消融：`B0 -> B5`
- 跳距策略消融：`uniform / xi05 / xi10 / reverse`

### 7.3 效率 `make reproduce-efficiency`

包括：

- 稠密基线与稀疏方法推理时延
- 显存占用
- 参数量统计
- 相对 `GCN` 加速比
- 预处理摊销 break-even 分析

### 7.4 补充实验脚本 `make reproduce-supplemental`

包括：

- `xi` 细粒度扫描
- Chameleon 与 ogbn-arxiv 的补充敏感性
- `sigma proxy` 谱代理量验证

说明：

- 这一批脚本用于额外验证与扩展分析
- 如只关心当前 thesis 正文复现，可不运行这一批

---

## 8. 实验分组与脚本说明

### 8.1 主实验

脚本：

- `scripts/experiments/run_main_benchmarks.py`
- `scripts/experiments/run_heterophily_benchmarks.py`

默认方法：

- `gcn`
- `sgc`
- `pprgo`
- `glnn`
- `b0`
- `b5`

异配图额外包含：

- `geom_gcn`
- `h2gcn`

### 8.2 消融实验

脚本：

- `scripts/experiments/run_ablation_modular.py`
- `scripts/experiments/run_ablation_hop_strategy.py`

模块消融定义：

- `B0`：SDGNN 兼容基线，和 `B1-B5` 共用分层 BFS 候选池、残差级联 `Phase-Θ` 与交替优化主干，但将自适应机制退化为全局统一 `lambda` 与均匀跳距预算
- `B1`：在 `B0` 上加入谱能量驱动的 `tau(i)`
- `B2`：在 `B1` 上加入中心性
- `B3`：在 `B2` 上加入 `k-core`
- `B4`：在 `B3` 上加入局部图熵
- `B5`：在 `B4` 上加入跳距预算，正文主方法
- `B2-RND`：沿 `B2` 路径对 `tau(i)` 施加随机扰动的对照
- `B5-frozen`：冻结 `W`，不执行 `Phase-W`
- `sdgnn_pure`：更贴近原论文的 SDGNN 训练协议，使用 `K-hop + D-hop fanout` 平坦候选池与单次 LARS/Lasso `Phase-Θ`

正文实验中，凡依赖教师表示 `H*` 的分解式方法统一采用两层 `GCN` 作为教师模型；因此
`B0`、`sdgnn_pure` 与 `B5` 的差异只来自稀疏分解与预算机制，而不来自教师骨干网络。其中
`B0` 与 `B5` 共享同一分层求解主干，`sdgnn_pure` 则在候选集构造和 `Phase-Θ` 口径上都单独对齐原始协议。

跳距策略消融定义：

- `uniform`
- `xi05`
- `xi10`
- `reverse`

### 8.3 效率实验

脚本：

- `scripts/experiments/run_efficiency.py`
- `scripts/experiments/run_breakeven.py`

默认对比方法：

- `gcn`
- `sgc`
- `pprgo`
- `glnn`
- `sdgnn`
- `msas_gnn`

### 8.4 补充实验脚本

脚本：

- `scripts/experiments/supplemental/run_xi_sweep.py`
- `scripts/experiments/supplemental/run_sensitivity_supplemental.py`
- `scripts/experiments/supplemental/run_spectral_proxy.py`
- `scripts/experiments/supplemental/backward_design_tau.py`

说明：

- 以上脚本统一放在 `supplemental/` 目录下
- 这一批脚本用于补充验证与扩展分析，不影响正文实验链

### 8.5 可视化与统计

脚本：

- `scripts/experiments/run_significance.py`
- `scripts/experiments/run_sensitivity.py`
- `scripts/experiments/run_tsne.py`
- `scripts/experiments/run_tau_distribution.py`
- `scripts/visualization/build_paper_figures.py`
- `scripts/visualization/build_paper_tables.py`

---

## 9. 配置系统说明

仓库采用“多层 YAML 叠加”的配置方式，最终由 `src/msas_gnn/config.py` 统一合成。

配置装配顺序：

1. `configs/global/`
2. `configs/models/`
3. `configs/datasets/`
4. `configs/ablations/`
5. 命令行传入的 `override_path`
6. 运行时 `overrides`

也就是说，一次实验的最终配置通常不是单个 YAML 文件，而是上述几层合并后的结果。

### 9.1 目录说明

- `configs/global/`：全局默认值、路径、运行时参数、种子
- `configs/models/`：主模型模板，如 `msas_gnn_b5.yaml`
- `configs/datasets/`：数据集级配置
- `configs/ablations/`：消融开关和局部覆盖
- `configs/teachers/`：各基线或教师模型配置
- `configs/profiling/`：效率与摊销口径
- `configs/sweeps/`：敏感性和搜索空间

### 9.2 常见关键字段

常见论文符号与代码字段映射：

| 论文符号 | 代码变量 |
|---------|--------|
| `xi` | `xi_budget` |
| `beta_tau` | `beta_tau` |
| `gamma` | `gamma` |
| `delta` | `delta` |
| `eta` | `eta` |
| `tau_base` | `tau_base` |
| `tau_min` | `tau_min` |
| `c_E` | `c_e` |
| `lambda` | `lambda_reg` |
| `k` | `lars.k` |
| `L` | `hop_dim.L` |
| `K_eig` | `spectral.K_eig` |
| `T_W` | `alternating_opt.t_w` |
| `eta_W` | `alternating_opt.eta_w` |

更多字段说明见 [docs/config_schema.md](docs/config_schema.md)。

### 9.3 一次典型配置是怎样被加载的

以 `dataset=cora, ablation_id=b5` 为例，最终会组合：

- `configs/global/*.yaml`
- `configs/models/msas_gnn_b5.yaml`
- `configs/datasets/cora.yaml`
- `configs/ablations/b5_full_main.yaml`

---

## 10. 代码层次结构

推荐把代码层次理解为“从外到内”的五层。

### 10.1 最外层：实验和任务入口

- `run.py`
- `scripts/`
- `Makefile`

作用：

- 接收命令行参数
- 组织实验批次
- 调度预处理、训练、评估、出图出表

### 10.2 配置层

- `configs/`
- `src/msas_gnn/config.py`

作用：

- 把论文口径转成可执行配置
- 统一不同数据集、模型模板和消融版本

### 10.3 数据与图指标层

- `src/msas_gnn/data/`
- `src/msas_gnn/spectral/`

作用：

- 加载 Planetoid / OGB / WikipediaNetwork 数据
- 构造训练/验证/测试划分
- 计算拉普拉斯、Lanczos 特征对、谱能量、图熵、中心性、`k-core`、同配率

### 10.4 主方法实现层

- `src/msas_gnn/adaptive/`
- `src/msas_gnn/decomposition/`
- `src/msas_gnn/training/`

作用：

- 构建三维自适应参数：`tau`、`k_budget`、`freq_weights`
- 完成候选构造、LARS 稀疏分解与推理近似
- 执行 `Phase-Theta / Phase-W` 交替优化

### 10.5 评估与产物层

- `src/msas_gnn/evaluation/`
- `scripts/visualization/`
- `outputs/`

作用：

- 计算准确率、近似误差、显著性、速度与显存
- 生成图、表与导出包

### 10.6 第一次读代码建议从哪里开始

如果你想快速串起一条完整主线，建议按下面顺序读：

1. `run.py`
2. `src/msas_gnn/api.py`
3. `src/msas_gnn/config.py`
4. `src/msas_gnn/training/msas_trainer.py`
5. `src/msas_gnn/training/data_utils.py`
6. `src/msas_gnn/spectral/metric_bundle.py`
7. `src/msas_gnn/adaptive/joint_budget.py`
8. `src/msas_gnn/training/alternating_opt.py`
9. `src/msas_gnn/decomposition/theta_optimizer.py`
10. `src/msas_gnn/evaluation/ablation_runner.py`

---

## 11. 论文-代码映射表

| 论文章节 | 核心内容 | 对应代码路径 |
|---------|---------|------------|
| 第 3 章 §3.1 | 拉普拉斯 + Lanczos | `src/msas_gnn/spectral/laplacian.py`, `src/msas_gnn/spectral/lanczos.py` |
| 第 3 章 §3.2 | 四类图复杂度指标 | `src/msas_gnn/spectral/` |
| 第 4 章 §4.1 | 频率维（等权谱能量） | `src/msas_gnn/adaptive/frequency_correction.py` |
| 第 4 章 §4.2 | 节点维 `tau(i)` | `src/msas_gnn/adaptive/tau_builder.py` |
| 第 4 章 §4.3 | 跳距维 `k_i^(l)` | `src/msas_gnn/adaptive/hop_budget.py` |
| 第 4 章 §4.4 | 三维参数统一封装 | `src/msas_gnn/adaptive/joint_budget.py` |
| 第 5 章 §5.1 | LARS 稀疏分解 | `src/msas_gnn/decomposition/lars_solver.py`, `theta_optimizer.py` |
| 第 5 章 §5.2 | 交替优化 | `src/msas_gnn/training/alternating_opt.py` |
| 第 6 章 §6.2 | 主实验 | `scripts/experiments/run_main_benchmarks.py` |
| 第 6 章 §6.3 | 消融实验 | `scripts/experiments/run_ablation_modular.py` |
| 第 6 章 §6.4 | 效率分析 | `scripts/experiments/run_efficiency.py` |
| 第 6 章 实验设置 | 大图 mini-batch 交替优化协议 | `src/msas_gnn/training/alternating_opt.py` |
| 补充实验脚本 | `xi` 扫描、补充敏感性、谱代理量验证 | `scripts/experiments/supplemental/` |
| 第 6 章图表与补充实验图表 | LaTeX 表格与 PDF 图 | `scripts/visualization/build_paper_tables.py`, `build_paper_figures.py` |

---

## 12. 结果产物与目录约定

运行过程中会自动生成以下产物：

- `outputs/results/`：实验原始 JSON 结果
- `outputs/figures/`：正文图，如敏感性、`tau(i)` 分布、t-SNE
- `outputs/figures/supplemental/`：补充实验输出图
- `outputs/tables/`：正文 LaTeX 表格，覆盖表 6.2-6.7
- `outputs/tables/supplemental/`：补充实验输出表
- `outputs/logs/`：日志
- `outputs/checkpoints/`：模型或中间 checkpoint

常见图表生成入口：

```bash
python scripts/visualization/build_paper_figures.py --main
python scripts/visualization/build_paper_figures.py --ablation
python scripts/visualization/build_paper_figures.py --efficiency
python scripts/visualization/build_paper_figures.py --supplemental
python scripts/visualization/build_paper_figures.py --all
```

导出复现包：

```bash
make export
```

---

## 13. 文档索引

仓库中还有几份补充文档，建议按用途查阅：

- [docs/quickstart.md](docs/quickstart.md)：最短上手命令
- [docs/config_schema.md](docs/config_schema.md)：配置字段说明
- [docs/ablation_guide.md](docs/ablation_guide.md)：消融逻辑说明
- [docs/efficiency_measurement.md](docs/efficiency_measurement.md)：效率口径说明
- [docs/reproduce_checklist.md](docs/reproduce_checklist.md)：完整复现检查表

如果你只是第一次接手仓库，优先顺序建议是：

1. 本 README
2. `docs/quickstart.md`
3. `docs/reproduce_checklist.md`
4. `docs/config_schema.md`

---

## 14. 测试与验证

运行测试：

```bash
make test
make test-integration
make test-all
```

对应内容：

- `tests/unit/`：算法与工具模块单元测试
- `tests/integration/`：主训练/消融流水线集成测试
- `tests/regression/`：数据集指标与协议回归测试

环境验证：

```bash
python scripts/setup/verify_env.py
```

建议最小验证顺序：

1. `python scripts/setup/verify_env.py`
2. `make download-data`
3. `make smoke-test-cora`
4. `make test`

---

## 15. 当前默认论文口径

- `Cora` 主方法默认配置中，`B5 / B0` 使用 `lr=0.005`、`dropout=0.3`
- 正文消融表同时报告两条 SDGNN 口径：`sdgnn_pure` 为原始协议基线，`B0` 为与 `B1-B5` 共用分层求解主干的兼容基线
- 正文实验中所有依赖教师表示 `H*` 的分解式方法统一采用两层 `GCN` 教师模型
- `GCN` 教师默认配置为 `hidden_dim=128`、`lr=0.005`、`dropout=0.3`
- 正文敏感性分析默认扫描 `tau_base / k / xi_budget`
- 仓库保留 `supplemental_sigma_proxy`、`supplemental_xi_sweep` 等补充实验脚本，便于扩展验证与额外分析
- 大图 `ogbn-arxiv` 默认按第 6 章实验设置口径使用 mini-batch 交替优化

---

## 16. 预期主实验结果

论文表 6.2 / 6.3 的目标结果如下。

| 方法 | Cora | Citeseer | PubMed | ogbn-arxiv |
|------|------|----------|--------|------------|
| SDGNN-compatible (B0) | 86.6±0.9 | 80.3±1.1 | 88.7±0.5 | 74.27±0.21 |
| **MSAS-GNN (B5)** | **88.3±0.7** | **82.1±0.9** | **89.4±0.4** | **75.13±0.23** |
| p 值 | 0.002 | 0.001 | 0.048 | 0.009 |

| 方法 | Chameleon | Squirrel |
|------|-----------|----------|
| SDGNN-compatible (B0) | 63.5±1.1 | 54.2±1.4 |
| **MSAS-GNN (B5)** | **67.2±0.9** | **56.9±1.2** |

复现完成后，可参考 [docs/reproduce_checklist.md](docs/reproduce_checklist.md) 做逐项核对。

---

## 17. 注意事项与常见问题

### 17.1 Chameleon / Squirrel 为什么不能直接和官方划分文献对比

因为本仓库采用的是 **60 / 20 / 20 重划分**，而不是所有文献都使用的固定官方划分。结果可以用于本仓库内部比较，但不应直接和另一套划分口径的文献数字横向对比。

### 17.2 SDGNN 基线为什么是仓库内兼容实现

因为该方法没有直接可用的官方代码，本仓库同时保留了两条口径：`B0` 作为兼容基线，`sdgnn_pure` 作为更贴近原论文的复现入口。为保证正文实验可比性，这两条口径当前都统一采用 `GCN` 教师模型。相关说明见 `references/sdgnn_impl_notes.md`。
其中，`B0` 与 `B1-B5` 共用分层求解主干，适合作为正文消融链的内部基线；`sdgnn_pure` 则在候选集与 `Phase-Θ` 口径上单独贴近原始 SDGNN。

### 17.3 为什么效率实验和论文数字不完全一致

常见原因包括：

- 没有使用 GPU
- CUDA / PyTorch / PyG 版本不同
- 没按论文口径区分“小图全图前向”和“ogbn-arxiv 按 batch 测量”
- 热身次数、重复次数、batch size 不一致

### 17.4 参数量是怎么统计的

- 稠密基线：按模型参数总量统计
- `SDGNN / MSAS-GNN`：按固化稀疏权重 `Theta_fixed` 的非零项加上线性头统计

### 17.5 收敛能否严格保证

不能。交替优化本质上是非凸问题，仓库实现追求的是工程上的稳定训练和论文协议复现，而不是严格全局收敛证明。

### 17.6 大图为什么有单独训练口径

`ogbn-arxiv` 节点规模较大，因此按第 6 章实验设置使用 mini-batch 交替优化。训练结束后，代码会固定最终 `Phi_tilde`，再补做一次全图 `Phase-Theta` 重对齐。

### 17.7 想快速确认当前结果有没有跑偏，应该看什么

优先看：

1. `outputs/results/` 中最新 JSON 是否完整
2. `docs/reproduce_checklist.md` 中主实验目标值
3. `run_significance.py` 的配对种子结果
4. `tau` 分布与 t-SNE 可视化是否大体符合预期

---

## 补充说明

这个 README 现在更偏向“第一次接手仓库时的总入口”。如果你已经明确只关心某个方向，可以直接跳转：

- 只想跑通：看“[3. 第一次拿到仓库应该怎么开始](#3-第一次拿到仓库应该怎么开始)”
- 只想复现实验：看“[7. 完整复现流程](#7-完整复现流程)”
- 只想改配置：看“[9. 配置系统说明](#9-配置系统说明)”
- 只想读代码：看“[10. 代码层次结构](#10-代码层次结构)”
