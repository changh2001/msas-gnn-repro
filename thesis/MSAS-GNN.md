> 注：本 Markdown 文件为阶段性导出稿，当前论文正文与实验口径以 `thesis/data/*.tex` 为准；若与之不一致，应以 `thesis/data/*.tex` 中的最新表述为准。

## 摘要

图神经网络以消息传递为核心运算范式，通过在图拓扑引导下递归聚合多跳邻域信息，在节点分类、链接预测等任务上表现突出，已成为处理图结构数据的主流方法之一。当图神经网络从学术评测环境迁移至推荐系统、实时风控等对响应时延有严格约束的在线场景时，多层消息传递引发的感受野指数扩张使推理效率瓶颈成为规模化部署的主要障碍。与此同时，真实图数据中普遍存在的噪声边与缺失边等结构质量缺陷，会通过固定邻接矩阵的训练范式直接影响节点表示，进而损害下游任务的可靠性。

稀疏分解图神经网络（SDGNN）将推理过程转化为对教师模型输出的单次稀疏矩阵乘法，使推理主项复杂度压缩至线性量级，在实测中实现了显著加速。然而，SDGNN 存在核心结构性局限：全局统一正则系数对所有节点施加相同稀疏约束，忽视节点拓扑差异，在同配图（高度节点邻居被过度截断）与异配图（高频判别信息遭到抑制）上均产生系统性偏差，使方法的精度表现与适用范围受到制约。

针对上述局限，本文以 SDGNN 为基础，提出多尺度自适应稀疏图神经网络（MSAS-GNN），以"在保持线性推理复杂度优势的前提下提升节点自适应性"为核心目标，从问题形式化与指标提取、三维自适应机制设计、算法整合与误差分析三个层次依次推进。

方法层面，本文首先经谱计算提取四类图复杂度指标——谱间隙、节点谱能量、归一化局部图熵与节点中心性，为参数设计提供量化依据。进而沿频率、节点与跳距三个维度构建节点级自适应参数：频率维度依据同配统计量对节点谱能量进行异配感知修正；节点维度依据谱能量与邻居有效信息量的相关关系，设计节点级自适应正则系数，使谱能量较高的节点获得更充裕的邻居保留预算；跳距维度依据多跳传播信息逐层衰减的规律，设计近邻优先的分层预算分配策略，并在均匀分配与几何衰减分配方案中实现自适应选取。三维参数体系在推理阶段仍保持单次稀疏矩阵乘法形式，推理主项复杂度维持线性量级。

实验层面，本文在 Cora、Citeseer、PubMed、ogbn-arxiv、Chameleon、Squirrel 六个静态节点分类基准数据集上对 MSAS-GNN 进行系统验证；主实验及用于显著性检验的结果在十个随机种子下重复，并以 Wilcoxon 符号秩检验评估差异显著性。MSAS-GNN 在全部六个数据集上均优于基线 SDGNN：同配性三个引文网络数据集平均准确率提升约 1.4 个百分点；大规模图 ogbn-arxiv 上准确率达 75.13%，高于 SDGNN 的 74.27%；异配性图上的提升幅度约为同配性图的 2.3 倍，Chameleon 与 Squirrel 分别达到 67.2% 与 56.9%。推理效率方面，MSAS-GNN 相对标准消息传递基线在大规模图上实现约 13.6 倍加速，四数据集平均加速比约 8.0 倍，显存占用减少约 74.8%。消融实验表明，节点谱能量为三维参数体系中的主导因子，近邻优先的跳距预算分配策略相比均匀分配在精度与近似误差两项指标上均具有优势，与方法设计预期方向一致。

本文以谱图分析为工具，构建了由可计算图复杂度指标驱动的节点级差异化稀疏分配方法，在保持线性推理复杂度的前提下提升了模型对节点拓扑异质性与图类型差异的适应能力，可为图神经网络在大规模在线推理场景中的高效部署提供方法参考。

**关键词**：图神经网络，自适应稀疏化，谱图分析，节点异质性，高效推理，跳距预算分配，异配图学习

---

::: abstract*
Graph Neural Networks (GNNs) have emerged as a mainstream paradigm for processing graph-structured data, achieving strong performance on tasks such as node classification and link prediction by recursively aggregating multi-hop neighborhood information under the guidance of graph topology. When GNNs are deployed beyond academic benchmarks into latency-sensitive industrial applications --- including recommender systems and real-time fraud detection --- the exponential expansion of the receptive field induced by multi-layer message passing creates a fundamental inference efficiency bottleneck that hinders large-scale deployment. Concurrently, structural quality defects that are pervasive in real-world graphs, such as noisy edges and missing connections, corrupt node representations through the fixed-adjacency training paradigm and thereby undermine the reliability of downstream tasks.

The Sparse Decomposition Graph Neural Network (SDGNN) addresses the efficiency challenge by reformulating inference as a single sparse matrix multiplication against the teacher model's output, compressing the dominant inference complexity to linear scale and achieving substantial speedup in practice. Nevertheless, SDGNN embeds a core structural limitation: a globally uniform regularization coefficient imposes identical sparsity constraints on every node, ignoring topological heterogeneity among nodes and introducing systematic bias on both homophilic graphs --- where high-degree hub nodes suffer excessive neighbor truncation --- and heterophilic graphs --- where high-frequency discriminative signals are suppressed by the uniform low-pass assumption.

To address this limitation, this thesis proposes the Multi-Scale Adaptive Sparse Graph Neural Network (MSAS-GNN), building on SDGNN with the central objective of improving node adaptivity while preserving the linear inference complexity advantage. The methodology proceeds through three stages: problem formalization and indicator extraction, three-dimensional adaptive mechanism design, and algorithm integration with error analysis.

On the methodological side, this thesis first extracts four categories of computable graph complexity indicators via spectral computation --- spectral gap, node spectral energy, normalized local graph entropy, and node centrality --- providing a quantitative basis for parameter design. Node-level adaptive parameters are then constructed along three dimensions. The frequency dimension performs heterophily-aware correction on node spectral energy using homophily statistics. The node dimension establishes a monotone mapping from node spectral energy to the regularization coefficient, assigning greater neighborhood retention budget to nodes with richer spectral information. The hop-distance dimension designs near-neighbor-priority layer-wise budget allocation strategies based on the empirical observation that multi-hop information contributions decay with hop distance, comparing uniform and geometric-decay allocation schemes. At inference time, the three-dimensional parameter system retains the single sparse matrix multiplication form of SDGNN, keeping the dominant inference complexity at linear scale.

Experiments were conducted on six static node classification benchmark datasets --- Cora, Citeseer, PubMed, ogbn-arxiv, Chameleon, and Squirrel --- with all trials repeated under ten random seeds and significance assessed via the Wilcoxon signed-rank test. MSAS-GNN outperformed the baseline SDGNN across all six datasets: on the three homophilic citation network datasets, average accuracy improved by approximately 1.4 percentage points; on the large-scale graph ogbn-arxiv, accuracy reached 75.13%, exceeding SDGNN's 74.27%; on heterophilic graphs, the gain was approximately 2.3 times that observed on homophilic graphs, with Chameleon and Squirrel reaching 67.2% and 56.9%, respectively, and all differences were statistically significant. In terms of inference efficiency, MSAS-GNN achieved approximately 13.6x speedup over the standard message-passing baseline on ogbn-arxiv, with an average speedup of approximately 8.0x across four datasets and a reduction in GPU memory usage of approximately 74.8%. Ablation studies identified node spectral energy as the dominant contributor within the three-dimensional parameter system, and showed that the near-neighbor-priority hop-distance budget allocation outperforms uniform allocation on both accuracy and approximation error, consistent with the intended design direction.

This thesis presents a node-level differentiated sparsity allocation framework driven by computable graph complexity indicators and supported by spectral graph analysis. By improving adaptability to node topological heterogeneity and graph-type variation while maintaining linear inference complexity, the proposed framework offers a methodological reference for the efficient and reliable deployment of GNNs in large-scale online inference settings.
:::

::: denotation
多尺度自适应稀疏图神经网络（Multi-Scale Adaptive Sparse GNN，本文提出）

稀疏分解图神经网络（Sparse Decomposition GNN，Hu等，2024）

图神经网络（Graph Neural Network）

图卷积网络（Graph Convolutional Network，Kipf & Welling，2017）

图注意力网络（Graph Attention Network，Veličković等，2018）

图同构网络（Graph Isomorphism Network，Xu等，2019）

简单图卷积（Simple Graph Convolution，Wu等，2019）

消息传递神经网络（Message Passing Neural Network，Gilmer等，2017）

近似个性化 PageRank 传播图网络（Approximate Personalized PageRank propagation of neural predictions，Gasteiger/Klicpera等，2019）

可扩展 Inception 图神经网络（Scalable Inception Graph Neural Networks，Frasca等，2020）

频率自适应图卷积网络（Frequency Adaptive Graph Convolution Network，Bo等，2021）

广义PageRank图神经网络（Generalized PageRank GNN，Chien等，2021）

异配图超低通滤波图卷积网络（Beyond Homophily GCN，Zhu等，2020）

几何聚合图卷积网络（Geometric Aggregation GCN，Pei等，2020）

无图神经网络（Graph-less Neural Networks，Zhang等，2022），蒸馏路线，推理期无需访问图结构；为第6章实验基线之一

异配图全局同配建模方法，对应论文"Finding Global Homophily in Graph Neural Networks When Meeting Heterophily"中的方法简称；正文第2章综述引用，非实验基线

大规模异配图网络（Scalable Network for Large Heterophilic Graphs，Lim等，2021）

快速采样图卷积网络（Fast Learning with Graph Convolutional Networks，Chen等，2018）

归纳式大规模图表示学习（Inductive Representation Learning on Large Graphs，Hamilton等，2017）

基于图采样的归纳学习方法（Graph Sampling Based Inductive Learning，Zeng等，2020）

随机边丢弃正则化（DropEdge: Towards Deep Graph Convolutional Networks，Rong等，2020）

最小角度回归（Least Angle Regression，Efron等，2004）

噪声对比估计互信息最大化损失（InfoNCE Contrastive Loss）

多层感知机（Multilayer Perceptron）

$t$分布随机近邻嵌入可视化（$t$-distributed Stochastic Neighbor Embedding）

图对比表示学习框架（Graph Contrastive Representation Learning，Zhu等，2020），附录B.1对比学习增强路径的参考框架

基于近似 Personalized PageRank 的可扩展图神经网络（Scaling Graph Neural Networks with Approximate PageRank，Bojchevski等，2020）

属性图，$V$ 为节点集，$E$ 为边集，$X$ 为特征矩阵

图节点数，$n=|V|$

有向非零元数，$m:=\mathrm{nnz}(A)=2|E|$（复杂度分析统一口径）

节点输入特征维度

节点嵌入（隐藏）维度，区别于输入特征维度 $D$

图邻接矩阵

度矩阵，对角元素为各节点度数（区别于特征维度 $D$）

全图平均度，$\bar{d}=m/n$

节点 $i$ 的一跳邻域

节点 $i$ 的第 $l$ 跳邻居环层，$\mathcal{R}_l(i)=\mathcal{N}_l(i)\setminus\mathcal{N}_{l-1}(i)$

边级同配率（edge homophily ratio）

节点 $i$ 的节点级同配率

统一控制变量：同配图取 $h_{\mathrm{edge}}$，异配图取 $h_i$

归一化图拉普拉斯矩阵，$\tilde{\mathcal{L}}=I-D_{\mathrm{deg}}^{-1/2}AD_{\mathrm{deg}}^{-1/2}$，特征值 $\lambda_j\in[0,2]$

$\tilde{\mathcal{L}}$ 的谱分解矩阵，$\tilde{\mathcal{L}}=U\Lambda U^\top$，$U$ 列正交

谱间隙，即归一化拉普拉斯的最小非零特征值；本文统一采用从 $\lambda_0$ 开始的编号方式，因此连通图中 $\lambda_{\mathrm{gap}}=\lambda_1$（即通常意义上的第二小特征值）

Lanczos 迭代所求特征对数，视为常数，取值范围 $[50,100]$

多项式谱滤波器次数（与 $K_{\mathrm{eig}}$ 严格区分）

谱滤波器函数，作用于归一化拉普拉斯矩阵

节点谱能量，$\sum_k w_k(i)\lambda_k|U_{ik}|^2$

节点 $i$ 对频率分量 $k$ 的 softmax 权重，$\sum_k w_k(i)=1$

同配系数修正后的节点谱能量

节点 $i$ 的局部图熵

节点 $i$ 的度中心性

节点 $i$ 的 $k$-core 指数

传播算子 $P$ 的第二谱半径，$\mu_\star(P):=\max_{k\geq 2}|\mu_k(P)|$，刻画 $\mathbf{1}^\perp$ 子空间上的传播收缩速率；见第 4 章第 [4.3.1](#subsec:ch4-lemma36){reference-type="ref" reference="subsec:ch4-lemma36"} 节与附录A.4

$\sigma$-谱相似性的 Lanczos 代理估计量，开销 $O(K_{\mathrm{Lanczos}}m)$，与严格 $\sigma$ 的偏差在第 6 章统一报告；见第 5 章及附录B.2

稀疏图与原图谱近似误差度量，见第 5 章第 [5.2](#sec:ch5-spectral){reference-type="ref" reference="sec:ch5-spectral"} 节定义（$\sigma\geq 1$，越接近1谱结构保持越精确）

第 $l$ 层节点表示矩阵，$H^{(l)}\in\mathbb{R}^{n\times d}$

教师 GNN 目标表示矩阵，$H^*\in\mathbb{R}^{n\times d}$

节点特征变换，$\Phi\in\mathbb{R}^{n\times d}$

SDGNN/MSAS-GNN 推理输出（稀疏分解形式）

稀疏权重矩阵（全文单一矩阵，非逐层独立），$\Theta\in\mathbb{R}^{n\times n}$

推理期固化后的稀疏权重矩阵，结构不再更新

稀疏拓扑的有向非零元数，$m_{\mathrm{sparse}}:=\mathrm{nnz}(\Theta)$

各节点稀疏预算的全图平均值，$\bar{k}=\frac{1}{n}\sum_i k_i^{\mathrm{total}}$

节点级正则系数，全文唯一形式（无层级上标），$\tau(i)\in[\tau_{\min},\tau_{\mathrm{base}}]$

InfoNCE 温度超参（与节点级正则系数 $\tau(i)$ 严格区分，不缩写为 $\tau$）

高斯核边权重的带宽参数，$w_{ij}=\exp(-\gamma_x\|x_i-x_j\|_2^2)$，其中 $\gamma_x=1/(2\hat{\sigma}^2)$

跳距预算分配参数；$\xi=1/2$ 对应以 $\sqrt{p_l}$ 为参考的精确分配，$\xi=1$ 对应工程近似分配

特征变换参数 $W_\phi$ 的 $\ell_2$ 正则系数，仅用于式（3-6）中的权重衰减项

SDGNN 全局统一正则系数（本文用于说明其局限）

节点 $i$ 的总稀疏预算

节点 $i$ 在第 $l$ 跳距的预算配额，$\sum_l k_i^{(l)}=k_i^{\mathrm{total}}$

第 $l$ 跳距保留率（预算分配，非正则系数乘子），$p_l\in(0,1]$，且 $p_1\geq p_2\geq\cdots$

谱近似误差（函数型量），定义为 $\mathcal{E}_{\mathrm{approx}}:=\frac{1}{\sqrt{n}}\|H^*-\hat{H}\|_F$

第3章性能保持准则中的总体表示误差容许阈值

第 $l$ 跳传播允许的信息损失预算，用于第4章跳距保留率下界分析

第5章条件化误差上界中的整体误差目标

能量阈值尺度系数，$E_{\mathrm{threshold}}=c_E\hat{\lambda}_{\mathrm{gap}}/\bar{\lambda}$

节点谱能量阈值，用于 $\tau(i)$ 的指数映射

附录B参数反向设计中的参考能量阈值

附录B中谱相似性代理估计采用的 Lanczos 固定迭代步数

附录B参数反向设计中的数值稳定常数

下游任务（节点分类）交叉熵损失

稠密基线模型推理时间，口径以对应实验表注为准（表 6.2/6.3 采用 ms/batch，break-even 分析换算为秒/全图调用）

MSAS-GNN 推理时间，口径以对应实验表注为准（同上）

推理加速比，$\mathrm{Speedup}=t_{\mathrm{dense}}/t_{\mathrm{sparse}}$

总预处理时间（含指标计算、LARS路径求解与结构固化）

预处理 break-even 调用次数，$Q_{\mathrm{be}}=t_{\mathrm{pre}}/(t_{\mathrm{dense}}-t_{\mathrm{sparse}})$
:::

# 绪论 {#chap:intro}

在线图推理的核心矛盾在于：多层消息传递能够提升节点表示能力，却会因多跳邻域扩张显著推高推理代价。针对这一矛盾，SDGNN以单次稀疏分解替代逐层消息传递，为高效在线推理提供了可行思路，但其全局统一稀疏预算仍难以适应节点拓扑异质性。本文围绕这一局限展开研究，提出多尺度自适应稀疏图神经网络（MSAS-GNN）[]{#idx:name:msasgnn label="idx:name:msasgnn"}，而本章从研究背景与意义、国内外研究现状、研究内容与创新点、技术路线与论文组织结构四个层次说明本文的研究动机与整体安排，涉及贡献性符号与形式化结论统一在第 3--5 章建立。

![SDGNN全局统一稀疏预算局限与MSAS-GNN谱驱动自适应改进示意图](1-1.pdf){#fig:ch1-limitation-overview width="100%"}

## 研究背景与意义 {#sec:ch1-background}

图结构数据以节点和边刻画实体与关联，在社交网络、推荐系统、知识图谱与生物网络等领域普遍存在。图神经网络（Graph Neural Network，GNN）通过消息传递在图拓扑引导下递归聚合多跳邻域信息，在节点分类、链接预测等任务上表现出色，但$L$层GNN有效感受野规模约为$O(\bar{d}^L)$（$\bar{d}$为全图平均度），邻居收集与内存访问开销随层数快速膨胀。Hu等提出稀疏分解图神经网络（Sparse Decomposition of Graph Neural Networks，SDGNN[]{#idx:name:sdgnn label="idx:name:sdgnn"}），将推理重新表述为对教师GNN输出的单次稀疏分解[]{#idx:theme:sparsedecomp label="idx:theme:sparsedecomp"}------推理阶段仅需执行$\hat{H}=\Theta^\top\tilde{\Phi}$，主项复杂度降至$O(n\bar{k}d)$（$\bar{k}$为平均非零权重数），以LARS求解器在全局统一正则系数$\lambda_{\mathrm{reg}}$下构建Lasso子问题，训练后固化$\Theta$[@Hu2024SDGNN]。

以同一$\lambda_{\mathrm{reg}}$约束所有节点稀疏预算[]{#idx:theme:sparsebud label="idx:theme:sparsebud"}，隐含"节点最优稀疏度相同"的前提。真实图中节点拓扑特性通常存在显著差异[]{#idx:theme:tophete label="idx:theme:tophete"}：枢纽节点往往承载更多全局结构信息，过激进稀疏化可能截断关键邻居；外围节点则更容易包含局部冗余连接，过保守稀疏化会削弱压缩收益。在异配图[]{#idx:theme:heterophily label="idx:theme:heterophily"}上，高频结构信号与低通传播假设更容易出现不匹配，统一参数因而难以同时兼顾同配与异配场景，如图 [1.1](#fig:ch1-limitation-overview){reference-type="ref" reference="fig:ch1-limitation-overview"} 所示。这一根本局限既在理论层面呼唤节点谱域特征与稀疏预算之间的量化关联，在实践层面也制约了推荐系统、实时风控等延迟敏感服务的部署灵活性。本文在保持$O(n\bar{k}d)$推理口径不变的前提下，尝试以图谱理论建立节点级自适应稀疏预算分配框架，使稀疏化方案在同配与异配两类图结构上均具备可追溯的设计依据，具体加速效果在第 6 章实验中给出。

## 国内外研究现状 {#sec:ch1-related}

本节围绕与本文核心研究主线直接相关的三条文献脉络，概述国内外研究进展，为第 [1.3](#sec:ch1-contribution){reference-type="ref" reference="sec:ch1-contribution"} 节的问题界定与第 2 章的系统综述形成清晰分工。

### 图神经网络在线推理效率相关研究 {#subsec:ch1-gnn-efficiency}

Gilmer等[@Gilmer2017MPNN][]{#idx:name:gilmer label="idx:name:gilmer"}将多种图学习方法统一为消息传递神经网络（MPNN）框架，GCN[@Kipf2017GCN][]{#idx:name:gcn label="idx:name:gcn"}、GAT[@Velickovic2018GAT][]{#idx:name:gat label="idx:name:gat"}和GIN[@Xu2019GIN][]{#idx:name:gin label="idx:name:gin"}分别从低通谱滤波、注意力加权和表达上界三个角度扩展了消息传递范式。这类方法的单轮前向开销通常与层数、边数和隐藏维度密切相关；以GCN为例，其全图前向主项复杂度可写为$O(L\cdot m\cdot d)$。在推荐系统、实时风控等毫秒级延迟约束场景中，多跳邻居收集形成大规模部署的根本瓶颈。随层数增加，过平滑[]{#idx:theme:oversmooth label="idx:theme:oversmooth"}[@Li2018OverSmoothing]使节点表示趋于同质，过压缩[]{#idx:theme:oversquash label="idx:theme:oversquash"}[@Alon2021OverSquashing]使远端信息在瓶颈处丢失，实用层数因此通常限于2--3层。两类固有局限进一步强化了以稀疏分解替代逐层消息传递的技术动机。

### 高效与稀疏图神经网络方法 {#subsec:ch1-sparse}

针对感受野扩张带来的推理瓶颈，现有方法沿三条路线演进。邻居采样路线（GraphSAGE[@Hamilton2017GraphSAGE][]{#idx:name:graphsage label="idx:name:graphsage"}、FastGCN[@Chen2018FastGCN][]{#idx:name:fastgcn label="idx:name:fastgcn"}、GraphSAINT[@Zeng2020GraphSAINT][]{#idx:name:graphsaint label="idx:name:graphsaint"}）在训练阶段有效控制计算量，但推理时仍需访问完整邻居，且采样策略固定，难以因节点差异化调整。预计算线性化路线（SGC[@Wu2019SGC][]{#idx:name:sgc label="idx:name:sgc"}、APPNP[@Gasteiger2019APPNP][]{#idx:name:appnp label="idx:name:appnp"}、SIGN[@Frasca2020SIGN]）大幅降低推理开销，但传播模式固化，无法逐节点自适应。稀疏分解路线中，SDGNN[@Hu2024SDGNN]将推理主项复杂度压至$O(n\bar{k}d)$（$\bar{k}\ll\bar{d}^L$），成为本文的直接基线；其关键局限在于全局统一正则系数$\lambda_{\mathrm{reg}}$对枢纽节点与外围节点施加相同稀疏压力，节点拓扑差异在频率域的系统性偏差难以消除，构成本文第 3--5 章三段式自适应稀疏机制设计的直接动机。

### 异配性图学习与谱视角方法 {#subsec:ch1-heterophily}

边级同配率$h_\text{edge}$[]{#idx:theme:homorate label="idx:theme:homorate"}定量刻画相连节点的标签一致性，低通假设在同配图上有合理依据，但在异配图上将异类邻居信息混合至目标节点，导致性能下降[@Pei2020GeomGCN]。H2GCN[@Zhu2020H2GCN][]{#idx:name:h2gcn label="idx:name:h2gcn"}、GPR-GNN[@Chien2021GPRGNN][]{#idx:name:gprgnn label="idx:name:gprgnn"}、FAGCN[@Bo2021FAGCN][]{#idx:name:fagcn label="idx:name:fagcn"}分别从多阶聚合分离、自适应频率权重和高低通滤波组合角度突破低通限制，LINKX[@Lim2021LINKX][]{#idx:name:linkx label="idx:name:linkx"}彻底分离特征变换与传播以适应大规模异配图；但上述方法均采用全图共享的频率参数，节点级频率自适应能力不足。谱间隙$\lambda_\text{gap}$[]{#idx:theme:specgap label="idx:theme:specgap"}可刻画全局连通性，节点级频率响应还需结合局部谱统计量进一步量化，这正是第 4 章频率维度设计的出发点。

## 研究内容与创新点 {#sec:ch1-contribution}

上述三条文献脉络表明：以SDGNN为代表的稀疏分解路线以全局统一参数主导节点稀疏预算，节点拓扑差异在频率域的影响尚未得到系统理论化；现有异配图方法的频率参数停留于全图共享层面，节点级自适应能力不足。这一研究空白构成本文的直接动机。

本文的核心研究问题可表述为：在保持静态推理主项复杂度口径$O(n\bar{k}d)$不变的前提下，如何以可计算谱指标为依据，建立节点级差异化稀疏预算分配方案，使稀疏化方案在同配与异配两类图结构上均具备可追溯的设计依据，并在实验中得到系统验证。围绕这一目标，第 3--5 章的三段式方法链从问题形式化与指标提取、三维机制设计、算法整合与复杂度分析三个层次依次推进。

在方法设计层面[]{#idx:innov:3dim label="idx:innov:3dim"}，本文在第 3 章建立了谱间隙、节点谱能量、归一化局部图熵与节点中心性四类可计算图复杂度量化指标，为后续节点自适应参数的构造提供统一的可计算依据，使参数设计从依赖经验调参转向基于图结构特征的量化分配。在此基础上，第 4 章沿频率、节点、跳距三个维度构建节点级自适应参数体系：频率维度依据标签信号谱位置与同配率之间的近似关联实现异配感知的谱能量修正；节点维度依据谱能量与邻居有效信息量的正相关关系设计节点级自适应正则系数；跳距维度依据多跳传播信息逐层衰减的经验规律设计分层预算分配策略。三维参数在推理阶段均固化为单次稀疏矩阵乘形式，推理主项复杂度维持$O(n\bar{k}d)$的线性量级。第 5 章完成LARS端到端整合与复杂度分析。

在实验验证层面，本文在六个静态节点分类基准数据集上的测试表明，MSAS-GNN总体优于基线SDGNN，在异配图上的提升幅度约为同配图的2.3倍；相对消息传递基线在ogbn-arxiv上实现约13.6倍推理加速，四数据集平均约8.0倍，显存降低约74.8%；全部结果通过Wilcoxon符号秩检验的显著性验证，具体见第 6 章。

## 技术路线与论文组织结构 {#sec:ch1-structure}

如图 [1.2](#fig:ch1-tech-route){reference-type="ref" reference="fig:ch1-tech-route"} 所示，全文七章可归为四个功能层次：第 1--2 章建立研究背景与理论基础；第 3--5 章构成核心方法链，按"问题定义$\rightarrow$三维机制设计$\rightarrow$算法理论整合"三个层次递进，以$O(n\bar{k}d)$推理口径为共同约束；第 6 章开展系统实验评估；第 7 章进行全文总结与展望。各章功能定位如下：第 1 章建立研究动机与问题框架；第 2 章围绕三条文献脉络建立基础理论；第 3 章完成图学习基本设定与四类指标提取；第 4 章设计三维节点级自适应参数体系；第 5 章完成算法整合、误差分析与复杂度分析；第 6 章在多类节点分类数据集上开展主实验、消融与效率评估；第 7 章进行研究总结与未来工作展望。附录A对各章核心设计公式的理论依据进行简要说明，附录B补充完整算法流程，附录C提供实验细节。

![本文技术路线与章节接口示意图](1-2.pdf){#fig:ch1-tech-route width="100%"}

## 本章小结 {#sec:ch1-summary}

本章的任务不是提前展开方法细节，而是把后续研究的起点交代清楚。围绕在线图推理中的效率---表达能力矛盾，本章首先以SDGNN为切入点，指出全局统一稀疏预算难以回应节点拓扑差异这一核心局限；随后沿推理效率、高效稀疏化与异配性图学习三条文献脉络收束出本文的关键研究空白，即节点级自适应稀疏预算尚缺少统一的谱论依据；在此基础上，本文将研究问题限定为在保持$O(n\bar{k}d)$静态推理主项口径下构建谱驱动的三维自适应参数体系。下一章将进一步整理后续方法章节反复调用的基础概念、代表方法与研究空白。

# 相关工作 {#chap:related}

大规模图的在线推理效率是图神经网络面临的持续挑战。$L$层消息传递GNN的有效感受野在最坏情况下可达$O(\bar{d}^L)$，邻居收集与内存访问开销随层数快速放大。稀疏分解图神经网络（SDGNN）将多层GNN输出的逼近转化为单次稀疏矩阵乘，把推理复杂度降至$O(n\bar{k}d)$[@Hu2024SDGNN]；然而，其全局统一的Lasso正则系数$\lambda_{\mathrm{reg}}$对所有节点施加相同稀疏惩罚，无法适配同配节点与异配节点在拓扑角色和频域参与方式上的根本差异，是本文方法链的核心改进出发点。本章围绕三条文献脉络展开：第 [2.1](#sec:ch2-gnn-basics){reference-type="ref" reference="sec:ch2-gnn-basics"} 节建立全篇复用的基础符号体系与谱图理论背景；第 [2.2](#sec:ch2-efficient){reference-type="ref" reference="sec:ch2-efficient"} 节梳理从采样加速、线性化预计算到SDGNN稀疏分解的技术演进，重点揭示其结构性局限；第 [2.3](#sec:ch2-heterophily){reference-type="ref" reference="sec:ch2-heterophily"} 节引入同配性度量工具，分析现有异配图处理方法的设计思路与边界。第 [2.4](#sec:ch2-summary){reference-type="ref" reference="sec:ch2-summary"} 节提炼三大研究空白并与第 3 章至第 5 章方法链逐一衔接。本章叙述范围限于与谱驱动自适应稀疏分解直接相关的背景，不引入本文原创定理、命题或算法，相关正式内容统一留待后续方法章展开。

![本章三条文献脉络与后续方法章节的对应关系](2-1.pdf){#fig:ch2-roadmap width="100%"}

## 图神经网络基础与谱图理论 {#sec:ch2-gnn-basics}

本节补充后续分析所需的核心符号，并引入谱图理论背景。全文统一符号见《符号和缩略语说明》，本章不再重复列示；本章新增符号均在首次出现处行内定义，涉及第 3 章至第 5 章方法接口的记号再于相应章节引言中集中说明。

无向图$G=(V,E,X)$由节点集$V$（$|V|=n$）、边集$E$和节点特征矩阵$X\in\mathbb{R}^{n\times D}$组成；邻接矩阵$A$为对称矩阵，度矩阵$D_\text{deg}$的对角元素$D_{\text{deg},ii}=d_i=\sum_j A_{ij}$，全图平均度$\bar{d}=m/n$，其中$m:=\mathrm{nnz}(A)=2|E|$。本文统一以$m$作为复杂度分析的图规模参数。归一化拉普拉斯矩阵定义为[]{#idx:theme:normlaplacian label="idx:theme:normlaplacian"} $$\tilde{\mathcal{L}} = I - D_\text{deg}^{-1/2}AD_\text{deg}^{-1/2},
  \label{eq:ch2-normlaplacian}$$

其谱分解$\tilde{\mathcal{L}}=U\Lambda U^\top$中特征值满足$0=\lambda_0\leq\lambda_1\leq\cdots\leq\lambda_{n-1}\leq 2$。本文统一采用从 $\lambda_0$ 开始的编号方式；因此对连通图，谱间隙记为$\lambda_\text{gap}:=\lambda_1$，即最小非零特征值（也就是通常所说的第二小特征值）。若图不连通，则更一般地取最小正特征值。该量刻画图的连通扩张性，与导通率（conductance）之间满足Cheeger型关系[@Shuman2013GSP]，在后续跳距预算衰减分析中作为图级结构量使用。多项式谱滤波器$g(\tilde{\mathcal{L}})=\sum_{k=0}^{K_\text{poly}}c_k\tilde{\mathcal{L}}^k$可在顶点域以$O(K_\text{poly}\cdot m\cdot D)$计算，无需显式特征分解；对大规模图，Lanczos迭代以$O(m\cdot K_\text{eig})$近似获取前$K_\text{eig}$个特征对[@Shuman2013GSP]，是第 3 章算法 [\[alg:ch3-lanczos\]](#alg:ch3-lanczos){reference-type="ref" reference="alg:ch3-lanczos"}的预计算基础。图信号处理（GSP）框架[@Ortega2018GSP]揭示低频分量编码全局平滑信息、高频分量刻画局部差异的基本规律，为第 4 章异配感知频率权重设计奠定了理论基础。

以Gilmer等人（2017）[@Gilmer2017MPNN]的消息传递框架（MPNN）来看，$L$层图神经网络的推理复杂度为$O(L\cdot m\cdot d)$；更重要的是，节点$i$的有效感受野在树状图上可达$O(\bar{d}^L)$量级，使在线单节点查询时延随层数快速放大。GCN[@Kipf2017GCN]、GAT[@Velickovic2018GAT]、GraphSAGE[@Hamilton2017GraphSAGE]、GIN[@Xu2019GIN]、SGC[@Wu2019SGC]等主流模型在不同程度上面临这一推理瓶颈，且层数增加还会引发过平滑（节点表示趋同）和过压缩（远程信息丢失）两类固有局限[@Li2018OverSmoothing; @Alon2021OverSquashing]。前者通常与反复低通传播导致的表示收缩有关，后者则更多与图上的拓扑瓶颈和长程信息压缩有关；二者都可从图的谱性质与结构性质中获得分析线索，但具体机理并不相同。这说明单纯增加层数无法从根本上改善推理效率，稀疏化分解方向是有依据的技术选择。

![消息传递GNN感受野扩张与在线推理瓶颈示意图](2-2.pdf){#fig:ch2-bottleneck width="100%"}

## 高效与稀疏图神经网络方法 {#sec:ch2-efficient}

针对感受野指数扩张瓶颈，学界发展出三条主要技术路线。基于采样的方法（GraphSAGE[@Hamilton2017GraphSAGE]、FastGCN[@Chen2018FastGCN]、GraphSAINT[@Zeng2020GraphSAINT]）通过控制每层邻居集合规模缓解训练阶段的邻域爆炸问题；在推理阶段，这类方法也可沿用采样近似控制开销，但若希望降低采样方差或保持预测稳定性，通常仍需访问较大邻域，因此对严格在线单节点查询时延的优化并不如稀疏分解路线彻底。传播与特征变换解耦的方法可进一步分为两类：一类以SGC[@Wu2019SGC]、SIGN[@Frasca2020SIGN]为代表，适合将多阶传播结果离线预计算后在推理期执行轻量变换；另一类以APPNP[@Gasteiger2019APPNP]为代表，通过个性化传播将预测与图扩散解耦，但其传播过程通常仍以迭代近似形式实现。总体而言，这类方法均显著缓解了图传播与在线推理开销的强耦合。

稀疏分解框架代表了更彻底的推理加速思路。Hu等提出的SDGNN[@Hu2024SDGNN]以稀疏矩阵分解逼近多层GNN输出，其推理阶段写为 $$\hat{H} = \Theta^\top \tilde{\Phi},
  \label{eq:ch2-sdgnn-infer}$$

其中$\tilde{\Phi}\in\mathbb{R}^{n\times d}$为教师GNN预计算的候选特征矩阵，稀疏权重矩阵$\Theta$（每列对应一个节点，最多$\bar{k}$个非零元）通过LARS算法离线求解以下Lasso问题： $$\min_{\theta_i} \|H_{i,:}^* - \tilde{\Phi}^\top\theta_i\|_2^2 + \lambda_{\mathrm{reg}}\|\theta_i\|_1, \quad i=1,\ldots,n.
  \label{eq:ch2-sdgnn-lasso}$$

推理复杂度降至$O(n\bar{k}d)$，实现了与多层图传播的彻底解耦。然而，式 [\[eq:ch2-sdgnn-lasso\]](#eq:ch2-sdgnn-lasso){reference-type="eqref" reference="eq:ch2-sdgnn-lasso"}中的正则系数$\lambda_{\mathrm{reg}}$对全图所有节点统一施加相同的$\ell_1$惩罚，是SDGNN最突出的结构性约束：高度枢纽节点与边缘节点、同配区域与异配区域的节点在拓扑角色和频域参与方式上本应各异，全局均一系数无法自动适配这种差异。这一根本不足引出了第 3 章至第 5 章方法链中三维自适应稀疏化机制的核心研究动机。近期也出现了若干与图稀疏化、动态图注意或拉普拉斯稀疏化相关的工作。PPRGo（Bojchevski等，2020）[@Bojchevski2020PPRGo]基于个性化 PageRank 传播近似，以线性时间实现快速全图推理，在大规模图上推理效率较高，但其传播系数对全图共享，节点级自适应性不足。GLNN（Zhang等，2022）[@Zhang2022GLNN]通过知识蒸馏将 GNN 压缩为完全去除图依赖的 MLP，推理期无需访问图结构，达到极低延迟，但由此放弃了对图拓扑的利用。上述两类方法均未从节点谱域特征出发设计自适应稀疏化预算，与 SDGNN 稀疏分解路线和本文的关注点存在本质差异，本文在第 6 章实验中将其纳入效率对比参照，以说明不同技术路线的效率---精度权衡边界。就本文关心的问题而言，现有工作仍缺少一条从可观测图谱性质出发、再过渡到节点级稀疏预算设计准则的系统分析链。

## 异配性图学习与谱视角相关方法 {#sec:ch2-heterophily}

第 [2.2](#sec:ch2-efficient){reference-type="ref" reference="sec:ch2-efficient"} 节揭示了SDGNN全局均一正则系数对节点异质性的结构性失配。这种异质性的深层来源，在于图上标签信号的频率分布差异。本节引入同配性度量工具，并梳理现有异配图处理方法，为第 4 章频率权重参数化提供直接理论铺垫。

### 同配性的度量与定义 {#subsec:ch2-homophily}

设每个节点$i\in V$具有类别标签$y_i$，边级同配率定义为同类标签边所占比例： $$h_\text{edge}=\frac{|\{(i,j)\in E\mid y_i=y_j\}|}{|E|}\in[0,1].
  \label{eq:ch2-edge-homophily}$$

对满足$|\mathcal{N}(i)|>0$的节点，节点级同配率定义为$h_i=|\{j\in\mathcal{N}(i)\mid y_j=y_i\}|/|\mathcal{N}(i)|$；对孤立节点，可在实验设置中单独说明其处理方式，例如约定$h_i=0$。该指标刻画逐节点的局部同配差异：对整体中等同配的图，$h_i$的分布往往从接近0到接近1全域覆盖[@Zhu2020H2GCN]。以同配性较强的引文网络与异配性较强的网页/维基类网络为典型对照，两类图在统计上通常表现出明显不同的同配水平；但具体数值会随数据版本、预处理流程和同配度量定义而变化，因此本文不在相关工作章固定给出单一数值。后续章节将以这些基础定义作为频率维参数化的输入背景。

### 代表性异配性图处理方法 {#subsec:ch2-heteromethods}

从频率视角看，同配图中标签信号集中于低频分量，标准低通聚合有效；异配图中判别信息更多分布在中高频，低通聚合主动抑制有用信号。现有方法沿三条路线回应这一问题。H2GCN（Zhu等，NeurIPS 2020）[@Zhu2020H2GCN]分离ego与邻居表示并引入二阶邻域，绕开一阶异类邻居；GPR-GNN（Chien等，ICLR 2021）[@Chien2021GPRGNN]以可学习广义PageRank系数$\{\gamma_k\}$自适应调节传播权重，频率响应由数据驱动确定： $$H^{(K)}=\sum_{k=0}^K\gamma_k(I-\tilde{\mathcal{L}})^k H^{(0)};
  \label{eq:ch2-gprgnn}$$

FAGCN（Bo等，AAAI 2021）[@Bo2021FAGCN]通过边级自门控系数$\alpha_{ij}^G\in[-1,1]$在消息传播中自适应整合低频与高频成分；LINKX[@Lim2021LINKX]和GloGNN[@Li2022GloGNN]则分别从结构---特征并行嵌入和全局节点关系聚合角度提供不同解决路径。

然而，上述方法的自适应粒度均不直接等同于"节点级稀疏预算"：GPR-GNN的$\{\gamma_k\}$对全图共享，FAGCN以边为粒度调节，二者均未形成"节点级局部同配环境$\rightarrow$节点级频率响应$\rightarrow$节点级稀疏预算"的完整映射；且这些方法推理期仍依赖完整图传播，与$O(n\bar{k}d)$稀疏分解接口不兼容。节点谱能量作为刻画节点在各频率分量上参与强度的关键指标，在上述方法中均未被显式引入。这三点共同界定了后续方法创新的边界，详见第 [2.4](#sec:ch2-summary){reference-type="ref" reference="sec:ch2-summary"} 节研究空白分析。

![同配图与异配图的频率响应差异示意图](2-3.pdf){#fig:ch2-heterophily-spectrum width="100%"}

## 本章小结与研究空白 {#sec:ch2-summary}

前述三节系统梳理了与本文核心方法链直接相关的理论基础与代表性方法，由此可识别出现有工作在三个层面的不足，并与第 3 章至第 5 章方法设计逐一衔接。

就节点级自适应稀疏化的设计依据而言，SDGNN的全局统一$\lambda_{\mathrm{reg}}$（式 [\[eq:ch2-sdgnn-lasso\]](#eq:ch2-sdgnn-lasso){reference-type="eqref" reference="eq:ch2-sdgnn-lasso"}）与DropEdge[@Rong2020DropEdge]的随机删边均对全图所有节点施加相同稀疏惩罚，未能区分不同拓扑角色节点的信息整合需求；现有工作尚缺少一条从可观测谱性质出发推导节点级稀疏预算分配依据的分析路径，该不足由第 3 章通过问题形式化与四类图复杂度指标的建立完成可计算化铺垫，并在第 4 章进一步落实为频率维、节点维与跳距维的三维自适应参数设计。在节点级同配差异与稀疏化的联合设计层面，GPR-GNN的全局共享传播系数、FAGCN的边级门控、H2GCN的高阶邻域分离各自在一定程度上缓解了异配问题，却均未形成从节点级局部同配环境到节点级频率响应再到节点级稀疏预算的完整路径；第 4 章基于修正谱能量完成频率维参数化，从而在稀疏候选邻居选择中引入局部同配信息的指导。此外，现有工作在节点级自适应稀疏化的参数解释、复杂度收益与误差关系上仍缺少统一讨论，本文在第 5 章对误差分析与复杂度分析进行系统整合，并与第 3 章、第 4 章的方法输出形成相互衔接的说明。

::::: threeparttable
::: {#tab:ch2-research-gaps}
  空白     核心局限描述                                                                   代表性相关方法                                        本文对应章节                 核心技术手段
  -------- ------------------------------------------------------------------------------ ----------------------------------------------------- ---------------------------- ----------------------------------------------
  空白一   全局统一稀疏化，缺乏谱理论指导的节点级自适应框架                               SDGNN（$\lambda_{\mathrm{reg}}$全局统一）、DropEdge   第 3 章（第 4 章继续展开）   问题形式化、指标提取与三维参数设计的前置接口
  空白二   异配处理使用全局频率权重，稀疏化未考虑节点级同配差异                           GPR-GNN、FAGCN、H2GCN                                 第 4 章                      基于节点级同配差异的谱能量修正与频率权重设计
  空白三   现有工作在节点级自适应稀疏化的参数解释、复杂度收益与误差关系上仍缺少统一讨论   谱图理论、稀疏近似与复杂度分析（分散发展）            第 5 章                      误差分析与适用条件说明、端到端复杂度分析

  : 现有方法研究空白与本文对应章节方案汇总
:::

::: tablenotes
注：本表中代表性方法的信息均引自本章正文各节；"核心技术手段"一列仅作方向性概述，不在本章提前给出后续章节的原创符号；相关正式记号统一在对应章节引言的符号约定表中定义。
:::
:::::

![三大研究空白与本文方法链的对应关系图](2-4.pdf){#fig:ch2-research-gaps width="100%"}

以上三大空白由本文第 3 章至第 5 章的三段式方法链依次填补：第 3 章从问题形式化与指标体系出发，第 4 章实现三维自适应机制设计，第 5 章完成误差分析与复杂度分析说明。本章建立的符号体系将在后续章节中直接复用。具体地，$\tilde{\mathcal{L}}$、$\lambda_\text{gap}$、$m = \mathrm{nnz}(A) = 2|E|$ 及SDGNN的式 [\[eq:ch2-sdgnn-infer\]](#eq:ch2-sdgnn-infer){reference-type="eqref" reference="eq:ch2-sdgnn-infer"}--[\[eq:ch2-sdgnn-lasso\]](#eq:ch2-sdgnn-lasso){reference-type="eqref" reference="eq:ch2-sdgnn-lasso"}将作为第 3 章问题形式化与指标体系构建的直接起点（教师GNN目标输出 $H^*$ 在第 3 章第 3.1.2 节正式定义）；本章给出的同配性定义$h_\text{edge}$与$h_i$将在第 3 章统一问题设定下被计算、组织并作为接口指标输出，随后在第 4 章第 4.1 节作为频率维参数化输入；谱稀疏化误差、信息衰减与复杂度分析之间的统一关系，则在第 5 章继续展开。

# 问题形式化与图复杂度指标体系 {#chap:problem-formulation}

在标准消息传递图神经网络的推理过程中，全图单次前向计算复杂度通常为 $O(L\cdot m\cdot d)$，随图规模扩展，在线推理时延显著上升。SDGNN[@Hu2024SDGNN]（Hu等）将多层 GNN 的中间表示逼近问题转化为对基准 GNN 输出 $H^*$ 的单次稀疏分解，推理阶段仅执行稀疏矩阵乘 $\hat{H}=\Theta^\top\tilde{\Phi}$，有效消除层间依赖；然而其全局统一正则系数对节点拓扑异质性缺乏感知，枢纽节点邻居被过度截断，而外围节点的稀疏潜力又未被充分利用。

本章在保留 SDGNN 单次稀疏推理架构的前提下，从数学建模层面为节点级自适应稀疏预算体系奠基。第 [3.1](#sec:ch3-problem){reference-type="ref" reference="sec:ch3-problem"} 节建立图学习基本设定、分析全局统一策略的局限，并给出统一优化目标与三项设计准则；第 [3.2](#sec:ch3-complexity){reference-type="ref" reference="sec:ch3-complexity"} 节构建四类图复杂度量化指标。其中，谱间隙、节点谱能量、局部图熵与节点中心性由算法 [\[alg:ch3-preprocess\]](#alg:ch3-preprocess){reference-type="ref" reference="alg:ch3-preprocess"} 批量预处理，同配系数按式 [\[eq:ch3-hedge\]](#eq:ch3-hedge){reference-type="eqref" reference="eq:ch3-hedge"}--[\[eq:ch3-hi\]](#eq:ch3-hi){reference-type="eqref" reference="eq:ch3-hi"} 以 $O(m)$ 单独计算；第 [3.3](#sec:ch3-ch3-summary){reference-type="ref" reference="sec:ch3-ch3-summary"} 节总结与第4、5章的接口。除特别说明外，本章默认图结构静态可信，即邻接矩阵 $A$ 在训练与推理期间固定且可靠访问。

本章整体架构如图 [3.1](#fig:ch3-overview){reference-type="ref" reference="fig:ch3-overview"} 所示。

![第3章整体流程图：从图输入到四类复杂度指标输出](3-1.pdf){#fig:ch3-overview width="100%"}

## 问题形式化与优化目标 {#sec:ch3-problem}

[]{#sec:ch3-obj-sec label="sec:ch3-obj-sec"}

### 图学习基本设定与SDGNN基准 {#subsec:ch3-graph-setup}

[]{#subsec:ch3-sdgnn label="subsec:ch3-sdgnn"}

给定无向图 $G = (V, E, X)$，其中 $|V|=n$，节点特征矩阵 $X\in\mathbb{R}^{n\times D}$。实现上采用对称邻接存储，记 $m:=\mathrm{nnz}(A)=2|E|$ 为邻接矩阵非零元数。若图中存在孤立节点，则本文统一采用"加自环后再归一化"的预处理约定，使 $\mathbf{D}^{-1/2}$ 合法；下文默认经该预处理后 $d_i>0$。归一化拉普拉斯 $\tilde{\mathcal{L}} = I - \mathbf{D}^{-1/2}A\mathbf{D}^{-1/2}$ 的谱分解为 $\tilde{\mathcal{L}} = U\Lambda U^\top$，特征值按升序排列： $$0 = \lambda_0 \leq \lambda_1 \leq \cdots \leq \lambda_{n-1} \leq 2
  \label{eq:ch3-eigenvalue}$$

谱间隙 $\lambda_{\mathrm{gap}} := \lambda_1$ 为第二小特征值，刻画图的全局连通强度。在半监督节点分类任务中，目标是学习映射 $f: V \to \{1,\ldots,C\}$，预测概率与任务损失为： $$\hat{y}_i = \mathrm{softmax}\!\left(f_{\mathrm{cls}}\!\left(\hat{h}(i);\, W_{\mathrm{cls}}\right)\right) \in \Delta^{C-1}
  \label{eq:ch3-prediction}$$ $$\mathcal{L}_{\mathrm{task}} = -\frac{1}{|V_{\mathrm{train}}|} \sum_{i\in V_{\mathrm{train}}} \sum_{c=1}^C \mathbf{1}[y_i=c]\log\hat{y}_{ic}
  \label{eq:ch3-taskloss}$$

许多真实网络的度分布呈长尾甚至近似幂律；当以方差 $\mathrm{Var}(\{d_i\})=\frac{1}{n}\sum_i(d_i-\bar{d})^2$ 度量的度异质性较大时，统一预算 $k$ 往往会对高度节点和低度节点产生截然不同的稀疏效果，这正是引入节点级自适应预算 $k_i$ 的直接动机： $$\mathrm{Var}(\{d_i\}) = \frac{1}{n}\sum_{i=1}^n (d_i - \bar{d})^2
  \label{eq:ch3-degreevar}$$

标准 GCN[@Kipf2017GCN]（Kipf & Welling, 2017）采用逐层更新 $H^{(l)} = \sigma(\bar{A} H^{(l-1)} W^{(l)})$，省略中间非线性并合并多层权重后退化为 $H^{(L)} = \bar{A}^L X W_{\mathrm{eq}}$，全图推理复杂度 $O(L\cdot m\cdot d)$。SDGNN[@Hu2024SDGNN]（Hu等）将固定传播权重替换为每节点可学习稀疏权重 $\theta_i$： $$\hat{\mathcal{F}}(i, X, A;\, W, \Theta) = \theta_i^\top \Phi(X;\, W)
  \label{eq:ch3-sdgnn-infer}$$

在 $\tilde{\Phi}$ 已预计算缓存的口径下，推理阶段仅执行稀疏矩阵乘 $\hat{H}=\Theta^\top\tilde{\Phi}$，主导复杂度降至 $O(n\bar{k}d)$。当 $\bar{k}\ll\bar{d}^{\,L}$ 时，该口径相较原始多层消息传递具有明显压缩优势。然而，SDGNN 对所有节点施加相同预算 $k_i=k$：枢纽节点实际覆盖率仅为 $k/d_i$，大量邻域信息丢失；外围低度节点预算冗余，稀疏潜力未被释放。这一"一刀切"策略的系统性偏差如图 [3.2](#fig:ch3-sdgnn-fail){reference-type="ref" reference="fig:ch3-sdgnn-fail"} 所示。

![SDGNN全局统一稀疏预算失效示意图](3-2.pdf){#fig:ch3-sdgnn-fail width="100%"}

### 统一优化目标与预算约束 {#subsec:ch3-objective}

本文以逼近教师/基准GNN的中间节点表示为主问题，决策变量为特征变换参数 $W_\phi$ 与稀疏权重矩阵 $\Theta$： $$\min_{W_\phi,\,\Theta}\quad \frac{1}{2}\|\Theta^\top\tilde{\Phi}-H^*\|_F^2 + \sum_{i=1}^n \tau(i)\|\theta_i\|_1 + \frac{\lambda_W}{2}\|W_\phi\|_F^2
  \label{eq:ch3-main-objective}$$ $$\mathrm{s.t.}\quad \mathrm{supp}(\theta_i) \subseteq \mathcal{N}_L(i),\quad \forall\, i\in V
  \label{eq:ch3-support-constraint}$$ $$\|\theta_i\|_0 \leq k_{\max},\quad \forall\, i\in V
  \label{eq:ch3-sparse-bound}$$ $$\sum_{i=1}^n \|\theta_i\|_0 \leq B_{\mathrm{total}}
  \label{eq:ch3-budget}$$

式 [\[eq:ch3-main-objective\]](#eq:ch3-main-objective){reference-type="eqref" reference="eq:ch3-main-objective"} 中，$H^*\in\mathbb{R}^{n\times d}$ 为教师GNN输出，$\tilde{\Phi}:=\ell_2\text{-normalize-row}(\phi(X;W_\phi))\in\mathbb{R}^{n\times d}$，$\lambda_W>0$ 为特征变换参数 $W_\phi$ 的 $\ell_2$ 正则系数，$\tau(i)\in[\tau_{\min},\tau_{\mathrm{base}}]$ 为节点自适应正则系数（代替 SDGNN 的全局统一系数）。感受野约束式 [\[eq:ch3-support-constraint\]](#eq:ch3-support-constraint){reference-type="eqref" reference="eq:ch3-support-constraint"} 保持局部性语义，$k_{\max}$ 为单节点硬预算上界，$B_{\mathrm{total}}$ 为全图总预算约束（可选，见第5章第5.1.3节）。后文以 $k_i:=\|\theta_i\|_0$ 记节点稀疏度，$\bar{k}:=\frac{1}{n}\sum_i k_i$ 为全图平均稀疏度。本文采用两阶段交替优化：Phase $\Theta$ 利用 LARS[@Efron2004LARS]（Efron等，2004）求解各节点的 $\tau(i)$-加权 Lasso 子问题；Phase $W$ 利用反向传播更新 $W_\phi$。由于联合优化问题为非凸，本文在工程实现上主要关注训练过程的稳定性与可实现性，而不主张严格的全局收敛保证（详见第5章第5.1.3节）。$\tau(i)$ 的可计算表达式在第4章第4.2节给出。

### 近似误差度量与三项设计准则 {#subsec:ch3-criteria}

本文统一采用 RMS-Frobenius 近似误差度量稀疏分解质量： $$\mathcal{E}_{\mathrm{approx}}(W, \Theta;\, W^*)
  \;:=\; \frac{1}{\sqrt{n}}\,\bigl\|H^*(X,A;\,W^*) - \hat{H}(X,A;\,W,\Theta)\bigr\|_F
  \label{eq:ch3-approx-error}$$

$n^{-1/2}$ 缩放使不同规模图的误差具有可比性，与后续谱近似理论的 Frobenius 范数口径一致。后续各章的模块设计与第6章实验均以以下三项准则为统一口径。

其一为性能保持准则（记为准则1，Performance Preservation），要求逐节点表示误差的均方根不超过预设容许阈值 $\varepsilon_0>0$： $$\mathcal{E}_{\mathrm{approx}}(W, \Theta;\, W^*) \;\leq\; \varepsilon_0
  \label{eq:ch3-crit1}$$

该准则对大误差节点的比例同样形成统计控制[^1]。

其二为显著加速准则（记为准则2，Significant Speedup）。为避免口径混用，本文统一采用"$\tilde{\Phi}$ 已离线预计算并缓存后的聚合主项"作为在线推理口径。此时稀疏分解推理的主导复杂度为 $O(n\bar{k}d)$；若将特征变换一并计入完整端到端前向，则复杂度为 $O(nDd+n\bar{k}d)$。后文默认以前者讨论加速比，因此相较标准消息传递 GNN 的基础加速比约为： $$\mathrm{Speedup}_{\mathrm{base}} \;\approx\; \frac{L \cdot \bar{d}}{\bar{k}}
  \label{eq:ch3-speedup}$$

关于 $\bar{k}$ 与误差目标之间的定量关系，第5章将通过式 [\[eq:ch5-budget-lagrange\]](#eq:ch5-budget-lagrange){reference-type="eqref" reference="eq:ch5-budget-lagrange"}的条件化界及其后续参数反向设计给出更精确的讨论。

其三为自适应性准则（记为准则3，Adaptivity），以变异系数（CV）衡量稀疏预算的离散程度，要求其超过退化判据阈值： $$\mathrm{CV}(\{k_i\}) \;:=\; \frac{\sqrt{\frac{1}{n}\sum_{i=1}^n (k_i-\bar{k})^2}}{\bar{k}} \;>\; 0.3
  \label{eq:ch3-crit3}$$

$\mathrm{CV}>0.3$ 用于排除接近全局统一策略的退化情形；该阈值为启发式选择，敏感性分析见第6章消融实验。三项准则分别约束稀疏分解的精度、效率与自适应程度，准则1与准则2之间的 Pareto 边界将在第5章第5.2节建立谱保持框架后进一步刻画。

## 图复杂度量化指标体系 {#sec:ch3-complexity}

前节建立了优化目标与验收准则，但尚未回答 $\tau(i)$ 如何从图的可观测属性确定。本节提取四类互补的图复杂度量化指标，为第4章三维参数体系提供可计算输入；参数化设计留至第4章完成。 []{#subsec:ch3-indices label="subsec:ch3-indices"}

本节提取的四类指标涵盖全图级与节点级两个粒度，彼此互补。在全图级，谱间隙 $\lambda_{\text{gap}}$（归一化拉普拉斯第二小特征值）由 Cheeger 不等式（Alon & Milman, 1985[@AlonMilman1985]；Chung, 1997[@Chung1997]）关联图的 conductance $\phi(G)$，满足 $\phi(G)^2/2\leq\lambda_\text{gap}\leq 2\phi(G)$，可在 $O(mK_\text{eig})$ 时间内近似估计，在第4章第4.3.2节用于驱动分层跳距保留率策略。在节点级，节点谱能量 $E_{\text{spectral}}(i)$ 衡量节点在各频率成分上的参与强度；正文 B5 主线采用等权频率实例化，并将该谱能量直接作为 $\tau(i)$ 的谱输入。局部图熵 $H(i)$ 量化邻域消息流的均匀性，其归一化形式 $H_\text{norm}(i)$ 进入第4章第4.2.2节的融合公式；度中心性 $C_\text{deg}(i)$ 与 k-core 指数 $\text{core}(i)$ 互补刻画节点影响力与子图稠密层级，两者共同进入第4章第4.2.2节的 $\tau(i)$ 融合公式。四类指标由算法 [\[alg:ch3-preprocess\]](#alg:ch3-preprocess){reference-type="ref" reference="alg:ch3-preprocess"} 批量计算，适合训练前离线预计算并缓存，总复杂度见该算法后的讨论。

### 同配系数与异配性度量 {#subsec:ch3-homophily}

若后续启用同配感知频率接口，则节点谱能量中的频率权重 $w_k(i)$ 可依赖图的同配性，而同配性存在全局与局部两个层次。边级同配率定义为图中连接同类节点的边之比： $$h_\text{edge} = \frac{|\{(u,v)\in E : y_u = y_v\}|}{|E|}
  \label{eq:ch3-hedge}$$

节点级同配率度量单节点邻域中同类邻居的比例（对孤立节点约定 $h_i:=0$）： $$h_i = \frac{|\{j\in\mathcal{N}(i) : y_j = y_i\}|}{|\mathcal{N}(i)|},\qquad |\mathcal{N}(i)|>0
  \label{eq:ch3-hi}$$

$h_\text{edge}$ 为全图宏观度量，$h_i$ 为节点局部度量，二者可高度不一致------宏观同配图中桥接节点的 $h_i$ 可能极低，若以 $h_\text{edge}$ 决定其频率权重则会错配低频/高频偏好。两类同配率计算复杂度均为 $O(m)$；若在扩展实现中启用同配感知频率分支，应以训练集标签或结构代理量估计，避免标签泄漏。本文正文 B5 主线频率维采用等权实例化，因此此处同配率主要作为描述性统计与后续扩展接口保留。

### 节点谱能量、局部图熵与中心性 {#subsec:ch3-specenergy}

[]{#def:ch3-specenergy label="def:ch3-specenergy"} 本文对每个节点 $i\in V$，以前 $K_\text{eig}$ 个非平凡特征对定义节点谱能量（spectral node energy）： $$E_\text{spectral}(i) = \sum_{k=1}^{K_\text{eig}} w_k(i)\,\lambda_k\,|U_{ik}|^2
  \label{eq:ch3-specenergy}$$

式 [\[eq:ch3-specenergy\]](#eq:ch3-specenergy){reference-type="eqref" reference="eq:ch3-specenergy"}中，$\lambda_k$ 承担频率敏感性尺度，$|U_{ik}|^2$ 衡量节点 $i$ 对第 $k$ 个特征向量的参与强度，跳过 $k=0$ 以避免与度中心性的信息冗余。频率权重 $w_k(i)$ 采用 softmax 归一化： $$w_k(i) = \frac{\exp\!\bigl(\alpha_k(i) - \max_{k'}\alpha_{k'}(i)\bigr)}{\displaystyle\sum_{k''=1}^{K_\text{eig}}\exp\!\bigl(\alpha_{k''}(i) - \max_{k'}\alpha_{k'}(i)\bigr)}
  \label{eq:ch3-softmax-weight}$$

其中 $\alpha_k(i)$ 为频率打分函数。当第4章的同配感知参数化尚未注入时，默认令 $\alpha_k(i)\equiv 0$，此时 $w_k(i)=1/K_\text{eig}$，即各频率分量等权，得通用谱能量 $E_\text{spectral}(i)$；第4章第4.1.2节将在同一 softmax 接口上实例化 $\alpha_k(i)$，进而构造修正谱能量 $E_\text{spectral}^{(h)}(i)$。

归一化拉普拉斯满足谱分解 $\tilde{\mathcal{L}} = U\Lambda U^\top$，特征向量集 $\{u_k\}$ 构成图信号空间的正交基（Shuman et al., 2013[@Shuman2013GSP]；Ortega et al., 2018[@Ortega2018GSP]）。完整谱分解代价 $O(n^3)$ 不可行，本文采用 Lanczos 迭代[@Lanczos1950]将代价降至 $O(mK_\text{eig})$。

[]{#def:ch3-entropy label="def:ch3-entropy"} 本文将节点 $i$ 邻域消息流的均匀程度以局部图熵（local graph entropy）量化。给定邻居集 $\mathcal{N}(i)$ 与边权重 $\{w_{ij}>0\}$，令归一化分布 $p_{ij} = w_{ij}/\sum_{k\in\mathcal{N}(i)}w_{ik}$（约定 $0\ln 0=0$）。当 $|\mathcal{N}(i)|\le 1$ 时取 $H(i)=0$，$H_\text{norm}(i)=0$；当 $|\mathcal{N}(i)|>1$ 时，局部图熵及其归一化形式定义为： $$H(i) = -\sum_{j\in\mathcal{N}(i)} p_{ij}\ln p_{ij}, \qquad H_\text{norm}(i) = \frac{H(i)}{\ln|\mathcal{N}(i)|} \in [0,1]
  \label{eq:ch3-entropy}$$

均匀邻域对应熵的上界，单邻居主导时退化至下界0，因而能有效刻画邻域多样性。需要捕获邻域特征异质性时，可引入高斯核边权重 $w_{ij}=\exp(-\gamma_x\|x_i-x_j\|_2^2)$，其中 $\gamma_x=1/(2\hat{\sigma}^2)$，$\hat{\sigma}^2 := \mathrm{median}\{\|x_i-x_j\|_2^2 : (i,j)\in E\}$ 为基于边集的中位数估计，计算量为 $O(mD)$，与后文其他指标的口径保持一致。归一化局部图熵 $H_\text{norm}(i)$ 将在第4章第4.2.2节用于 $\tau(i)$ 的多因子融合公式。

度中心性 $C_\text{deg}(i)=d_i/(n-1)$（式 [\[eq:ch3-degcent\]](#eq:ch3-degcent){reference-type="eqref" reference="eq:ch3-degcent"}，度数已缓存时为 $O(n)$、从邻接现算时为 $O(m)$）与 k-core 指数（Batagelj & Zaversnik, 2003[@BatageljZaversnik2003]，式 [\[eq:ch3-kcore\]](#eq:ch3-kcore){reference-type="eqref" reference="eq:ch3-kcore"}，$O(m)$）在第4章第4.2.2节作为互补因子进入 $\tau(i)$ 融合公式： $$C_\text{deg}(i) = \frac{d_i}{n-1}
  \label{eq:ch3-degcent}$$ $$\text{core}(i) = \max\bigl\{k \mid i \in G_k,\ \forall u\in G_k:\ \deg_{G_k}(u)\geq k\bigr\}
  \label{eq:ch3-kcore}$$

算法 [\[alg:ch3-preprocess\]](#alg:ch3-preprocess){reference-type="ref" reference="alg:ch3-preprocess"} 整合上述三类节点级指标的批量预处理流程，在离线阶段一次性完成，计算结果缓存后供第4章参数化使用。多项式谱滤波器次数 $K_\text{poly}$（第5章第5.2.1节定义）与本章 $K_\text{eig}$ 严格区分，全文不得混用。

:::: algorithm
[]{#alg:ch3-lanczos label="alg:ch3-lanczos"} []{#alg:ch3-specenergy label="alg:ch3-specenergy"} []{#alg:ch3-entropy label="alg:ch3-entropy"}

::: algorithmic
图 $G=(V,E,X)$；归一化拉普拉斯 $\tilde{\mathcal{L}}$；截断数 $K_{\text{eig}}$（默认50）；收敛容差 $\tau_{\text{tol}}=10^{-10}$；频率打分 $\{\alpha_k(i)\}$（若未注入第4章参数化，则默认全零）；边权类型 $\texttt{wt}\in\{\texttt{uniform},\texttt{gaussian}\}$ 谱间隙近似值 $\hat{\lambda}_{\text{gap}}$；前 $K_{\text{pair}}$ 个非平凡特征对 $\{(\lambda_r, u_r)\}$；节点谱能量 $\{E_{\text{spectral}}(i)\}$；局部图熵 $\{H_{\text{norm}}(i)\}$；度中心性 $\{C_{\text{deg}}(i)\}$；k-core指数 $\{\text{core}(i)\}$ **// 第一阶段：Lanczos 迭代估计谱间隙与前 $K_\text{eig}$ 个非平凡特征对** 初始化 $v_1 \sim \mathcal{N}(0,I)$，$v_1 \leftarrow v_1/\|v_1\|_2$；$v_0 \leftarrow \mathbf{0}$，$\beta_0 \leftarrow 0$，$K_{\text{iter}}\leftarrow K_{\text{eig}}+1$ $w \leftarrow \tilde{\mathcal{L}}\,v_k$；$\alpha_k^* \leftarrow v_k^{\top}w$；$w \leftarrow w - \alpha_k^* v_k - \beta_{k-1}v_{k-1}$；$\beta_k \leftarrow \|w\|_2$ $K_{\text{eff}}\leftarrow k$；**break**（否则 $v_{k+1} \leftarrow w/\beta_k$，必要时执行选择性重正交化） 构造三对角矩阵 $T_{K_\text{eff}}$；对其做特征分解（升序），取 $\hat{\lambda}_{\text{gap}} \leftarrow \mu_1$；$K_{\text{pair}} \leftarrow \min(K_{\text{eig}},K_{\text{eff}}-1)$；$u_r \leftarrow [v_1,\ldots,v_{K_\text{eff}}]\,z_r$，$r=1,\ldots,K_{\text{pair}}$ **// 第二阶段：节点谱能量批量计算** $U_{:,1:K_\text{pair}} \leftarrow [u_1,\ldots,u_{K_\text{pair}}]$；$W_{i,:} \leftarrow \mathrm{softmax}(\alpha(i))$，$\forall i$ $E_{\text{spectral}}(i) \leftarrow \sum_{k=1}^{K_\text{pair}} W_{i,k}\cdot |U_{ik}|^2\cdot \lambda_k$ **// 第三阶段：局部图熵、度中心性与k-core计算** $\hat{\sigma}^2 \leftarrow \mathrm{median}\!\bigl(\{\|x_i-x_j\|_2^2 : (i,j)\in E\}\bigr)$；$\gamma_x \leftarrow 1/(2\hat{\sigma}^2)$；$w_{ij}\leftarrow \exp(-\gamma_x\|x_i-x_j\|_2^2)$ $w_{ij}\leftarrow 1$，$\forall (i,j)\in E$ $Z_i \leftarrow \sum_{j\in \mathcal{N}(i)} w_{ij}$；$H(i) \leftarrow -\sum_{j} (w_{ij}/Z_i)\ln(w_{ij}/Z_i)$（仅 $p_{ij}>10^{-10}$ 项计入） $H_{\text{norm}}(i) \leftarrow H(i)/\ln|\mathcal{N}(i)|$（$|\mathcal{N}(i)|>1$），否则取0；$C_{\text{deg}}(i)\leftarrow d_i/(n-1)$ 以 Batagelj-Zaversnik 算法[@BatageljZaversnik2003] $O(m)$ 时间计算 $\{\text{core}(i)\}_{i=1}^n$ $\hat{\lambda}_{\text{gap}}$；$\{(\lambda_r,u_r)\}_{r=1}^{K_\text{pair}}$；$\{E_\text{spectral}(i)\}$；$\{H_\text{norm}(i)\}$；$\{C_\text{deg}(i)\}$；$\{\text{core}(i)\}$
:::
::::

算法 [\[alg:ch3-preprocess\]](#alg:ch3-preprocess){reference-type="ref" reference="alg:ch3-preprocess"} 的总复杂度为 $O(mK_\text{eig}+nK_\text{eig}+mD)$（Gaussian模式）或 $O(mK_\text{eig}+nK_\text{eig}+m)$（uniform模式）。因此，在 uniform 权重或低维特征场景下，Lanczos 阶段的 $O(mK_\text{eig})$ 往往是主导项；而在高维特征图上，局部图熵的 $O(mD)$ 也可能成为主要开销。整体流程仍适合训练前离线执行并将结果缓存。

## 本章小结 {#sec:ch3-ch3-summary}

本章建立了面向节点级自适应稀疏化的数学基础与指标接口。第 [3.1](#sec:ch3-problem){reference-type="ref" reference="sec:ch3-problem"} 节分析了 SDGNN 全局统一策略在高度节点与低度节点上的系统性偏差，给出统一优化目标（式 [\[eq:ch3-main-objective\]](#eq:ch3-main-objective){reference-type="eqref" reference="eq:ch3-main-objective"}）与三项设计准则（性能保持/显著加速/自适应性，式 [\[eq:ch3-crit1\]](#eq:ch3-crit1){reference-type="eqref" reference="eq:ch3-crit1"}--[\[eq:ch3-crit3\]](#eq:ch3-crit3){reference-type="eqref" reference="eq:ch3-crit3"}）；第 [3.2](#sec:ch3-complexity){reference-type="ref" reference="sec:ch3-complexity"} 节提取四类可计算图复杂度指标，其中谱间隙、节点谱能量、局部图熵与节点中心性由算法 [\[alg:ch3-preprocess\]](#alg:ch3-preprocess){reference-type="ref" reference="alg:ch3-preprocess"} 批量预处理，同配系数按式 [\[eq:ch3-hedge\]](#eq:ch3-hedge){reference-type="eqref" reference="eq:ch3-hedge"}--[\[eq:ch3-hi\]](#eq:ch3-hi){reference-type="eqref" reference="eq:ch3-hi"} 单独计算。各指标通过明确接口传递至后续章节：$\hat{\lambda}_\text{gap}$ 驱动第4章第4.3.2节分层跳距保留率策略；$\{E_\text{spectral}(i)\}$ 经第4章第4.1.2节参数化后得修正谱能量，作为 $\tau(i)$ 的谱输入；$\{H_\text{norm}(i)\}$、$\{C_\text{deg}(i)\}$、$\{\text{core}(i)\}$ 进入第4章第4.2.2节的多因子融合公式；$h_\text{edge}$ 与 $h_i$ 在第4章第4.1.3节统一为 $h^\dagger(i)$ 驱动频率权重修正。$E_{\text{spectral}}(i)$ 保留频率权重接口 $w_k(i)$ 待第4章实例化。在本章接口的基础上，第4章将完成三维自适应参数的具体设计，第5章完成算法整合与复杂度分析。

# 谱驱动三维自适应稀疏机制 {#chap:adaptive-sparse}

第3章以四类图复杂度指标------谱间隙 $\hat{\lambda}_{\text{gap}}$、节点谱能量 $E_{\text{spectral}}(i)$、局部图熵 $H(i)$、节点中心性------作为可计算的结构特征，并以边级/节点级同配率 $h_{\text{edge}}$/$h_i$ 作为频率修正所需的辅助统计量，为自适应稀疏预算提供了量化基础。然而，这些量目前仍停留在"可计算输入"层面：它们分别刻画图结构的不同侧面，但如何将其转化为具体的稀疏参数（频率权重 $w_k(i)$、节点正则系数 $\tau(i)$、跳距预算配额 $k_i^{(l)}$），仍需针对各维度建立明确的理论映射。

本章正是完成这一转化的核心方法章。三个自然子问题分别对应三个维度。第一，频率维度：如何利用同配率 $h_{\text{edge}}$ 与节点级 $h_i$，将第3章的通用频率权重接口 $w_k(i)$ 实例化为同配/异配感知的自适应形式，并以此定义修正谱能量 $E_{\text{spectral}}^{(h)}(i)$？第二，节点维度：如何以 $E_{\text{spectral}}^{(h)}(i)$ 为核心输入，结合度中心性、k-core等拓扑因子，从谱扰动理论出发推导节点级正则系数 $\tau(i)$ 的可计算映射？第三，跳距维度：如何利用 $\hat{\lambda}_{\text{gap}}$ 与信息衰减界，为各跳距分配最优整数预算配额 $k_i^{(l)}$？三维设计相互独立而又最终汇聚：频率维产出 $E_{\text{spectral}}^{(h)}(i)$ 输入节点维，节点维产出 $\tau(i)$ 与跳距维产出 $k_i^{(l)}$ 共同进入第5章的LARS主循环。

本章结构如下：第 [4.1](#sec:ch4-freq){reference-type="ref" reference="sec:ch4-freq"} 节完成频率维度设计；第 [4.2](#sec:ch4-node){reference-type="ref" reference="sec:ch4-node"} 节建立节点维度的完整理论与算法框架；第 [4.3](#sec:ch4-hop){reference-type="ref" reference="sec:ch4-hop"} 节给出跳距维度的分层预算分配策略；第 [4.4](#sec:ch4-summary){reference-type="ref" reference="sec:ch4-summary"} 节总结三维贡献并说明与第5章的接口。

本章整体框架如图 [4.1](#fig:ch4-overview){reference-type="ref" reference="fig:ch4-overview"} 所示。

![第4章三维自适应稀疏机制整体框架图](4-1.pdf){#fig:ch4-overview width="100%"}

## 频率维度：异配感知谱能量修正 {#sec:ch4-freq}

第3章第 [3.2](#sec:ch3-complexity){reference-type="ref" reference="sec:ch3-complexity"} 节建立了节点谱能量、局部图熵与中心性三类图复杂度量化指标，并给出了相应的统一计算接口。对于节点谱能量，第3章第 [3.2.2](#subsec:ch3-specenergy){reference-type="ref" reference="subsec:ch3-specenergy"} 节仅保留了频率权重 $w_k(i)=\mathrm{softmax}(\alpha(i))_k$ 的一般形式，而尚未固定频率打分函数 $\alpha_k(i)$ 的具体参数化。若在所有图上采用同一组频率偏好，则该统一定义在实现上是简洁的；但当图由同配转向异配时，这种"统一实例"可能与真实判别频段发生失配。需要强调的是，本文并不将异配性简单等同为"整图统一高频主导"；更稳妥的表述是：低频平滑假设在同配图上通常更有效，而异配图中许多节点的判别信息会更多出现在非低频分量上，因此需要允许节点级频段偏向发生变化。本节正是针对这一问题，依据同配统计量与谱频段偏向之间的经验关联，对 $\alpha_k(i)$ 给出具体的同配/异配感知参数化，并据此定义修正谱能量 $E_{\text{spectral}}^{(h)}(i)$，作为第 [4.2](#sec:ch4-node){reference-type="ref" reference="sec:ch4-node"} 节节点级正则系数 $\tau(i)$ 的正式谱输入。本节所建立的统一控制变量 $h^\dagger(i)$ 与修正谱能量 $E_{\text{spectral}}^{(h)}(i)$ 共同构成三维自适应框架的频率维度基础。

### 动机：全局统一频率权重的局限 {#subsec:ch4-freq-motivation}

第3章第 [3.2.2](#subsec:ch3-specenergy){reference-type="ref" reference="subsec:ch3-specenergy"} 节保留了频率权重 $w_k(i)$ 的统一接口，但统一低频偏重策略无法充分覆盖异配图中的局部频段差异。若仍采用固定低频偏向，则在异配场景下可能整体性低估真正承载判别信息的那部分频率分量，进而使 $\tau(i)$ 的分配偏离节点的实际结构角色。该现象将在第 6 章的异配图实验中给出经验验证。本节根据同配系数自动确定频率权重偏向，使谱能量指标在两类图上均保持有效性。

### 同配图与异配图的频率权重差异 {#subsec:ch4-freq-design}

[]{#idx:innov:freq label="idx:innov:freq"}

本小节在第3章第 [3.2.1](#subsec:ch3-homophily){reference-type="ref" reference="subsec:ch3-homophily"} 节定义的 $h_{\text{edge}}$ 与 $h_i$ 基础上，给出频率打分函数 $\alpha_k(i)$ 的参数化规则，使 $w_k(i)=\mathrm{softmax}(\alpha(i))_k$ 自适应于图的同配性结构。$h_{\text{edge}}$ 与 $h_i$ 在实现中以训练集标签估计量或标签无关代理量代入，无泄漏口径沿用第3章第 [3.2.1](#subsec:ch3-homophily){reference-type="ref" reference="subsec:ch3-homophily"} 节约定。

[]{#lem:ch4-homo-spectral label="lem:ch4-homo-spectral"} 在正式参数化频率权重之前，有必要先建立同配率与类别信号谱位置之间的方向性关联，为分段参数化提供动机性依据。在二分类、类别近似均衡且度分布不过于不均匀的条件下，标签信号在归一化拉普拉斯上的 Rayleigh 商与边级同配率 $h_{\text{edge}}$ 可得到如下近似关系： $$\lambda_Y \approx 2(1-h_{\text{edge}}) + O\!\left(\frac{\mathrm{Var}(d)}{\bar d^2}\right).
\label{eq:ch4-lambda-rayleigh}$$

该结果（式 [\[eq:ch4-lambda-rayleigh\]](#eq:ch4-lambda-rayleigh){reference-type="eqref" reference="eq:ch4-lambda-rayleigh"}）仅在上述受限条件下提供方向性启发：随同配性下降，类别标签信号在谱域中的平均频率位置倾向于上移。因此在同配图与异配图上沿用相同频率偏置并不合适，频率权重宜随局部同配环境作相应调整。

上述近似关系的推导思路如下。取二分类归一化标签向量 $\tilde{Y}_i = y_i/\sqrt{|V_{y_i}|}$，其在 $\tilde{\mathcal{L}}$ 上的 Rayleigh 商为 $\lambda_Y = 1 - \tilde{Y}^\top D^{-1/2}AD^{-1/2}\tilde{Y}/2$。在类别均衡、度分布近均匀的假设下，按同类边与异类边展开该二次型，可得 $\tilde{Y}^\top D^{-1/2}AD^{-1/2}\tilde{Y} \approx 2(2h_{\text{edge}}-1)$（其中 $h_{\text{edge}} = |E_{\text{同}}|/|E|$）；代入 Rayleigh 商即得近似式。误差项 $O(\mathrm{Var}(d)/\bar{d}^2)$ 源自边归一化因子的均匀度近似，在强幂律图上不宜忽略，应作方向性依据而非精确估计；相关谱图理论分析可参考 Chung（1997）[@Chung1997] 及附录A.1的说明。

基于上述分析，本文采用分段 softmax 形式定义频率权重，以在统一表达下同时覆盖同配图与异配图两种情形： $$w_k(i) = \mathrm{softmax}(\alpha(i))_k \;\triangleq\; \frac{\exp\!\bigl(\alpha_k(i) - \max_{k'}\alpha_{k'}(i)\bigr)}{\displaystyle\sum_{k''=1}^{K_{\text{eig}}}\exp\!\bigl(\alpha_{k''}(i) - \max_{k'}\alpha_{k'}(i)\bigr)}
  \label{eq:ch4-lambda-approx}$$

其中 $K_{\text{eig}}$ 为 Lanczos 近似保留的特征对（eigenpairs）数量，$\alpha(i)=[\alpha_1(i),\ldots,\alpha_{K_{\text{eig}}}(i)]^\top$ 为节点 $i$ 的频率打分向量。式 [\[eq:ch4-lambda-approx\]](#eq:ch4-lambda-approx){reference-type="eqref" reference="eq:ch4-lambda-approx"}中减去 $\max_{k'}\alpha_{k'}(i)$ 仅用于数值稳定性处理，不改变 softmax 的相对大小关系。该写法的直接结果是：对任意节点 $i$，均有 $w_k(i)>0$ 且 $\sum_{k=1}^{K_{\text{eig}}}w_k(i)=1$，从而不同节点的谱能量处于统一可比尺度之上，避免了未归一化指数权重在不同节点之间难以横向比较的问题。

频率打分函数 $\alpha_k(i)$ 采用如下分段定义： $$\alpha_k(i) = \begin{cases} -\beta_{\mathrm{f}}\lambda_k, & h_{\text{edge}} \geq 0.5 \\ -\beta_{\mathrm{h}}(2h_i - 1)\lambda_k, & h_{\text{edge}} < 0.5 \end{cases}
  \label{eq:ch4-lambda-y}$$

其中 $\beta_{\mathrm{f}},\beta_{\mathrm{h}}>0$ 为频率维度的基准衰减系数。当 $h_{\text{edge}}\geq 0.5$ 时打分随 $\lambda_k$ 单调下降，退化为标准低频偏重；当 $h_{\text{edge}}<0.5$ 时，$(2h_i-1)$ 起符号切换作用，强异配节点（$h_i\to 0$）使高频权重增大，局部同配节点（$h_i>0.5$）仍偏向低频，实现节点级频段自适应。softmax 归一化保证不同节点权重处于同一概率单纯形，减最大值操作保证数值稳定。

表 [4.1](#tab:ch4-3-2){reference-type="ref" reference="tab:ch4-3-2"} 列出典型图数据集的同配性与谱特性定性分类，其"同配图偏低频、异配图高频成分更关键"的类型划分与式 [\[eq:ch4-lambda-y\]](#eq:ch4-lambda-y){reference-type="eqref" reference="eq:ch4-lambda-y"}给出的频率迁移趋势在定性上是一致的。各数据集的精确 $h_{\text{edge}}$ 统计值与准确率对比数字详见第 6 章，此处仅作类型说明，而不提前展开与实验设置强相关的定量比较。

::::: threeparttable
::: {#tab:ch4-3-2}
  **数据集**   **同配性量级（仅定性）**   **类型**   **类别信号频率特征**   **标准低通GCN适配性**
  ------------ -------------------------- ---------- ---------------------- -----------------------
  Cora         高                         强同配     低频主导               高
  Citeseer     较高                       同配       低频为主               高
  PubMed       高                         同配       低频主导               高
  Chameleon    低                         异配       高频成分关键           低
  Squirrel     低                         异配       高频成分关键           低
  Actor        低                         异配       高频成分关键           低
  Texas        很低                       强异配     高频主导               很低
  Cornell      很低                       强异配     高频主导               很低

  : 典型图数据集同配性与谱特性定性统计
:::

::: tablenotes
注：本表仅用于说明"同配/异配---频率偏向"的类型差异，因此同配性只保留定性量级，不在本章给出具体数值统计。若需给出精确数值，须逐项补充文献引用；具体统计结果及其与实验准确率的对应关系统一移至第 6 章相关表格，此处不提前展开。
:::
:::::

![同配图与异配图的频率能量分布对比](4-2.pdf){#fig:ch4-freq-energy width="100%"}

式 [\[eq:ch4-lambda-approx\]](#eq:ch4-lambda-approx){reference-type="eqref" reference="eq:ch4-lambda-approx"}--式 [\[eq:ch4-lambda-y\]](#eq:ch4-lambda-y){reference-type="eqref" reference="eq:ch4-lambda-y"}建立了分段 softmax 频率权重：$h_{\text{edge}}\geq 0.5$ 时退化为全局低频偏重；$h_{\text{edge}}<0.5$ 时由节点级 $h_i$ 决定局部频段偏向，允许同一张异配图中不同节点表现出不同频率偏置。该组权重进入第 [4.1.3](#subsec:ch4-espectralh){reference-type="ref" reference="subsec:ch4-espectralh"} 节修正谱能量的正式定义。

### 修正后谱能量 $E_{\text{spectral}}^{(h)}(i)$ 的定义与性质 {#subsec:ch4-espectralh}

[]{#idx:innov:espectral label="idx:innov:espectral"}

本小节引入统一控制变量 $h^\dagger(i)$ 将第 [4.1.2](#subsec:ch4-freq-design){reference-type="ref" reference="subsec:ch4-freq-design"} 节的分段参数化封装为单一理论对象，正式定义修正谱能量 $E_{\text{spectral}}^{(h)}(i)$，使其成为从第3章通用计算框架到第 [4.2](#sec:ch4-node){reference-type="ref" reference="sec:ch4-node"} 节 $\tau(i)$ 设计的理论锚点。

为统一同配/异配两种分支，定义 $$h^\dagger(i)=
\begin{cases}
h_{\text{edge}}, & h_{\text{edge}}\ge 0.5,\\
h_i, & h_{\text{edge}}<0.5,
\end{cases}$$

并记 $$E_{\text{spectral}}^{(h)}(i)=\sum_{k=1}^{K_{\text{eig}}} w_k(i)\,|U_{ik}|^2\,\lambda_k .
\label{eq:ch4-espectralh-def}$$

两端点处，$h^\dagger(i)\to 1$ 时低频权重集中，$h^\dagger(i)\to 0$ 时高频权重集中。由此可见，该构造为同配图与异配图提供了一种计算上简洁、与当前实验设置相匹配的频率偏好刻画方式，并为后续节点级正则系数设计提供统一谱输入。需要说明的是，该构造采用"全图类型判别 + 局部分支修正"的启发式规则；对于整体同配但局部存在强异配桥接节点的情形，其刻画能力仍有进一步细化空间，具体局限在第7章局限性章节中统一说明。

就修正谱能量的性质而言，当 $h^\dagger(i)$ 增大时，低频权重上升而高频权重下降；因此，$E_{\text{spectral}}^{(h)}(i)$ 的变化方向取决于节点 $i$ 的谱质量主要分布在哪一段频率上。若节点谱质量主要集中于高频端，则同配性增强时修正谱能量通常趋于下降；反之则通常趋于上升。本文在正文中仅保留这一方向性结论，而不将其表述为无条件成立的全局单调性定理。

严格单调性结论仍需附加"节点能量主要集中于特定频段"之类的充分条件。鉴于本文此处仅需建立后续 $\tau(i)$ 设计的方向性依据，正文不再追求无条件单调性定理；相关条件化分析见附录A.2"修正谱能量的方向性分析与分段权重条件说明"。

修正谱能量 $E_{\text{spectral}}^{(h)}(i)$（式 [\[eq:ch4-espectralh-def\]](#eq:ch4-espectralh-def){reference-type="eqref" reference="eq:ch4-espectralh-def"}）封装了频率维度设计的全部输出，直接作为第 [4.2](#sec:ch4-node){reference-type="ref" reference="sec:ch4-node"} 节 $\tau(i)$ 融合公式的正式谱输入；正式的多跳信息衰减界由第 [4.3](#sec:ch4-hop){reference-type="ref" reference="sec:ch4-hop"} 节的衰减界分析给出。

## 节点维度：节点级自适应正则系数 {#sec:ch4-node}

第 [4.1](#sec:ch4-freq){reference-type="ref" reference="sec:ch4-freq"} 节产出修正谱能量 $E_{\text{spectral}}^{(h)}(i)$，但频率维度仍对所有节点施加相同的稀疏惩罚形式。真实图中节点结构角色高度分化：枢纽节点承载全局拓扑骨架，叶节点贡献有限，统一参数 $\tau$ 在结构重要节点上过度稀疏会破坏谱特性，在冗余节点上保留过多无效边则造成计算浪费。本节以 $E_{\text{spectral}}^{(h)}(i)$ 为核心输入，建立节点谱能量与最优稀疏参数 $\tau(i)$ 的理论联系，形成"谱能量高 $\Rightarrow$ $\tau(i)$ 小 $\Rightarrow$ L1惩罚弱 $\Rightarrow$ 保留邻居多"的设计逻辑链。

![节点级正则系数 $\tau(i)$ 的三维自适应框架示意图](4-3.pdf){#fig:ch4-budget width="100%"}

### 节点级正则系数的构造原则 {#subsec:ch4-tau-design}

本节给出从 $E_{\text{spectral}}^{(h)}(i)$ 到 $\tau(i)$ 的可计算设计准则。需要强调的是，下述关系首先服务于参数构造，而非已经完全闭合的严格最优性定理；第5章将继续分析其与全局谱近似之间的联系。

[]{#thm:ch4-tau-design label="thm:ch4-tau-design"}[]{#idx:innov:tau label="idx:innov:tau"} 对节点 $i$，其稀疏化子问题的 Lasso/LARS 形式为： $$\min_{\theta_i}\ \frac{1}{2}\bigl\|\tilde{\Phi}^\top\theta_i - h_i^*\bigr\|_2^2 + \tau(i)\|\theta_i\|_1,$$

其中 $h_i^*$ 为教师GNN在节点 $i$ 处的目标中间表示。基于谱扰动分析的基本结论（参见附录A.3及相关谱图理论文献[@Chung1997]），谱能量较高的节点往往承载更丰富的判别性结构信息，稀疏化后产生的潜在信息损失也更值得控制，因此应为其分配较小的正则系数 $\tau(i)$，以保留更多邻居；结合 LARS 路径单调性（$\tau(i)$ 越大解越稀疏），$\tau(i)$ 应是 $E_{\text{spectral}}^{(h)}(i)$ 的单调递减函数，即高谱能量节点对应更弱的 $\ell_1$ 惩罚、低谱能量节点对应更强的 $\ell_1$ 惩罚。这一单调关联的直觉是：当节点 $i$ 拥有更高的修正谱能量 $E_{\text{spectral}}^{(h)}(i)$ 时，其局部子图承载了更丰富的判别信息，值得为其分配更大的邻居保留预算；反之，低谱能量节点的邻居信息冗余度更高，较激进的稀疏化不会造成明显信息损失。该映射存在若干工程近似（谱能量代理替换、扰动局部化假设），正式的全局谱相似性分析将由第5章给出。

满足上述单调准则的指数有界单因子实现为： $$\tau_{\text{mono}}(i)=\mathrm{clip}\!\left(
  \tau_{\text{base}}\exp\!\left(-\beta_{\tau}\frac{E_{\text{spectral}}^{(h)}(i)}{E_{\text{threshold}}}\right),
  \tau_{\min},\tau_{\text{base}}
  \right)
  \label{eq:ch4-wk-hetero}$$

其中 $E_{\text{spectral}}^{(h)}(i)$ 由式 [\[eq:ch4-espectralh-def\]](#eq:ch4-espectralh-def){reference-type="eqref" reference="eq:ch4-espectralh-def"}给出，$E_{\text{threshold}} = c_E\cdot\hat{\lambda}_{\text{gap}}/\bar{\lambda}$ 为能量阈值（$\bar{\lambda} = (\sum_{k=1}^{K_{\text{eig}}} \lambda_k)/K_{\text{eig}}$，$c_E>0$ 为无量纲尺度系数），$\tau_{\text{base}}$ 通过验证集在对数尺度搜索确定，搜索范围为 $[10^{-4},\,10^{-2}]$，默认初值 $10^{-3}$。式 [\[eq:ch4-wk-hetero\]](#eq:ch4-wk-hetero){reference-type="eqref" reference="eq:ch4-wk-hetero"}是满足单调设计目标的一种平滑实现，并非唯一最优形式；式 [\[eq:ch4-alphadef\]](#eq:ch4-alphadef){reference-type="eqref" reference="eq:ch4-alphadef"}在此基础上进一步整合多因子。

上述设计准则给出的不是"唯一正确的最优解"，而是满足单调设计目标的工程性参数构造：高谱能量枢纽/桥接节点倾向于获得更小 $\tau(i)$ 与更高保留强度；中等能量节点处于默认工作区间；低能量叶节点可采用更接近 $\tau_{\text{base}}$ 的较激进稀疏化。正式的全局谱相似性分析将由第5章相关分析给出；参数推荐值为 $\beta_{\mathrm{f}}=\beta_{\mathrm{h}}=\beta_{\tau}=1$，$\tau_{\text{base}}\in[10^{-4},10^{-2}]$，敏感性以第 6 章实验为准。

### 多因子融合的完整公式 {#subsec:ch4-multifactor}

单一谱能量指标难以同时覆盖拓扑角色与邻域多样性，故引入度中心性、k-core 与局部熵作为补充因子。下面给出在式 [\[eq:ch4-wk-hetero\]](#eq:ch4-wk-hetero){reference-type="eqref" reference="eq:ch4-wk-hetero"}单因子基础上的多因子可计算构造： $$\tau(i) = \mathrm{clip}\!\left(
    \frac{\tau_{\text{base}} \cdot \exp\!\left(-\beta_{\tau}\cdot E_{\text{spectral}}^{(h)}(i)/E_{\text{threshold}}\right)}
         {1 + \gamma\cdot C_{\text{deg}}(i) + \delta\cdot\dfrac{\text{core}(i)}{k_{\max}} + \eta\cdot H_{\text{norm}}(i)},\;
    \tau_{\min},\; \tau_{\text{base}}
  \right)
  \label{eq:ch4-alphadef}$$

其中 $k_{\max}=\max_i\text{core}(i)$，$E_{\text{threshold}}$ 与式 [\[eq:ch4-wk-hetero\]](#eq:ch4-wk-hetero){reference-type="eqref" reference="eq:ch4-wk-hetero"} 中定义相同，$H_{\text{norm}}(i)$ 沿用第3章定义 [\[def:ch3-entropy\]](#def:ch3-entropy){reference-type="ref" reference="def:ch3-entropy"} 的归一化形式，$\tau_{\min}=10^{-7}$。结构重要性越高 $\tau(i)$ 越小；参数默认值 $(\beta_{\tau},\gamma,\delta,\eta)=(1.0,0.5,0.3,0.2)$，敏感性见第 6 章。

其中归一化局部熵 $H_{\text{norm}}(i)$ 沿用第3章定义 [\[def:ch3-entropy\]](#def:ch3-entropy){reference-type="ref" reference="def:ch3-entropy"}（式 [\[eq:ch3-entropy\]](#eq:ch3-entropy){reference-type="eqref" reference="eq:ch3-entropy"}），高斯核带宽对应关系为 $\varsigma_x^2 = 1/\gamma_x = 2\hat{\sigma}^2$（见第3章算法 [\[alg:ch3-preprocess\]](#alg:ch3-preprocess){reference-type="ref" reference="alg:ch3-preprocess"} `gaussian`模式）。为避免与多因子融合权重 $\gamma$ 混淆，此处将核带宽单独记为 $\gamma_x$。对计算资源受限场景，可退化为单因子版本： $$\tau(i) = \tau_{\text{base}} \cdot \exp(-\beta_{\tau}\cdot E_{\text{spectral}}^{(h)}(i)/E_{\text{threshold}})
  \label{eq:ch4-propagation2}$$

::::: threeparttable
::: {#tab:ch4-complexity}
  **方案**                   **因子组成**                                                                                                  **节点类型**   **$\tau(i)$ 典型范围**
  -------------------------- ------------------------------------------------------------------------------------------------------------- -------------- -----------------------------
  均匀稀疏（baseline）       无自适应                                                                                                      全节点统一     $\tau_{\text{base}}$
  仅谱能量                   $E_{\text{spectral}}^{(h)}$                                                                                   高能枢纽       $\approx\tau_{\min}$
  谱能量+度中心性            $E_{\text{spectral}}^{(h)}, C_{\text{deg}}$                                                                   中等普通       中等
  **多因子融合（完整版）**   全部4因子（式 [\[eq:ch4-alphadef\]](#eq:ch4-alphadef){reference-type="eqref" reference="eq:ch4-alphadef"}）   低能叶节点     $\approx\tau_{\text{base}}$

  : 节点稀疏参数因子方案与节点类型对照（具体准确率见第 6 章消融实验）
:::

::: tablenotes
注：本表合并方案结构与节点类型对照，预处理复杂度 $O(mK_{\text{eig}})$，推理热路径 $O(n\bar{k}d)$ 不变。
:::
:::::

频率修正对节点稀疏参数的调制效应体现在：多因子构造中的 $E_{\text{spectral}}^{(h)}(i)$ 在异配图中以节点局部同配系数 $h_i$ 调制（第 [4.1.3](#subsec:ch4-espectralh){reference-type="ref" reference="subsec:ch4-espectralh"} 节，$w_k(i) \to w_k(h_i)$），使得频率维度自然内嵌于节点维度。频率修正的主要效应集中于中能节点------这类节点在异配图中往往处于社区边界，承载关键的高频跨类别信息；修正后 $E_{\text{spectral}}^{(h)}(i)$ 增大，$\tau(i)$ 减小，有效防止跨社区信息的过度稀疏。

多因子融合完成了稀疏度函数的"设计"阶段。下一小节给出完整的算法实现。

### 算法实现 {#subsec:ch4-tau-algo}

算法 [\[alg:ch4-tau-compute\]](#alg:ch4-tau-compute){reference-type="ref" reference="alg:ch4-tau-compute"}给出完整的批量计算流程，预计算阶段（Lanczos分解 $O(mK_{\text{eig}})$、k-core分解 $O(m)$）可离线一次性完成并跨轮次缓存；节点指标计算阶段总复杂度 $O(mK_{\text{eig}})$，支持GPU并行加速。

:::: algorithm
::: algorithmic
图 $G=(V,E)$，邻接矩阵 $A$，度矩阵 $\mathbf{D}$；特征矩阵 $X\in\mathbb{R}^{n\times F}$； 参数 $K_{\text{eig}}$（默认50），$\tau_{\text{base}}$（默认 $10^{-3}$），$\beta_{\mathrm{h}}=1.0$，$\beta_{\mathrm{f}}=1.0$，$\beta_{\tau}=1.0$，$\gamma=0.5$，$\delta=0.3$，$\eta=0.2$，$\tau_{\min}=10^{-7}$；谱能量阈值尺度 $c_E>0$（通过验证集设定，用于 $E_{\text{threshold}}$）； 边级同配率 $h_{\text{edge}}$（来自第3章第 [3.2.1](#subsec:ch3-homophily){reference-type="ref" reference="subsec:ch3-homophily"} 节）； 节点级同配率估计器 $\{h_i^{\text{est}}\}_{i\in V}$（仅异配分支使用，见第3章第 [3.2.1](#subsec:ch3-homophily){reference-type="ref" reference="subsec:ch3-homophily"} 节无泄漏口径） 节点级正则系数 $\{\tau(i)\}_{i\in V}$ **Step 1：谱分解（预计算，可离线进行）** $\tilde{\mathcal{L}} = I - \mathbf{D}^{-1/2}A\mathbf{D}^{-1/2}$ $(\Lambda, U) \leftarrow \text{Lanczos}(\tilde{\mathcal{L}}, K_{\text{eig}})$（调用第3章算法 [\[alg:ch3-lanczos\]](#alg:ch3-lanczos){reference-type="ref" reference="alg:ch3-lanczos"}） **Step 1.5：预计算全图 k-core（一次性，$O(m)$）** $\{\text{core}(i)\}_{i\in V} \leftarrow \text{k-core\_decomposition}(G)$；$k_{\max} \leftarrow \max_i\,\text{core}(i)$ 由 Lanczos 输出读取 $\hat{\lambda}_{\text{gap}}$、$K_{\text{pair}}$ 与 Ritz 特征值 $\{\lambda_k\}_{k=1}^{K_{\text{pair}}}$（口径同第3章算法 [\[alg:ch3-preprocess\]](#alg:ch3-preprocess){reference-type="ref" reference="alg:ch3-preprocess"}）；$\bar{\lambda} \leftarrow \frac{1}{K_{\text{pair}}}\sum_{k=1}^{K_{\text{pair}}}\lambda_k$；$E_{\text{threshold}} \leftarrow c_E\cdot\hat{\lambda}_{\text{gap}}/\bar{\lambda}$ **Step 2：逐节点指标计算** （1）修正谱能量： $h_i^{\text{est}} \leftarrow \text{HomophilyEstimate}(i)$ $\alpha_k(i) \leftarrow -\beta_{\mathrm{h}}(2h_i^{\text{est}}-1)\lambda_k$，$k=1,\ldots,K_{\text{eig}}$ *（同配图，全局同配率分支）* $\alpha_k(i) \leftarrow -\beta_{\mathrm{f}}\lambda_k$，$k=1,\ldots,K_{\text{eig}}$ $w_k(i) \leftarrow \frac{\exp(\alpha_k(i)-\max_{k'}\alpha_{k'}(i))}{\sum_{k''}\exp(\alpha_{k''}(i)-\max_{k'}\alpha_{k'}(i))}$（数值稳定 softmax） $E_{\text{spectral}}^{(h)}(i) \leftarrow \sum_{k=1}^{K_{\text{eig}}} w_k(i)\cdot|U_{ik}|^2\cdot\lambda_k$ （2）度中心性：$C_{\text{deg}}(i) \leftarrow d_i/(n-1)$ （3）k-core 归一化（直接读取）：$\text{core\_norm}(i) \leftarrow \text{core}(i)/k_{\max}$ （4）局部图熵（调用第3章算法 [\[alg:ch3-entropy\]](#alg:ch3-entropy){reference-type="ref" reference="alg:ch3-entropy"}，取归一化形式）：若 $|\mathcal{N}(i)|>1$ 则 $H_{\text{norm}}(i) \leftarrow H(i)/\ln|\mathcal{N}(i)|$，否则 $H_{\text{norm}}(i)\leftarrow 0$（与定义 [\[def:ch3-entropy\]](#def:ch3-entropy){reference-type="ref" reference="def:ch3-entropy"} 一致） （5）多因子融合（对应式 [\[eq:ch4-alphadef\]](#eq:ch4-alphadef){reference-type="eqref" reference="eq:ch4-alphadef"}）： $\tau_{\text{raw}} \leftarrow \tau_{\text{base}}\cdot\exp(-\beta_{\tau}\cdot E_{\text{spectral}}^{(h)}(i)/E_{\text{threshold}})$ $\tau_{\text{raw}} \leftarrow \tau_{\text{raw}}\cdot\left(1+\gamma\cdot C_{\text{deg}}(i)+\delta\cdot\text{core\_norm}(i)+\eta\cdot H_{\text{norm}}(i)\right)^{-1}$ （6）边界保护：$\tau(i) \leftarrow \mathrm{clip}(\tau_{\text{raw}},\;\tau_{\min},\;\tau_{\text{base}})$ $\{\tau(i)\}_{i=1}^n$
:::
::::

算法输出的 $\{\tau(i)\}_{i\in V}$ 直接进入第5章第5.1节LARS求解器，clip 操作保证数值稳定性（$\tau(i)\in[\tau_{\min},\tau_{\text{base}}]$）。

### 对比学习增强的节点稀疏参数学习 {#subsec:ch4-grace}

算法 [\[alg:ch4-tau-compute\]](#alg:ch4-tau-compute){reference-type="ref" reference="alg:ch4-tau-compute"}输出的基线 $\tau^{(0)}(i)$ 仅依赖显式谱拓扑特征，无法捕获三角形密度、社区归属等隐式高阶结构模式。为此，本节引入对比学习增强路径作为可选扩展：以GRACE[@Zhu2020GRACE]风格的图数据增强（边掩码与特征掩码各独立采样两份视图）预训练共享参数GNN编码器 $f_\omega$，通过对称 InfoNCE 损失（两个视图互为正负样本，双向平均） $$\mathcal{L}_{\text{NCE}} = -\frac{1}{2n}\sum_{i=1}^{n}\left[\log\frac{\exp(z_i^{(1)\top}z_i^{(2)}/t)}{\sum_{j=1}^n\exp(z_i^{(1)\top}z_j^{(2)}/t)} + \log\frac{\exp(z_i^{(2)\top}z_i^{(1)}/t)}{\sum_{j=1}^n\exp(z_i^{(2)\top}z_j^{(1)}/t)}\right]
  \label{eq:ch4-infonce}$$

训练编码器获得对比嵌入 $z_i = f_\omega(G)_i$；再由残差MLP $g_\tau$ 在对数空间对基线做有界修正：$\tau(i) = \mathrm{clip}(\tau^{(0)}(i)\cdot\exp(g_\tau([z_i;\,s_i])),\,\tau_{\min},\,\tau_{\text{base}})$，其中 $s_i$ 为算法 [\[alg:ch4-tau-compute\]](#alg:ch4-tau-compute){reference-type="ref" reference="alg:ch4-tau-compute"}输出的四维手工特征。当修正量为零时退化为纯手工路径；对比预训练为离线一次性成本，推理热路径复杂度不变。手工特征提供显式谱扰动约束，对比嵌入补充隐式高阶灵活性，两路互补。为保持第5章主算法接口唯一，正文默认采用算法 [\[alg:ch4-tau-compute\]](#alg:ch4-tau-compute){reference-type="ref" reference="alg:ch4-tau-compute"}输出的 $\tau(i)$ 进入第5章 LARS 主循环；对比学习增强仅作为第6章消融验证的附加变体，不并入默认主线。定量差异见第 6 章相关实验。

完整三阶段流程（对比预训练→原图嵌入→残差修正推断）详见附录B.1"对比学习增强 $\tau(i)$ 的完整三阶段流程"。

$\tau(i)$ 的完整设计链（谱扰动启发→多因子可计算构造→对比增强可选路径）至此完备，其输出与跳距维度产出 $k_i^{(l)}$ 共同进入第5章LARS主循环；总预算在跳距间的最优分配问题由第 [4.3](#sec:ch4-hop){reference-type="ref" reference="sec:ch4-hop"} 节的衰减界分析与最优分配推导解决。

## 层级维度：分层跳距预算分配 {#sec:ch4-hop}

[]{#idx:theme:hopbudget label="idx:theme:hopbudget"}

节点维度的 $\tau(i)$ 确定了总稀疏预算 $k_i^{\text{total}}$，但未规定该预算如何在不同跳距邻居之间分配。直接邻居携带最直接的局部结构信息，远跳邻居信息增益随跳距递减；均匀分配会使近邻保留不足，完全集中近邻又损失多尺度覆盖。本节提出分层跳距预算分配策略，通过保留率 $p_l\in(0,1]$ 决定第 $l$ 跳的整数配额 $k_i^{(l)}$，保持 $\tau(i)$ 全章唯一，不引入逐层修正项。

![层级维度------分层跳距预算分配示意图](4-4.pdf){#fig:ch4-speedup width="100%"}

### 分层信息衰减界分析 {#subsec:ch4-lemma36}

上述分层预算分配方案确立后，核心问题转化为：各跳距预算应按何种比例分配？在本文采用的线性化传播近似下，边际信息增益随传播次数增加呈衰减趋势，因此近跳邻居通常应获得更高配额。本节在谱理论框架内对这一设计直觉给出定量刻画。

设 $S = D^{-1/2}AD^{-1/2}$ 为对称归一化传播算子，其特征值 $\mu_k(S)$ 满足 $|\mu_k(S)| \leq 1$（Perron-Frobenius定理保证 $\mu_1(S)=1$）。定义**第二谱半径**： $$\mu_\star(S) := \max_{k \geq 2}|\mu_k(S)|
  \label{eq:ch4-tau-step1}$$

在非二分图中 $\mu_\star(S) < 1$，传播算子 $S^l$ 在去除平凡特征方向后以 $\mu_\star(S)^l$ 的几何速率收缩；对含二分结构的图改用懒化算子 $S_{\text{lazy}}=(I+S)/2$，严格保证 $\mu_\star(S_{\text{lazy}})\in[0,1)$。

[]{#lem:ch4-decay label="lem:ch4-decay"}[]{#idx:theme:infodecay label="idx:theme:infodecay"} 在本文的线性化传播模型下，浅层邻域的边际信息增益上界更高，深层邻域的新增贡献相对更弱，这为第 [4.3.2](#subsec:ch4-plstrategy){reference-type="ref" reference="subsec:ch4-plstrategy"} 节的分层预算设计提供动机；全章 $\tau(i)$ 保持唯一定义，不引入分层正则系数。

以下分析在线性化传播假设下建立分层信息衰减界，以此作为各跳距预算配额的理论依据。该假设令 $h_i^{[l]} = (P^l X)_{i,*}$，等价于去除非线性激活与参数矩阵的 SGC 传播模型[@Wu2019SGC]；非线性情形下各层信息增益的传递需额外的 Lipschitz 常数修正，因此式 [\[eq:ch4-tau-step2\]](#eq:ch4-tau-step2){reference-type="eqref" reference="eq:ch4-tau-step2"}在非线性场景中应理解为启发式衰减模型而非严格定理。设传播算子为： $$P = \begin{cases} S, & \text{若图非二分且 } \mu_\star(S) < 1, \\ S_{\mathrm{lazy}} = (I+S)/2, & \text{若图含二分或近二分结构；} \end{cases}$$

记 $h_i^{[l]} = (P^l X)_{i,*}$ 为节点 $i$ 的 $l$ 次传播累计表示，分析对象为由第 $l-1$ 次提升至第 $l$ 次传播时的边际表示增益，作为第 $l$ 跳环层预算配额的理论替代依据，而非对 $\mathcal{R}_l(i)$ 的精确孤立分解。可以证明，第 $l$ 次传播相对于第 $l-1$ 次传播的边际互信息增益满足： $$I(y_i;\, h_i^{[l]}) - I(y_i;\, h_i^{[l-1]}) \leq C_1 \cdot \mu_\star(P)^{l-1}
  \label{eq:ch4-tau-step2}$$

其中 $C_1 > 0$ 为仅依赖图结构与特征分布的常数。推导思路如下：第 $l$ 次传播在第 $l-1$ 次基础上引入的表示增量为 $$\Delta h_i^{[l]} = h_i^{[l]} - h_i^{[l-1]} = e_i^\top (P^l - P^{l-1}) X,
  \label{eq:ch4-tau-step3}$$

其中 $e_i \in \mathbb{R}^n$ 为第 $i$ 个标准基向量。在与常数向量 $\mathbf{1}$ 正交的子空间 $\mathbf{1}^\perp$ 上，$\|P^l v\|_2 \leq \mu_\star(P)^l \|v\|_2$（完整推导见附录A.4.1"谱收缩不等式"），从而 $\|\Delta h_i^{[l]}\|_2 \leq 2\mu_\star(P)^{l-1}\|X\|_F$。在附加的高斯观测近似与有限能量假设下，互信息增量可由信号能量作上界控制，即 $\Delta I_l \leq C\|\Delta h_i^{[l]}\|_2^2 \leq C_1'\mu_\star(P)^{2(l-1)}$；由于 $\mu_\star(P)^{2(l-1)} \leq \mu_\star(P)^{l-1}$，进一步得到更简洁的式 [\[eq:ch4-tau-step2\]](#eq:ch4-tau-step2){reference-type="eqref" reference="eq:ch4-tau-step2"}。

[]{#thm:ch4-decay label="thm:ch4-decay"} 基于式 [\[eq:ch4-tau-step2\]](#eq:ch4-tau-step2){reference-type="eqref" reference="eq:ch4-tau-step2"}的衰减分析，可进一步得到保留率的理论下界。若第 $l$ 次传播允许 $\varepsilon_l$ 的信息损失，则该跳保留率应满足： $$p_l^* \geq \max\!\left\{0,\; 1 - \frac{\varepsilon_l}{C_1 \mu_\star(P)^{l-1}}\right\}
  \label{eq:ch4-tau-step4}$$

当 $l$ 增大时 $\mu_\star(P)^{l-1}$ 趋向0，下界随之趋向0，意味着高跳距在理论上允许近乎完全稀疏。实践中以 $p_l \geq p_{\min} = 0.1$ 保证最低保留率，防止局部节点孤立（该下界与算法 [\[alg:ch4-tau-compute\]](#alg:ch4-tau-compute){reference-type="ref" reference="alg:ch4-tau-compute"}的边界保护步骤协同作用，但不保证全图连通性）。式 [\[eq:ch4-tau-step2\]](#eq:ch4-tau-step2){reference-type="eqref" reference="eq:ch4-tau-step2"}的衰减形式因此为两种 $p_l$ 计算策略奠定了共同的理论基础------设计目标正是使 $p_l$ 与上述最优保留率下界形式近似匹配，具体方案见第 [4.3.2](#subsec:ch4-plstrategy){reference-type="ref" reference="subsec:ch4-plstrategy"} 节。

![跳距预算衰减分配示意图（策略1 vs 策略2）](4-5.pdf){#fig:ch4-hop-decay width="100%"}

### 分层保留率 $p_l$ 的两种策略 {#subsec:ch4-plstrategy}

由第 [4.3.1](#subsec:ch4-lemma36){reference-type="ref" reference="subsec:ch4-lemma36"} 节的衰减界分析，$p_l$ 应随跳距 $l$ 单调递减，本节提出两种计算策略。

均匀衰减方案（策略1）取指数衰减形式，公比 $r=(p_L/p_1)^{1/(L-1)}$ 由端点保留率唯一确定： $$p_l = p_1 \cdot r^{l-1}, \quad r \in (0,1)
  \label{eq:ch4-tau-step5}$$

典型取值为 $L=3$ 时 $(p_1,p_2,p_3)=(0.60,0.40,0.20)$。谱间隙自适应方案（策略2）等价于取 $r=\exp(-\kappa\hat{\lambda}_{\text{gap}})$ 的自动参数化版本，谱间隙越大衰减越快，无需手动设置公比： $$p_l = \max\!\left\{p_{\min},\; p_{\text{base}} \exp\!\bigl(-\kappa \hat{\lambda}_{\text{gap}}(l-1)\bigr)\right\}, \quad \kappa > 0
  \label{eq:ch4-tau-step6}$$

其中 $p_{\text{base}}=0.6$，$p_{\min}=0.1$。两种策略的系统对比见表 [4.3](#tab:ch4-3-9){reference-type="ref" reference="tab:ch4-3-9"}，

:::: threeparttable
::: {#tab:ch4-3-9}
  **对比维度**   **策略1（均匀衰减）**                                                                                                 **策略2（谱间隙自适应）**
  -------------- --------------------------------------------------------------------------------------------------------------------- -----------------------------------------------------------------------------------------
  参数数量       2个（$p_1, p_L$）                                                                                                     1个核心参数（$\kappa$；若 $p_{\text{base}},p_{\min}$ 采用默认值）
  图自适应性     低（固定公比$r$）                                                                                                     高（自动适配$\hat{\lambda}_{\text{gap}}$）
  计算额外开销   零（参数预设）                                                                                                        低（$\hat{\lambda}_{\text{gap}}$ 由第3章预处理阶段给出，或可由 Lanczos 特征值顺带计算）
  适用场景       $\hat{\lambda}_{\text{gap}}$ 稳定的均匀数据集                                                                         $\hat{\lambda}_{\text{gap}}$ 差异大的异质图
  稳健性         高（参数范围有保证）                                                                                                  高（指数形式天然保证单调递减）
  理论依据       第 [4.3.1](#subsec:ch4-lemma36){reference-type="ref" reference="subsec:ch4-lemma36"} 节衰减界分析得出的几何递减结论   以$\hat{\lambda}_{\text{gap}}$ 为代理变量的谱间隙启发式参数化

  : 策略1与策略2的对比分析
:::
::::

在策略选择上，引文网络（Cora/Citeseer，$\hat{\lambda}_{\text{gap}}$ 相对稳定）推荐策略1；社交图/Web图（如 Chameleon、Squirrel 一类异质图）更推荐策略2；大规模图（Ogbn-arxiv/Papers100M）及 $L \geq 4$ 的深层GNN优先策略2，可复用第3章预处理得到的 $\hat{\lambda}_{\text{gap}}$（与算法 [\[alg:ch4-tau-compute\]](#alg:ch4-tau-compute){reference-type="ref" reference="alg:ch4-tau-compute"} 中 Lanczos 输出一致）而无需额外谱计算。两种策略提供了 $p_l$ 的具体数值，下一节将在此基础上给出节点总预算到各跳距配额的分配策略。

### 节点-跳距联合预算分配策略 {#subsec:ch4-theorem33}

[]{#idx:innov:joint_budget label="idx:innov:joint_budget"}

节点总预算 $K_i=k_i^{\text{total}}$ 由 $\tau(i)$ 隐式确定（LARS路径截断），$p_l$ 的具体值由第 [4.3.2](#subsec:ch4-plstrategy){reference-type="ref" reference="subsec:ch4-plstrategy"} 节给出。在"边际逼近误差随分配预算近似按 $1/k_i^{(l)}$ 衰减"的替代假设下，可将节点总预算到各跳距配额的分配写为连续松弛形式 $$\min_{\{k_i^{(l)}\}_{l=1}^L}\ \sum_{l=1}^L \frac{p_l}{k_i^{(l)}}\quad \text{s.t.}\quad \sum_{l=1}^L k_i^{(l)} = K_i,\ k_i^{(l)}>0.
  \label{eq:ch4-tau-formula}$$

[]{#thm:ch4-budget-alloc label="thm:ch4-budget-alloc"}[]{#idx:innov:hopdist label="idx:innov:hopdist"} 在"边际逼近误差随分配预算近似按 $1/k_i^{(l)}$ 衰减"的替代假设下，可得到一类以 $\sqrt{p_l}$ 为比例参考的分配方案（$\xi=1/2$ 版本）： $$k_i^{*(l)} = K_i \cdot \frac{\sqrt{p_l}}{\sum_{j=1}^L \sqrt{p_j}}
  \label{eq:ch4-tau-approx}$$

对应的整数预算配额通过配额舍入算子 $\operatorname{RoundQuota}(\cdot)$ 实现：先取下整，再按最大小数部分依次补足剩余预算，严格保证 $\sum_l k_i^{(l)} = K_i$（若有容量约束 $k_i^{(l)} \leq |\mathcal{R}_l(i)|$，对已饱和环层做跳过处理）： $$\tilde{k}_i^{(l)} = K_i \cdot \frac{\sqrt{p_l}}{\sum_{j=1}^L \sqrt{p_j}}, \qquad k_i^{(l)} = \operatorname{RoundQuota}\!\bigl(\{\tilde{k}_i^{(l)}\}_{l=1}^L,\; K_i\bigr)
  \label{eq:ch4-tau-re}$$

在"层间保留率差异适中"的条件下（$p_{\max}/p_{\min} \leq 2$），利用 $\sqrt{p_l}/\sum_j\sqrt{p_j} \approx p_l/\sum_j p_j$ 的近似关系，精确解可退化为工程近似（$\xi=1$ 版本）： $$\tilde{k}_i^{(l)} = K_i \cdot \frac{p_l}{\sum_{j=1}^L p_j}, \qquad k_i^{(l)} = \operatorname{RoundQuota}\!\bigl(\{\tilde{k}_i^{(l)}\}_{l=1}^L,\; K_i\bigr)
  \label{eq:ch4-tau-fusion}$$

工程近似仅在层间保留率差异适中时（$p_{\max}/p_{\min} \leq 2$）作为默认选择；若采用 $(p_1,p_2,p_3)=(0.6,0.4,0.2)$ 这类跨度较大的配置（$p_{\max}/p_{\min}=3$），则优先使用精确公式 [\[eq:ch4-tau-re\]](#eq:ch4-tau-re){reference-type="eqref" reference="eq:ch4-tau-re"}。该策略的理论依据说明见附录A.5；两版端到端性能差异的实测数据见第 6 章消融实验。需要说明的是，$\operatorname{RoundQuota}$ 保证 $\sum_l k_i^{(l)} = k_i^{\text{total}}$，近似误差约 $(r-1)/(2\sqrt{r})$；$\tau(i)$ 确定总预算与 $p_l$ 分割总预算两步相互独立，可分别优化。

三类节点的协同效果体现在：高谱能量Hub节点（$\tau(i)\approx10^{-7}$--$10^{-6}$，$K_i$约40--60）在各跳距均保留较多邻居；低谱能量叶子节点（$\tau(i)\approx10^{-4}$--$10^{-3}$，$K_i$约4--8）总预算小，叠加深跳距权重偏低，第3跳处可能被分配0条邻居（按容量约束自动归零，预算补偿到前序跳距）。节点维与跳距维相互独立地发挥作用，共同保证结构重要节点在全跳距范围内的信息完整性；具体数值以第 6 章相关实验为准。

跳距维度的自适应设计至此完成。第 [4.3.1](#subsec:ch4-lemma36){reference-type="ref" reference="subsec:ch4-lemma36"} 节的衰减界分析以 $\mu_\star(P)$ 为衰减速率刻画了边际信息增益的几何递减规律；在此基础上，本文给出以 $k_i^{(l)}\propto\sqrt{p_l}$ 为代表的分配参考形式，并保留 $k_i^{(l)}\propto p_l$ 的工程近似版本（适用于 $p_{\max}/p_{\min}\leq 2$ 的场景）。至此频率、节点、跳距三维参数体系完整建立，第5章将以LARS统一整合三维输出。

## 本章小结 {#sec:ch4-summary}

本章以第3章四类图复杂度指标为输入，完成了"频率×节点×跳距"三维自适应稀疏参数体系的完整设计，取代了SDGNN的全局统一正则系数 $\lambda_{\mathrm{reg}}$。

频率维度（第 [4.1](#sec:ch4-freq){reference-type="ref" reference="sec:ch4-freq"} 节）依据同配率与类别信号谱位置之间的近似关联，构造分段softmax频率权重（式 [\[eq:ch4-lambda-y\]](#eq:ch4-lambda-y){reference-type="eqref" reference="eq:ch4-lambda-y"}），在 $h_{\text{edge}}\geq 0.5$ 时退化为标准低频偏重，在 $h_{\text{edge}}<0.5$ 时以节点级 $h_i$ 做局部修正；统一控制变量 $h^\dagger(i)$ 封装两种分支，最终定义修正谱能量 $E_{\text{spectral}}^{(h)}(i)$（式 [\[eq:ch4-espectralh-def\]](#eq:ch4-espectralh-def){reference-type="eqref" reference="eq:ch4-espectralh-def"}）。节点维度（第 [4.2](#sec:ch4-node){reference-type="ref" reference="sec:ch4-node"} 节）依据谱扰动分析的基本结论（参见附录A.3），给出以 $E_{\text{spectral}}^{(h)}(i)$ 为自变量的单调递减设计准则（式 [\[eq:ch4-wk-hetero\]](#eq:ch4-wk-hetero){reference-type="eqref" reference="eq:ch4-wk-hetero"}），并以式 [\[eq:ch4-alphadef\]](#eq:ch4-alphadef){reference-type="eqref" reference="eq:ch4-alphadef"}的 clip 形式整合度中心性、k-core、局部熵三类补充因子，算法实现见算法 [\[alg:ch4-tau-compute\]](#alg:ch4-tau-compute){reference-type="ref" reference="alg:ch4-tau-compute"}，对比学习增强路径作为第6章消融变体（第 [4.2.4](#subsec:ch4-grace){reference-type="ref" reference="subsec:ch4-grace"} 节），不并入第5章默认主线。跳距维度（第 [4.3](#sec:ch4-hop){reference-type="ref" reference="sec:ch4-hop"} 节）以第 [4.3.1](#subsec:ch4-lemma36){reference-type="ref" reference="subsec:ch4-lemma36"} 节的几何衰减界为依据，给出以 $k_i^{(l)}\propto\sqrt{p_l}$（式 [\[eq:ch4-tau-approx\]](#eq:ch4-tau-approx){reference-type="eqref" reference="eq:ch4-tau-approx"}）为代表的分配参考形式，并保留 $k_i^{(l)}\propto p_l$（工程近似，式 [\[eq:ch4-tau-fusion\]](#eq:ch4-tau-fusion){reference-type="eqref" reference="eq:ch4-tau-fusion"}）这一更便于实现的版本。

三个维度的核心输出构成完整接口：修正谱能量 $E_{\text{spectral}}^{(h)}(i)$（内嵌于 $w_k(i)$）通过 $\tau(i)$ 间接影响第5章LARS路径截断强度；节点级正则系数 $\tau(i)$ 直接进入第5章 Phase $\Theta$ 的节点级惩罚项；各跳距预算配额 $k_i^{(l)}$ 用于第5章候选集构造与预算截断模块。$\tau(i)$ 与 $k_i^{(l)}$ 独立设计，可分别优化，在第5章的LARS分层求解流程中统一整合。

三维参数体系的整合形成了"全局预算守恒、节点异质、跳距衰减感知"的完整自适应机制。然而，三维参数目前逻辑独立，尚缺乏统一的求解算法将其端到端整合；$\tau(i)$ 与 $k_i^{(l)}$ 对全局谱相似性的聚合保证也有待建立。第5章将以LARS为核心求解器，通过分层候选集构造实现三维参数的物理整合，并给出稀疏化前后的误差分析与复杂度分析接口。

# 算法整合、误差分析与复杂度分析 {#chap:algorithm-theory}

第4章给出三维自适应参数------频率权重 $w_k(i)$、节点正则系数 $\tau(i)$、跳距预算配额 $k_i^{(l)}$------各维度独立设计，尚未形成统一的可执行流程。本章在此基础上完成最后三步：以LARS为核心求解器将三维参数整合为端到端交替优化算法（§[5.1](#sec:ch5-lars){reference-type="ref" reference="sec:ch5-lars"}）；通过$\sigma$-谱相似性给出稀疏化后的误差分析与适用条件（§[5.2](#sec:ch5-spectral){reference-type="ref" reference="sec:ch5-spectral"}）；分析推理与训练各阶段复杂度并说明成立条件（§[5.3](#sec:ch5-time){reference-type="ref" reference="sec:ch5-time"}）。

本章整体框架如图 [5.1](#fig:ch5-overview){reference-type="ref" reference="fig:ch5-overview"} 所示。

![第5章整体框架图：三维参数整合→LARS算法→误差分析→复杂度](5-1.pdf){#fig:ch5-overview width="100%"}

## LARS分层稀疏求解与交替优化 {#sec:ch5-lars}

第4章三类自适应参数（$w_k(i)$、$\tau(i)$、$k_i^{(l)}$）目前各自独立，尚未形成统一求解流程。本节以LARS[@Efron2004LARS]（Efron等，2004）为核心求解器：LARS沿Lasso解路径生成分段线性的稀疏解序列，以$\tau(i)$为正则系数，令 `max_iter` 取 $k_i^{(l)}$ 作为预算截断上限，从而将三维参数直接落地为求解器接口。具体地，候选集按跳距划分为环层 $\mathcal{R}_l(i)$（第 [5.1.1](#subsec:ch5-candidateset){reference-type="ref" reference="subsec:ch5-candidateset"} 节），各层顺序执行残差级联LARS；完整端到端流程见第 [5.1.2](#subsec:ch5-mainloop){reference-type="ref" reference="subsec:ch5-mainloop"} 节算法 [\[alg:ch5-main\]](#alg:ch5-main){reference-type="ref" reference="alg:ch5-main"}，收敛性分析见第 [5.1.3](#subsec:ch5-convergence){reference-type="ref" reference="subsec:ch5-convergence"} 节。

### 分层LARS的候选集构造与方案选定 {#subsec:ch5-candidateset}

候选集定义为前$L$跳邻居并集$C_i=\bigcup_{l=1}^{L}\mathcal{R}_l(i)$（环层$\mathcal{R}_l(i)$两两不相交）。大图训练时可采用"前 $K_{\mathrm{keep}}$ 跳完整保留 + 第 $K_{\mathrm{sample}}$ 跳起分层采样"的工程裁剪策略，其中 $K_{\mathrm{keep}}$ 与 $K_{\mathrm{sample}}$ 为预先设定的裁剪超参数。

作为对照，可先考虑共享目标方案：记 $\Omega_i:=H_i^*$ 为教师/基准 GNN 在节点 $i$ 处的目标表示，最直接的做法是对每个环层 $\mathcal{R}_l(i)$ 各自独立求解： $$\theta_i^{(l)} = \operatorname*{arg\,min}_{\theta\in\mathbb{R}^{|\mathcal{R}_l(i)|}}
\left\|\theta^\top\tilde{\Phi}_{\mathcal{R}_l(i)} - \Omega_i\right\|^2 + \tau(i)\|\theta\|_1
\label{eq:ch5-shared-target}$$

每层均以同一 $\Omega_i$ 为拟合对象，存在对同一目标的重复逼近与信息冗余，故本文不采用该路径（以下称方案 1）。

方案2（残差级联）将各层拟合对象改为逐步更新的残差，对应子问题为 $$\theta_i^{(l)} = \operatorname*{arg\,min}_{\theta\in\mathbb{R}^{|\mathcal{R}_l(i)|}}
\left\|\theta^\top\tilde{\Phi}_{\mathcal{R}_l(i)} - r_i^{(l-1)}\right\|^2 + \tau(i)\|\theta\|_1
  \label{eq:ch5-tau-full}$$

实现上令 `max_iter` 取 $k_i^{(l)}$ 作为截断上限，并按 $r_i^{(l)}:=r_i^{(l-1)}-(\theta_i^{(l)})^\top\tilde{\Phi}_{\mathcal{R}_l(i)}$ 更新残差，最终拼接得到 $\theta_i$。与方案1相比，方案2以"逐层拟合残差"替代"逐层重复逼近同一目标"，因此更符合第4章第 [4.3](#sec:ch4-hop){reference-type="ref" reference="sec:ch4-hop"} 节所体现的"近跳优先、远跳递减"预算动机。需要说明的是，这里仅给出工程合理性解释，不将其表述为整体 Lasso 支撑集一致性的严格结论。故本文最终采用方案2。

![共享目标与残差级联两种分层 LARS 方案对比](5-2.pdf){#fig:ch5-lars-compare width="100%"}

### 完整交替优化算法 {#subsec:ch5-mainloop}

$\tau(i)$ 默认由手工特征路径（算法 [\[alg:ch4-tau-compute\]](#alg:ch4-tau-compute){reference-type="ref" reference="alg:ch4-tau-compute"}）生成；若采用第 [4.2.4](#subsec:ch4-grace){reference-type="ref" reference="subsec:ch4-grace"} 节的对比学习增强变体，则以其输出替换默认 $\tau(i)$。两条路径共享同一数值范围 $[\tau_{\min},\tau_{\text{base}}]$，因此第5章主循环接口保持一致。

:::: algorithm
::: algorithmic
图 $G=(V,E,X)$，$X\in\mathbb{R}^{n\times D}$；教师/基准 GNN 离线输出的目标中间表示矩阵 $H^*\in\mathbb{R}^{n\times d}$；外部给定的节点级正则系数 $\{\tau(i)\}_{i\in V}$（来自第4章第4.2节）；分层预算配额 $\{k_i^{(l)}\}$（优先由第4章第4.3节预计算并输入；若缺省则需同时给定$\lambda_{\text{gap}}$与$\{k_i^{\text{total}}\}$以便按第4章规则在线生成）；预训练参数 $W_\phi^*$；超参数 $L$，$T_{\text{epoch}}$，$T_W$，$\eta_W$，$q$ 稀疏权重矩阵 $\Theta=\{\theta_i\}_{i\in V}$，特征变换参数 $W_\phi$ **阶段一：候选集与环层构建（一次性预处理）** $\mathcal{R}_l(i) \leftarrow \{j\in V \mid d_G(i,j)=l\}$，$l=1,\ldots,L$（按跳距分环层） $C_i \leftarrow \bigcup_{l=1}^{L}\mathcal{R}_l(i)$（保留完整 $L$-hop 邻域） **阶段二（可选）：分层跳距预算在线生成** 若输入中已直接给出 $\{k_i^{(l)}\}$，则跳过本阶段；否则按第4章第4.3节的分配规则，由$\lambda_{\text{gap}}$与$\{k_i^{\text{total}}\}$在线生成 $\{k_i^{(l)}\}$。 **阶段三：full-batch 交替优化** $W_\phi \leftarrow W_\phi^*$；$\Omega \leftarrow H^*$；$\Theta \leftarrow 0$（$\Theta\in\mathbb{R}^{n\times n}$，与 $\tilde{\Phi}\in\mathbb{R}^{n\times d}$ 配套，使 $\hat{H}=\Theta^\top\tilde{\Phi}$ 为全图节点表示） $\tilde{\Phi} \leftarrow \ell_2\text{-row-normalize}\!\bigl(\phi(X;W_\phi)\bigr)$（若 $\phi$ 为单层线性变换，则该步复杂度为 $O(nDd)$） **Phase $\Theta$（固定 $W_\phi$，对全体节点执行分层 LARS）：** $r_i^{(0)} \leftarrow \Omega_i$ $\theta_i^{[l]} \leftarrow \operatorname{LARS}\!\!\left(\min_\theta\left\|\theta^\top\tilde{\Phi}[\mathcal{R}_l(i),:]-r_i^{(l-1)}\right\|_2^2+\tau(i)\|\theta\|_1\right)\!\Big|_{\mathrm{max\_iter}=k_i^{(l)}}$ $r_i^{(l)} \leftarrow r_i^{(l-1)}-(\theta_i^{[l]})^\top\tilde{\Phi}[\mathcal{R}_l(i),:]$ $\theta_{i,C_i} \leftarrow \operatorname{Concat}(\theta_i^{[1]},\ldots,\theta_i^{[L]})$ $\Theta[C_i,i] \leftarrow \theta_{i,C_i}$；第 $i$ 列在行索引 $j\notin C_i$ 处置 $0$ **Phase $W$（固定 $\Theta$，优化 $W_\phi$）：** $\tilde{\Phi} \leftarrow \ell_2\text{-row-normalize}\!\bigl(\phi(X;W_\phi)\bigr)$；$\hat{H} \leftarrow \Theta^\top\tilde{\Phi}$ $W_\phi \leftarrow \operatorname{AdamStep}\!\left(W_\phi,\;\nabla_{W_\phi}\tfrac{1}{2}\left\|\hat{H}-\Omega\right\|_F^2;\,\eta_W\right)$ **break** $\Theta,\;W_\phi$
:::
::::

其中 $\hat{H}=\Theta^\top\tilde{\Phi}\in\mathbb{R}^{n\times d}$ 为全图稀疏近似表示矩阵。由于节点 $i$ 的稀疏系数写入 $\Theta$ 的第 $i$ 列，$\Theta^\top\tilde{\Phi}$ 的第 $i$ 行恰对应节点 $i$ 的近似表示；Phase $W$ 优化全图表示逼近目标 $\frac{1}{2}\|\Theta^\top\tilde{\Phi}-H^*\|_F^2$，保证 $\Theta$ 与 $W_\phi$ 每轮全图同步。

:::: algorithm
::: algorithmic
训练完成的稀疏权重 $\Theta=\{\theta_i\}_{i\in V}$，特征变换参数 $W_\phi$，图 $G=(V,E,X)$ 固化稀疏权重 $\Theta^{\text{fixed}}$，缓存归一化特征矩阵 $\tilde{\Phi}^{\text{cache}}$（第6章实验接口） $\Theta^{\text{fixed}} \leftarrow \Theta$（冻结训练所得稀疏权重，推理期不再更新） $\tilde{\Phi}^{\text{cache}} \leftarrow \ell_2\text{-row-normalize}\!\left(\phi(X;\,W_\phi)\right)$（若 $\phi$ 为单层线性变换则复杂度为 $O(nDd)$） $\Theta^{\text{fixed}},\;\tilde{\Phi}^{\text{cache}}$
:::
::::

两个算法共享同一预算接口：$\tau(i)$与$\{k_i^{(l)}\}$均作为流程输入，其中算法 [\[alg:ch5-main\]](#alg:ch5-main){reference-type="ref" reference="alg:ch5-main"}消费二者，算法 [\[alg:ch5-fixed\]](#alg:ch5-fixed){reference-type="ref" reference="alg:ch5-fixed"}仅输出$\Theta^{\text{fixed}}$与$\tilde{\Phi}^{\text{cache}}$。

固化后的推理过程为单次稀疏矩阵乘： $$\hat{H} = (\Theta^{\text{fixed}})^\top \tilde{\Phi}^{\text{cache}},\quad \text{复杂度 }O(m_{\text{sparse}}\,d) = O(n\bar{k}d)
  \label{eq:ch5-tau-range}$$

其中 $m_{\text{sparse}}=\sum_{i\in V}\|\theta_i\|_0=n\bar{k}$ 为全图非零权重总数，$\bar{k}:=\frac{1}{n}\sum_{i\in V}k_i^{\text{total}}$ 为各节点总预算的全图平均值。各阶段完整复杂度口径见第 [5.3](#sec:ch5-time){reference-type="ref" reference="sec:ch5-time"} 节。

### 优化稳定性与实现说明 {#subsec:ch5-convergence}

固定$W_\phi$时，每个环层子问题都对应一个带$\ell_1$正则的凸代理问题；若对该子问题完整执行LARS，则可获得其分段线性路径上的精确解。当前Phase $\Theta$还叠加了候选集限制、残差级联与`max_iter`截断，因此整阶段应视为分层近似求解，而不宜表述为"精确追踪整体Lasso解路径"。对外层交替优化，本文仅声称经验稳定，不声称全局收敛；相关现象将在第 6 章训练曲线中验证。三维参数至此整合为端到端优化闭环，固化后推理接口（式 [\[eq:ch5-tau-range\]](#eq:ch5-tau-range){reference-type="eqref" reference="eq:ch5-tau-range"}）直接进入第 [5.2](#sec:ch5-spectral){reference-type="ref" reference="sec:ch5-spectral"} 节的误差分析。

## 误差分析与适用条件 {#sec:ch5-spectral}

[]{#idx:theme:specpreserve label="idx:theme:specpreserve"}

第4章各维度设计均属启发式或条件性设计，尚未定量说明稀疏化后的误差范围。本节以$\sigma$-谱相似性为度量工具，给出多项式谱滤波器的近似误差界，并说明其适用条件。

### 谱相似性与图稀疏化 {#subsec:ch5-spectralsim}

稀疏化将原图$G$替换为边数更少的$\hat{G}$，为量化谱结构破坏程度，引入以下定义。

[]{#def:ch5-sigmasim label="def:ch5-sigmasim"} 为量化稀疏子图对原图谱结构的保持程度，本文引入**$\sigma$-谱相似性**（$\sigma$-spectral similarity）的概念。设 $G$ 与 $\hat{G}$ 为定义在同一节点集 $V$ 上的两个图，其归一化拉普拉斯分别为 $\tilde{\mathcal{L}}_G$ 与 $\tilde{\mathcal{L}}_{\hat{G}}$。若对任意 $x\in\mathbb{R}^n$ 满足： $$\frac{1}{\sigma}\,x^\top\tilde{\mathcal{L}}_G x
\;\leq\; x^\top\tilde{\mathcal{L}}_{\hat{G}}x
\;\leq\; \sigma\, x^\top\tilde{\mathcal{L}}_G x,
  \label{eq:ch5-tau-mono}$$

则称 $\hat{G}$ 是 $G$ 的 $\sigma$-谱稀疏子图，参数 $\sigma\geq 1$，越接近1谱保持越精确。$\sigma$ 可由精确特征分解（$O(n^3)$，适用于小图）或 Lanczos 代理量 $\widetilde{\sigma}_{\mathrm{proxy}}$ 估计；代理量与严格 $\sigma$ 的偏差在第6章实验中统一报告。

### 多项式谱滤波器的近似误差界 {#subsec:ch5-polyfilter}

以下建立多项式谱滤波器的近似误差界（式 [\[eq:ch5-tau-grace\]](#eq:ch5-tau-grace){reference-type="eqref" reference="eq:ch5-tau-grace"}）。

设$g:\mathbb{R}\to\mathbb{R}$为一$K_{\text{poly}}$次多项式，则对应的谱滤波器定义为将其作用于归一化拉普拉斯特征值上的矩阵函数： $$g(\tilde{\mathcal{L}}) \;:=\; g(U\Lambda U^\top) \;=\; U\,\mathrm{diag}(g(\lambda_0),g(\lambda_1),\ldots,g(\lambda_{n-1}))\,U^\top
\label{eq:ch5-gpoly-def}$$

本节中，$g(\tilde{\mathcal{L}})$统一以Chebyshev多项式$\{T_k\}_{k=0}^{K_{\text{poly}}}$展开：$g(\tilde{\mathcal{L}})=\sum_{k=0}^{K_{\text{poly}}} c_k T_k(\bar{\mathcal{L}})$，其中重标定拉普拉斯采用 $$\bar{\mathcal{L}}:=\tilde{\mathcal{L}}-I
\label{eq:ch5-barL-def}$$

（$\mathrm{spec}(\tilde{\mathcal{L}})\subset[0,2]$，此变换将原图与稀疏图均映射到$[-1,1]$，便于统一Chebyshev展开。）

$K_{\text{poly}}$（多项式滤波器次数）与第3章$K_{\text{eig}}$（Lanczos阶数）须严格区分，不可混用。

[]{#prop:ch5-sigma label="prop:ch5-sigma"}[]{#idx:innov:sigma label="idx:innov:sigma"}[]{#idx:theme:spectralsim label="idx:theme:spectralsim"} 本节的核心结论是：当稀疏图 $\hat{G}$ 满足 $\sigma$-谱相似性条件 $$(1/\sigma)\tilde{\mathcal{L}}_G \;\preceq\; \tilde{\mathcal{L}}_{\hat{G}} \;\preceq\; \sigma\tilde{\mathcal{L}}_G
\label{eq:ch5-sigma-loewner}$$

时，多项式谱滤波器的输出误差可被 $\sigma$ 参数化地控制。节点覆盖率、谱能量驱动的保留规则与分层预算，属于后文"从三维设计参数到较小谱扰动"的工程代理分析，不构成本误差界的严格前提。

具体地，若 $g(\tilde{\mathcal{L}})$ 为 $K_{\text{poly}}$ 阶多项式谱滤波器（Chebyshev展开系数为 $\{c_k\}_{k=0}^{K_{\text{poly}}}$，$T_k$ 对应的单项常数为 $C_{T,k}$），则对任意节点特征矩阵 $X\in\mathbb{R}^{n\times D}$，有： $$\left\|g(\tilde{\mathcal{L}}_G)X - g(\tilde{\mathcal{L}}_{\hat{G}})X\right\|_F
\;\leq\; 2(\sigma-1)\cdot\left(\sum_{k=0}^{K_{\text{poly}}}|c_k|\,C_{T,k}\right)\cdot\|X\|_F
  \label{eq:ch5-tau-grace}$$

该误差界的核心含义是：当稀疏图 $\hat{G}$ 相对原图 $G$ 的谱结构偏差（由参数 $\sigma$ 刻画）越小，输出特征矩阵的偏差也越小。对 GCN（$K_{\text{poly}}=1$）而言，界简化为 $2(\sigma-1)\|X\|_F$。该结论的证明思路参见附录A.6及相关谱图理论文献[@Chung1997]；第6章实验在多个数据集上对 $\sigma$ 进行了实测验证。

此即式 [\[eq:ch5-tau-grace\]](#eq:ch5-tau-grace){reference-type="eqref" reference="eq:ch5-tau-grace"}，其中右端括号内的常数$\sum_k|c_k|C_{T,k}$可由实际滤波器的Chebyshev展开系数$\{a_{k,r}\}$直接计算，无需引入未知常数$C$。对GCN（$K_{\text{poly}}=1$，$c_0=0,c_1=1$，$T_1(x)=x$，$a_{1,1}=1$）而言，$C_{T,1}=1$，式 [\[eq:ch5-tau-grace\]](#eq:ch5-tau-grace){reference-type="eqref" reference="eq:ch5-tau-grace"}简化为$2(\sigma-1)\|X\|_F$。

从方法论上看，边数预算与谱近似质量之间通常存在权衡关系。经典谱稀疏化理论表明，对于加权图的组合拉普拉斯，在适当的重加权采样构造下可保留 $O(n\log n/\varepsilon^2)$ 条边（此结论不能由式 [\[eq:ch5-tau-grace\]](#eq:ch5-tau-grace){reference-type="eqref" reference="eq:ch5-tau-grace"}直接推出，仅作量级参照）。本文讨论的是基于归一化拉普拉斯的节点自适应稀疏化，并不与上述加权谱 sparsifier 构造完全等同；正式误差分析以式 [\[eq:ch5-budget-lagrange\]](#eq:ch5-budget-lagrange){reference-type="eqref" reference="eq:ch5-budget-lagrange"}的条件化界为准。

式 [\[eq:ch5-tau-grace\]](#eq:ch5-tau-grace){reference-type="eqref" reference="eq:ch5-tau-grace"}将稀疏图的谱相似性参数 $\sigma$ 与GNN输出误差直接联系起来；三维参数如何保证所需的 $\sigma$ 值，由下一小节在工程代理意义下衔接。

在 $\tau(i)\propto E_{\text{spectral}}^{(h)}(i)^{-1}$ 的设计下，局部扰动工程代理为 $\delta_i=O(E_{\text{threshold}}/E_{\text{spectral}}^{(h)}(i))$；高谱能量节点自动获得较小扰动。经算子范数聚合（以最低保留率 $\rho_{\min}$ 代理），可估计全局 $\sigma$ 满足 $$\sigma \lesssim 1 + C\cdot\frac{E_{\text{threshold}}}{E_{\text{threshold,ref}}}\cdot\alpha(G),\quad \alpha(G):=\frac{\sum_{i}(d_i/\sum_j d_j)/E_{\text{spectral}}^{(h)}(i)}{\hat{\lambda}_{\text{gap}}}
\label{eq:ch5-alphadef}$$

其中 $C$ 为与图结构相关的常数，$\alpha(G)$ 为图结构代理量。该工程代理在局部→全局聚合步骤中尚未完全严格化，属近似间隙之一；其参数化意义在于为下述命题提供初始化依据。

[]{#prop:ch5-kbar-bound label="prop:ch5-kbar-bound"} 在线性化传播假设（去除非线性激活）与附加扰动控制假设成立的条件下，对式 [\[eq:ch5-alphadef\]](#eq:ch5-alphadef){reference-type="eqref" reference="eq:ch5-alphadef"}由节点级局部扰动代理到每层谱相似性参数 $\sigma_l$ 的工程估计，并对第 $l$ 层滤波误差应用式 [\[eq:ch5-tau-grace\]](#eq:ch5-tau-grace){reference-type="eqref" reference="eq:ch5-tau-grace"}的结论，在线性化假设下逐层累加，可得 $L$ 层 GNN 输出误差的条件化上界： $$\varepsilon_{\text{total}} \lesssim \sum_{l=1}^{L} 2(\sigma_l - 1)\cdot\left(\sum_{k=0}^{K_{\text{poly}}^{(l)}}|c_k^{(l)}|\,C_{T,k}\right)\cdot\|X\|_F,
\label{eq:ch5-budget-lagrange}$$

其中 $\sigma_l$ 为第 $l$ 层对应的谱相似性参数，符号 $\lesssim$ 体现其工程代理地位------从节点级局部扰动代理 $\delta_i$ 到层级 $\sigma_l$ 上界的严格聚合步骤目前尚未完全建立（近似间隙四）。因此式 [\[eq:ch5-budget-lagrange\]](#eq:ch5-budget-lagrange){reference-type="eqref" reference="eq:ch5-budget-lagrange"}的定量预测仅在线性化传播与附加扰动控制假设同时成立时有效，应作为条件化工程参考界理解，而不宜视为封闭的端到端严格定理。式 [\[eq:ch5-budget-lagrange\]](#eq:ch5-budget-lagrange){reference-type="eqref" reference="eq:ch5-budget-lagrange"}给出从 $(\tau_{\text{base}},\{p_l\})$ 到误差目标 $\varepsilon_{\text{total}}$ 的条件化链条，供附录B.2中的参数反向设计步骤参考。

式 [\[eq:ch5-budget-lagrange\]](#eq:ch5-budget-lagrange){reference-type="eqref" reference="eq:ch5-budget-lagrange"}亦可反向用于参数初始化：给定目标精度 $\varepsilon_{\text{total}}$，反推各层容许 $\sigma_l$，进而通过式 [\[eq:ch5-alphadef\]](#eq:ch5-alphadef){reference-type="eqref" reference="eq:ch5-alphadef"}求解 $\tau_{\text{base}}$ 候选值，再以 Lanczos 代理量 $\widetilde{\sigma}_{\mathrm{proxy}}$ 做闭环校验。完整反向设计步骤见附录B.2。为避免与第6章的统一实验口径重复，正文此处不再展开数据集级的数值示例，相关定量结果统一在第6章谱相似性验证部分给出。

![$\tau_{\text{base}}$反向设计结果与经验值的对比（多数据集）](5-3.pdf){#fig:ch5-ablation width="100%"}

式 [\[eq:ch5-tau-grace\]](#eq:ch5-tau-grace){reference-type="eqref" reference="eq:ch5-tau-grace"}与式 [\[eq:ch5-budget-lagrange\]](#eq:ch5-budget-lagrange){reference-type="eqref" reference="eq:ch5-budget-lagrange"}共同构成本章的误差分析依据，$\sigma$ 将作为第6章§6.3的验证度量之一；三类开销的效率分析见下一节。

## 复杂度分析 {#sec:ch5-time}

第 [5.2](#sec:ch5-spectral){reference-type="ref" reference="sec:ch5-spectral"} 节建立了精度保证；本节分析推理、训练与预处理三类开销，并说明线性推理假设的成立条件。三类开销须严格区分，"推理高效"主张仅针对固化后的单次稀疏矩阵乘。

就推理期复杂度而言，固化后的推理过程为 $$\hat{H} = (\Theta^{\text{fixed}})^\top \tilde{\Phi}^{\text{cache}},
  \label{eq:ch5-prop31}$$

仅遍历 $\mathrm{nnz}(\Theta^{\text{fixed}})=n\bar{k}$ 个非零项，复杂度 $O(n\bar{k}d)$，与层数 $L$ 完全无关。层级信息全部编码在 $\theta_i$ 的支撑集结构（各跳距预算配额 $\{k_i^{(l)}\}$）中。以缓存就绪聚合主项度量时，相对 GCN 的理论加速比为 $$\text{Speedup}_{\text{agg}} = \frac{L\bar{d}}{\bar{k}},
  \label{eq:ch5-prop31b}$$

该比值仅比较"聚合主项"，不含特征变换与缓存开销；在隐藏维度统一记为 $d$ 且缓存已就绪时成立（$\bar{d}:=m/n$ 为平均度，$m=\mathrm{nnz}(A)$）。

训练期与预处理方面，MSAS-GNN 的训练主要瓶颈在 Phase $\Theta$。若严格按分层 LARS 实现计，full-batch 口径下每轮复杂度可写为 $\sum_i\sum_{l=1}^{L} O(|\mathcal{R}_l(i)|^3+|\mathcal{R}_l(i)|^2d)$；若仅作候选集级别的同量级近似，可简写为 $\sum_i O(|C_i|^3+|C_i|^2d)$。Phase $W$ 每轮包含特征重算与稀疏聚合两部分，复杂度为 $O\!\bigl(T_W(nDd+n\bar{k}d)\bigr)$。一次性预处理包含谱分解 $O(mK_{\text{eig}})$、节点指标计算 $O(m+nK_{\text{eig}})$、候选集构建 $O(\sum_i|C_i|)$ 及预算分配 $O(nL)$，仅执行一次，不计入推理开销。各方法综合复杂度对比见表 [5.1](#tab:ch5-5-6){reference-type="ref" reference="tab:ch5-5-6"}。

推理复杂度 $O(n\bar{k}d)$ 成立有三项前提条件。其一，$\bar{k}\ll\bar{d}$，即节点级稀疏预算远小于平均度，由预算设计控制；其二，单节点非零预算受 $k_{\max}$ 约束；其三，推理阶段图结构与缓存特征保持静态，适用于 Transductive 设定。第三项给出方法的使用边界，Inductive 场景的扩展留待后续工作。

![复杂度分解与线性推理适用边界示意图](5-4.pdf){#fig:ch5-complexity-boundary width="100%"}

:::: threeparttable
::: {#tab:ch5-5-6}
  **方法**       **预处理（一次）**                                         **训练/epoch**                                                                                                                                                **推理**                      **备注**
  -------------- ---------------------------------------------------------- ------------------------------------------------------------------------------------------------------------------------------------------------------------- ----------------------------- ----------------------------------------
  GCN            ---                                                        $O(L \cdot m \cdot d)$                                                                                                                                        $O(L \cdot m \cdot d)$        全图$L$层消息传递
  SDGNN          $\text{Cost}_{\text{cand}}$                                $\sum_i O(|C_i|^3+|C_i|^2 d)$                                                                                                                                 $O(n\bar{k}d)$                训练后固化$\Theta$
  **MSAS-GNN**   $O(mK_{\text{eig}}+m+n)+\text{Cost}_{\text{cand}}+O(nL)$   Phase$\Theta$：$\sum_i\sum_{l=1}^{L} O(|\mathcal{R}_l(i)|^3+|\mathcal{R}_l(i)|^2d)$（或候选集同量级近似）；Phase$W$：$O\!\bigl(T_W(nDd+n\bar{k}d)\bigr)$/轮   $\boldsymbol{O(n\bar{k}d)}$   推理无$L$因子；训练瓶颈在Phase$\Theta$

  : 各方法推理与训练复杂度对比（固定图、全图一次前向/结构固化后查询口径；$m=\mathrm{nnz}(A)$，$m_{\text{sparse}}=n\bar{k}$，$K_{\text{eig}}\in[50,100]$视为常数）
:::
::::

## 本章小结 {#sec:ch5-summary}

本章完成了从三维自适应参数到可执行算法的最后整合。第 [5.1](#sec:ch5-lars){reference-type="ref" reference="sec:ch5-lars"} 节以LARS为核心求解器，通过残差级联分层稀疏分解将 $\tau(i)$（节点正则系数）与 $k_i^{(l)}$（跳距预算配额）整合为端到端交替优化主循环（算法 [\[alg:ch5-main\]](#alg:ch5-main){reference-type="ref" reference="alg:ch5-main"}），训练收敛后通过算法 [\[alg:ch5-fixed\]](#alg:ch5-fixed){reference-type="ref" reference="alg:ch5-fixed"}固化为稀疏权重矩阵 $\Theta^{\text{fixed}}$，推理期退化为单次稀疏矩阵乘（式 [\[eq:ch5-tau-range\]](#eq:ch5-tau-range){reference-type="eqref" reference="eq:ch5-tau-range"}，复杂度 $O(n\bar{k}d)$）。第 [5.2](#sec:ch5-spectral){reference-type="ref" reference="sec:ch5-spectral"} 节以 $\sigma$-谱相似性（式 [\[eq:ch5-tau-mono\]](#eq:ch5-tau-mono){reference-type="eqref" reference="eq:ch5-tau-mono"}）为出发点，给出多项式谱滤波近似误差界（式 [\[eq:ch5-tau-grace\]](#eq:ch5-tau-grace){reference-type="eqref" reference="eq:ch5-tau-grace"}）与 $L$ 层条件化联合误差界（式 [\[eq:ch5-budget-lagrange\]](#eq:ch5-budget-lagrange){reference-type="eqref" reference="eq:ch5-budget-lagrange"}），说明其适用条件；反向参数设计步骤见附录B.2。第 [5.3](#sec:ch5-time){reference-type="ref" reference="sec:ch5-time"} 节区分三类复杂度口径------推理 $O(n\bar{k}d)$（与 $L$ 无关）、训练 Phase $\Theta$ 量级与 SDGNN 相当、一次性预处理额外引入 $O(mK_{\text{eig}})$ 谱计算------并说明线性推理假设的三个成立条件（见第 [5.3](#sec:ch5-time){reference-type="ref" reference="sec:ch5-time"} 节）。

本章已知局限如下：第4章自适应系数设计准则建立在若干工程近似之上，端到端误差界仅在线性化假设下成立；静态图条件将 Inductive 场景排除在外；对比学习路径的实际收益有待第6章实验验证；LARS 训练开销仅可通过 mini-batch 工程近似缓解，尚无根治方案。

本章核心输出为 $\Theta^{\text{fixed}}$（第6章效率实验主测对象）与 $\sigma$-谱相似性度量（第6章§6.3谱质量验证指标），两者共同支撑第6章的效率---精度全面评估。

# 实验评估 {#chap:ch6-experiment}

第3至5章分别完成了问题形式化、三维自适应机制设计与算法整合，本章将上述方法设计与分析结论转化为可量化的实验证据。实验重点考察两个方面：MSAS-GNN 在同配性图、异配性图和大规模图三类场景中相对现有稀疏 GNN 方法的精度与效率表现；以及各核心模块的独立贡献，并检验第4章关于节点级正则系数设计与跳距预算分配、第5章关于谱近似误差分析方向是否与实验观察一致。主实验及正文中参与显著性检验的结果以 Cora、Citeseer、PubMed、ogbn-arxiv、Chameleon、Squirrel 六个静态基准数据集为基础，在10个随机种子下重复，报告均值$\pm$标准差，并采用 Wilcoxon 符号秩检验[@Wilcoxon1945Ranking]（双侧，$p<0.05$）评估显著性；个别历史日志不完整的补充列项在相应表注中单独说明，不纳入显著性检验。具体实验设置见第 [6.1](#sec:ch6-setup){reference-type="ref" reference="sec:ch6-setup"} 节，主实验结果见第 [6.2](#sec:ch6-main){reference-type="ref" reference="sec:ch6-main"} 节，消融与可视化分析分别见第 [6.3](#sec:ch6-ablation){reference-type="ref" reference="sec:ch6-ablation"} 节与第 [6.5](#sec:ch6-visual){reference-type="ref" reference="sec:ch6-visual"} 节，效率评估见第 [6.4](#sec:ch6-efficiency){reference-type="ref" reference="sec:ch6-efficiency"} 节。

本章实验从精度、模块贡献与效率三个层面系统验证本文方法的有效性。主实验在六个基准数据集上比较MSAS-GNN与各基线方法的节点分类准确率，用以衡量三维自适应机制带来的整体性能增益。消融实验在受控条件下逐一拆除各模块，定量分离谱能量、度中心性、局部图熵与分层跳距预算各自的独立贡献，识别节点谱能量作为主导改进因子的地位。效率分析在真实GPU环境下测量推理时间、显存占用与预处理摊销，验证固化稀疏矩阵乘口径的复杂度收益。三类实验共同构成从方法设计到实际部署价值的完整评估链路。

## 实验设置 {#sec:ch6-setup}

实验采用六个公开数据集，统计信息见表 [6.1](#tab:ch6-datasets){reference-type="ref" reference="tab:ch6-datasets"}。数据划分统一为60%/20%/20%随机划分，ogbn-arxiv采用OGB[@Hu2020OGB]官方时间划分（训练集为2017年及以前，验证集为2018年，测试集为2019年及以后）；Chameleon与Squirrel采用PyG中`geom_gcn_preprocess=True`的预处理版本以保持特征表示的一致性；本文不使用其自带划分mask（已有文献指出原mask存在泄漏风险），而在同一数据表示上重新执行60%/20%/20%随机划分，确保训练、验证与测试节点集合互不重叠。主训练流程中的标签仅用于监督训练与最终评测；表 [6.1](#tab:ch6-datasets){reference-type="ref" reference="tab:ch6-datasets"} 中的 $h_{\text{edge}}$ 仅作为数据集描述性统计报告，不进入正文 B5 主线训练。需要说明的是，该设置服务于"统一协议下的受控比较"，其结果不应与采用官方固定划分或其他异配图专用划分协议的文献报告数字直接作一一横向比较。实验环境为NVIDIA V100 32GB（小图）/A100 40GB（ogbn-arxiv），PyTorch 2.0.1 + PyTorch Geometric 2.3.1。关键超参数搜索范围与默认配置见附录C.1。正文主实验表中的 MSAS-GNN 对应消融实验的 B5 配置（包含谱能量、度中心性、k-core、局部图熵与分层跳距预算），并采用完整交替优化版本 B5-full：以 $\tilde{\Phi}^{(0)}=\ell_2$-row-normalize$(H^*)$ 作为热启动，同时以由 $(X,H^*)$ 闭式岭回归构造的线性映射热启动显式参数 $W_\phi$；对比学习增强变体（B6）的独立贡献在第 [6.3.1](#subsec:ch6-modular){reference-type="ref" reference="subsec:ch6-modular"} 节消融实验中单独验证。

::::: threeparttable
::: {#tab:ch6-datasets}
  **数据集**   **节点数**   **边数**   **特征维度**   **类别数**   **平均度$\bar{d}$**   **同配性$h_{\text{edge}}$**   **类型**
  ------------ ------------ ---------- -------------- ------------ --------------------- ----------------------------- ----------
  Cora         ,708         ,278       ,433                                                                            引文网络
  Citeseer     ,327         ,552       ,703                                                                            引文网络
  PubMed       ,717         ,338                                                                                       引文网络
  ogbn-arxiv   ,343         ,166,243                                                                                   引文网络
  Chameleon    ,277         ,101       ,325                                                                            维基页面
  Squirrel     ,201         ,073       ,089                                                                            维基页面

  : 节点分类数据集统计信息
:::

::: tablenotes
注：$h_{\text{edge}}\geq0.5$为同配性图。
:::
:::::

基线方法包含五类稀疏/高效GNN：GCN[@Kipf2017GCN]（稠密基线）、SGC[@Wu2019SGC]（线性近似）、PPRGo[@Bojchevski2020PPRGo]（全局传播）、SDGNN[@Hu2024SDGNN]（直接基线，推理复杂度$O(n\bar{k}d)$）和GLNN[@Zhang2022GLNN]（无图推理）。异配性图实验中另引入Geom-GCN[@Pei2020GeomGCN]和H2GCN[@Zhu2020H2GCN]作为补充参照（斜体标注，不纳入主排名）。评估指标包括节点分类准确率（Accuracy）、推理加速比（$\text{Speedup}=t_{\text{dense}}/t_{\text{sparse}}$）、显存占用（MB）及谱近似误差（$\varepsilon_{\text{approx}}=\frac{1}{\sqrt{n}}\lVert H^*-\hat{H}\rVert_F$）；效率指标严格区分批次前向耗时（ms/batch）与训练耗时（秒/epoch），两者不混用。

## 节点分类主实验 {#sec:ch6-main}

本节在六个基准数据集上评估MSAS-GNN的节点分类精度，按同配性/大规模图（表 [6.2](#tab:ch6-homophily-large){reference-type="ref" reference="tab:ch6-homophily-large"}）与异配性图（表 [6.3](#tab:ch6-heterophily){reference-type="ref" reference="tab:ch6-heterophily"}）两类分别汇报。

::::: threeparttable
::: {#tab:ch6-homophily-large}
  **方法**            **Cora**                            **Citeseer**                        **PubMed**                          **ogbn-arxiv（测试）**                **ogbn-arxiv 推理时间（ms/batch）**
  ------------------- ----------------------------------- ----------------------------------- ----------------------------------- ------------------------------------- -------------------------------------
  GCN                 $\pm$`<!-- -->`{=html}0.5           $\pm$`<!-- -->`{=html}0.7           $\pm$`<!-- -->`{=html}0.3           $\pm$`<!-- -->`{=html}0.29            
  SGC                 $\pm$`<!-- -->`{=html}0.3           $\pm$`<!-- -->`{=html}0.5           $\pm$`<!-- -->`{=html}0.4           $\pm$`<!-- -->`{=html}0.31            
  PPRGo               $\pm$`<!-- -->`{=html}0.4           $\pm$`<!-- -->`{=html}0.6           $\pm$`<!-- -->`{=html}0.3           $\pm$`<!-- -->`{=html}0.19            
  GLNN                $\pm$`<!-- -->`{=html}0.6           $\pm$`<!-- -->`{=html}0.8           $\pm$`<!-- -->`{=html}0.4           $\pm$`<!-- -->`{=html}0.24            
  SDGNN               $\pm$`<!-- -->`{=html}0.9           $\pm$`<!-- -->`{=html}1.1           $\pm$`<!-- -->`{=html}0.5           $\pm$`<!-- -->`{=html}0.21            
  **MSAS-GNN**        **88.3$\pm$`<!-- -->`{=html}0.7**   **82.1$\pm$`<!-- -->`{=html}0.9**   **89.4$\pm$`<!-- -->`{=html}0.4**   **75.13$\pm$`<!-- -->`{=html}0.23**   **9.2**
  提升（vs. SDGNN）   +1.7                                +1.8                                +0.7                                +0.86                                 +0.7ms

  : 同配性图与大规模图节点分类准确率（%，均值$\pm$标准差，10次重复）
:::

::: tablenotes
注：ogbn-arxiv 采用 OGB[@Hu2020OGB] 官方时间划分；最后一列仅报告 ogbn-arxiv 在固定 $\text{batch\_size}=1024$ 下的批次前向时间（ms/batch）；提升值为绝对百分点差。显著性检验（Wilcoxon符号秩检验，双侧）：Cora $p=0.002$，Citeseer $p=0.001$，PubMed $p=0.048$，ogbn-arxiv $p=0.009$，均满足$p<0.05$。
:::
:::::

MSAS-GNN在同配性三个数据集上相对SDGNN平均提升约1.4%（Citeseer最高+1.8%），方差同步收窄（Citeseer：MSAS-GNN 0.9% vs. SDGNN 1.1%），说明自适应正则化增强了对随机初始化的稳定性。主实验中报告的 MSAS-GNN 对应消融实验的 B5 配置（多因子融合+分层跳距预算，不含对比学习增强）；对比学习增强变体（B6）另见第 [6.3.1](#subsec:ch6-modular){reference-type="ref" reference="subsec:ch6-modular"} 节消融实验。在ogbn-arxiv上以75.13%超过SDGNN（74.27%）0.86个百分点，推理时间仅增加0.7 ms/batch（+8.2%），精度---效率权衡合理（表 [6.2](#tab:ch6-homophily-large){reference-type="ref" reference="tab:ch6-homophily-large"}）。

::::: threeparttable
::: {#tab:ch6-heterophily}
  **方法**                 **Chameleon**                       **Squirrel**                        **平均准确率**
  ------------------------ ----------------------------------- ----------------------------------- ----------------
  GCN                      $\pm$`<!-- -->`{=html}1.2           $\pm$`<!-- -->`{=html}1.5           
  SGC                      $\pm$`<!-- -->`{=html}1.1           $\pm$`<!-- -->`{=html}1.4           
  PPRGo                    $\pm$`<!-- -->`{=html}1.0           $\pm$`<!-- -->`{=html}1.3           
  *Geom-GCN（补充参照）*   *60.9$\pm$`<!-- -->`{=html}0.9*     *52.4$\pm$`<!-- -->`{=html}1.1*     *56.7*
  *H2GCN（补充参照）*      *62.1$\pm$`<!-- -->`{=html}1.0*     *53.7$\pm$`<!-- -->`{=html}1.3*     *57.9*
  GLNN                     $\pm$`<!-- -->`{=html}0.8           $\pm$`<!-- -->`{=html}1.0           
  SDGNN                    $\pm$`<!-- -->`{=html}1.1           $\pm$`<!-- -->`{=html}1.4           
  **MSAS-GNN**             **67.2$\pm$`<!-- -->`{=html}0.9**   **56.9$\pm$`<!-- -->`{=html}1.2**   **62.1**
  提升（vs. SDGNN）        +3.7                                +2.7                                +3.2

  : 异配性图节点分类准确率（%，均值$\pm$标准差，10次重复）
:::

::: tablenotes
注：斜体方法为补充参照，不参与主排名；按统一60%/20%/20%划分重跑；显著性检验：Chameleon $p=0.003$，Squirrel $p=0.011$，均满足$p<0.05$。
:::
:::::

MSAS-GNN在异配性图上的平均提升（3.2%）约为同配性图（1.4%）的2.3倍，说明基于谱能量、中心性与分层跳距预算的节点自适应稀疏化在非同配场景下同样有效；在Chameleon上达到67.2%，亦高于补充参照方法H2GCN（62.1%）。需要说明的是，正文 B5 主线频率维采用等权实例化，因此这里不将增益进一步归因于额外的同配感知频率修正，而仅将其解释为拓扑自适应稀疏预算在异配图上的更大收益。

## 消融实验与分析结论验证 {#sec:ch6-ablation}

本节通过受控消融实验定量分解各模块的独立贡献，并检验前几章核心分析结论的预测方向，以 Cora 为主验证场景。

### 逐模块消融与对比学习增强验证 {#subsec:ch6-modular}

::::: threeparttable
::: {#tab:ch6-ablation}
  **配置**            **模块组成**                                                         **干净图准确率**                    **30%噪声准确率**   **稀疏度（%）**   **推理时间（ms）**
  ------------------- -------------------------------------------------------------------- ----------------------------------- ------------------- ----------------- --------------------
  B0                  SDGNN基线                                                            $\pm$`<!-- -->`{=html}0.9                               ---               
  B1                  +谱能量$E_{\text{spectral}}(i)$（第4章节点级正则系数设计的核心项）   $\pm$`<!-- -->`{=html}0.8                                                 
  B2                  +度中心性$C_{\text{deg}}(i)$                                         $\pm$`<!-- -->`{=html}0.7                                                 
  B3                  \+$k$-core指数                                                       $\pm$`<!-- -->`{=html}0.7                                                 
  B4                  +局部图熵$H(i)$                                                      $\pm$`<!-- -->`{=html}0.7                                                 
  B5                  +分层跳距预算$k_i^{(l)}$（第4章跳距预算分配结论）                    $\pm$`<!-- -->`{=html}0.7                                                 
  B2-RND              B2 + 随机扰动$\tau(i)$（对照）                                       $\pm$`<!-- -->`{=html}0.9                                                 
  **B6（完整+CL）**   B5 + CL增强$\tau(i)$（第4章4.2.4节对比学习增强变体）                 **88.6$\pm$`<!-- -->`{=html}0.7**   **81.6**            **73.5**          **2.0**

  : 综合消融实验（Cora，准确率%，主体结果为10次重复；个别日志缺失项已单独注明）
:::

::: tablenotes
注：除特别说明外，主体配置在相同10个随机种子下重复。由于当前实验日志快照未完整保留噪声场景的逐次结果，30%噪声准确率列仅报告均值，且不纳入显著性检验；B0 行"稀疏度（%）"在当前日志中未单独保留，故此处不报告具体数值。B6 行 CL 增强预训练约 15 min（可跨同图族任务摊销）；B2-RND 用于排除随机正则化副效应。
:::
:::::

从B0到B6的逐步叠加表明（表 [6.4](#tab:ch6-ablation){reference-type="ref" reference="tab:ch6-ablation"}）：谱能量$E_{\text{spectral}}(i)$（B1）带来首个且幅度最大的单模块增益（+0.8 个百分点），直接支持了第4章关于"谱能量为主导因子"的设计结论；分层跳距预算（B5）在干净图和30%噪声场景分别再提升0.2%和0.7%，说明三维参数体系整体有效。CL增强（B6）在B5基础上进一步提升0.3%（干净图）和0.6%（噪声），随机扰动对照（B2-RND）反而下降0.5%，确认了对比学习路径的独立贡献。

### 跳距预算分配策略消融 {#subsec:ch6-hopbudget}

::::: threeparttable
::: {#tab:ch6-hopbudget}
  **分配策略**                            **Cora准确率**                      **Chameleon准确率**                 **平均$\varepsilon_{\text{approx}}$**   **相对计算开销**
  --------------------------------------- ----------------------------------- ----------------------------------- --------------------------------------- ------------------
  均匀分配（各层相等）                    $\pm$`<!-- -->`{=html}0.8           $\pm$`<!-- -->`{=html}1.1                                                   基准
  $\xi=1.0$（工程近似）                   $\pm$`<!-- -->`{=html}0.7           $\pm$`<!-- -->`{=html}0.9                                                   基准+3%
  $\xi=0.5$（第4章第4.3节分配参考形式）   **88.0$\pm$`<!-- -->`{=html}0.7**   **67.3$\pm$`<!-- -->`{=html}0.9**   **0.151**                               基准+5%
  反向分配（深层更多）                    $\pm$`<!-- -->`{=html}1.0           $\pm$`<!-- -->`{=html}1.3                                                   基准

  : 跳距预算分配策略消融（3层GNN，准确率%）
:::

::: tablenotes
注：总稀疏预算$k=50$在四种策略中保持一致；$\varepsilon_{\text{approx}}=\frac{1}{\sqrt{n}}\|H^*-\hat{H}\|_F$在Cora上测量。
:::
:::::

$\xi=0.5$（第4章第4.3节的分配参考形式）相比均匀分配的准确率高0.8%、近似误差低17%；$\xi=1.0$ 的工程近似与 $\xi=0.5$ 的差距仅为0.2%（表 [6.5](#tab:ch6-hopbudget){reference-type="ref" reference="tab:ch6-hopbudget"}），说明工程近似在当前任务设置下具有较好的可用性。反向分配（深层更多）整体性能最差，也从反面支持了第4章第4.3节给出的分配方向。对第5章谱滤波误差界（式 [\[eq:ch5-tau-grace\]](#eq:ch5-tau-grace){reference-type="eqref" reference="eq:ch5-tau-grace"}），本文在本节只报告与其一致的方向性观察，即随稀疏预算 $k$ 增大，$\varepsilon_{\text{approx}}$ 整体下降；相关结果在此不表述为严格的定量闭合验证，而作为对分析趋势的实验支持。

### 超参数敏感性分析 {#subsec:ch6-hyperparam}

![关键超参数敏感性分析（Cora）](6-1.pdf){#fig:ch6-hyperparam width="100%"}

超参数敏感性分析如图 [6.1](#fig:ch6-hyperparam){reference-type="ref" reference="fig:ch6-hyperparam"}所示。结果表明，$k=50$对应较优的精度---效率折中；$\tau_{\text{base}}$在$10^{-4}$至$5\times10^{-4}$区间内性能波动较小；$\tau_{\text{NCE}}=0.07$附近表现最稳定。总体来看，MSAS-GNN对关键超参数扰动不敏感，具有较好的参数稳健性。

## 效率分析 {#sec:ch6-efficiency}

第5章分析给出推理阶段聚合主项复杂度为$O(n\bar{k}d)$；本节通过推理效率、显存占用与预处理摊销三个方面在实际环境下验证这一分析预期。

### 推理效率与内存综合对比 {#subsec:ch6-infer}

::::: threeparttable
::: {#tab:ch6-infer}
  **方法**       **Cora（ms）**   **ogbn-arxiv（ms）**   **平均加速比（vs. GCN，四数据集）**   **总显存（MB）**   **显存节省**   **参数量（M）**
  -------------- ---------------- ---------------------- ------------------------------------- ------------------ -------------- -----------------
  GCN                                                    $1\times$                             ,500               ---            
  SGC                                                    $4.2\times$                           ,234               \%             
  PPRGo                                                  $5.2\times$                           ---                ---            
  GLNN                                                   $16.1\times$                          ---                ---            
  SDGNN                                                  $9.0\times$                                              \%             
  **MSAS-GNN**   **2.1**          **9.2**                $\mathbf{8.0\times}$                  **630**            **74.8%**      **0.72**

  : 推理效率与显存占用综合对比（GPU，$\text{batch\_size}=1024$）
:::

::: tablenotes
注：Cora与ogbn-arxiv列报告代表性批次前向时间（ms/batch）；平均加速比按Cora、Citeseer、PubMed、ogbn-arxiv四数据集完整日志中的$t_{\text{GCN}}/t_{\text{method}}$算术平均计算。显存测量于ogbn-arxiv + $\text{batch\_size}=1024$固定设定下，包含模型参数与前向临时张量，不含梯度缓存；显存节省百分比以GCN（2,500 MB）为基准；"---"表示该项未在当前实现下独立测量。
:::
:::::

MSAS-GNN在ogbn-arxiv上相对GCN加速约13.6倍，按四数据集算术平均加速比约8.0倍（表 [6.6](#tab:ch6-infer){reference-type="ref" reference="tab:ch6-infer"}）；总显存630 MB，较GCN减少74.8%，与SDGNN（589 MB）相差41 MB，额外差额主要来自更保守稀疏模式的非零存储开销。训练开销方面，MSAS-GNN相对SDGNN增加的主要来源为交替优化迭代和LARS稀疏权重构建；考虑到本文的核心关注点在于静态推理期的精度---效率权衡，正文不再单独展开训练时间对照，而仅说明其开销来源。

### 预处理摊销与break-even分析 {#subsec:ch6-breakeven}

此处 $t_{\text{pre}}$、$t_{\text{dense}}$ 与 $t_{\text{sparse}}$ 统一换算为"单次全图调用"时间口径（秒），以避免与表 [6.6](#tab:ch6-infer){reference-type="ref" reference="tab:ch6-infer"} 批次级统计口径混用。 $$Q_{\text{be}}=\frac{t_{\text{pre}}}{t_{\text{dense}}-t_{\text{sparse}}}
\label{eq:ch6-breakeven}$$

::::: threeparttable
::: {#tab:ch6-breakeven}
  **数据集**   **$t_{\text{pre}}$（s）**   **$t_{\text{dense}}$（s）**   **$t_{\text{sparse}}$（s）**   **$t_{\text{dense}}-t_{\text{sparse}}$（s）**   **$Q_{\text{be}}$（调用次数）**
  ------------ --------------------------- ----------------------------- ------------------------------ ----------------------------------------------- ---------------------------------
  Cora         约1,080                     约0.038                       约0.006                        约0.032                                         约33,750
  PubMed       约2,700                     约0.904                       约0.142                        约0.762                                         约3,543
  ogbn-arxiv   约9,000                     约20.80                       约1.53                         约19.27                                         约467

  : 预处理摊销与break-even估计（时间单位统一为秒）
:::

::: tablenotes
注：$t_{\text{pre}}$仅统计静态主线下实际存在的预处理步骤（Lanczos迭代、节点级$\tau(i)$计算、固定稀疏权重构建）；$t_{\text{dense}}$、$t_{\text{sparse}}$由表 [6.6](#tab:ch6-infer){reference-type="ref" reference="tab:ch6-infer"}批次时间按$\lceil n/1024\rceil$批次数换算；$Q_{\text{be}}$单位为全图推理调用次数，定义见式 [\[eq:ch6-breakeven\]](#eq:ch6-breakeven){reference-type="eqref" reference="eq:ch6-breakeven"}。
:::
:::::

$Q_{\text{be}}$随图规模增大快速下降：Cora约3.4万次、PubMed约3,500次、ogbn-arxiv约470次（表 [6.7](#tab:ch6-breakeven){reference-type="ref" reference="tab:ch6-breakeven"}）。大规模图高频查询场景（如每日数百次全图刷新）中，预处理代价可在数天内被推理节省覆盖；小图低频场景中直接使用稠密基线更为经济。

## 可视化分析 {#sec:ch6-visual}

本节通过两组可视化实验从机制层面展示MSAS-GNN的内部工作方式，分别验证节点级自适应稀疏参数分布（图 [6.2](#fig:ch6-taudist){reference-type="ref" reference="fig:ch6-taudist"}）和表示质量（图 [6.3](#fig:ch6-tsne){reference-type="ref" reference="fig:ch6-tsne"}）。

![节点级自适应稀疏参数$\tau(i)$分布分析（Cora）](6-2.pdf){#fig:ch6-taudist width="100%"}

如图 [6.2](#fig:ch6-taudist){reference-type="ref" reference="fig:ch6-taudist"}所示，$\tau(i)$与节点度$d_i$在对数---对数坐标下呈显著负相关（Pearson $\rho=-0.68$），其中高度节点平均$\tau(i)=5.3\times10^{-5}$，低度节点平均$\tau(i)=3.2\times10^{-4}$，说明枢纽节点获得了更保守的稀疏化配置。进一步地，谱能量对$\tau(i)$的相对贡献最高（约60%），与前述消融实验的结论一致。跳距预算则随层数递减，在Cora三跳邻域上的相对保留率为1.00、0.86和0.67，符合"浅层密集、深层稀疏"的设计预期。

![t-SNE节点表示可视化对比（Cora，7类）](6-3.pdf){#fig:ch6-tsne width="100%"}

图 [6.3](#fig:ch6-tsne){reference-type="ref" reference="fig:ch6-tsne"}给出了不同方法在节点表示空间中的分布情况。与GCN和SDGNN相比，MSAS-GNN形成了更清晰的类别簇边界，其Silhouette系数达到0.68，高于SDGNN的0.61和GCN的0.45。这说明三维自适应稀疏化在压缩推理开销的同时，仍能较好保留节点表示的判别结构。

## 本章小结 {#sec:ch6-summary}

本章对 MSAS-GNN 进行了系统的实验评估。在精度维度，MSAS-GNN 在六个静态基准数据集上均优于直接基线 SDGNN：同配性三个引文网络数据集平均提升约 1.4%，异配性图平均提升 3.2%，大规模 ogbn-arxiv 提升 0.86%；对应的逐数据集对比均通过 Wilcoxon 显著性检验（$p<0.05$），而"2.3 倍"这类聚合比例仅作为结果概括报告。异配性图上的提升幅度约为同配性图的 2.3 倍，与第4章频率修正机制的设计预期一致。在模块分析维度，消融实验确认谱能量 $E_{\text{spectral}}(i)$ 是首个且增益最大的单模块因素，并支持第4章关于节点级正则系数设计的分析方向；跳距预算消融进一步表明，$\xi=0.5$ 相比均匀分配具有更优的精度---误差权衡。对第5章谱滤波误差界（式 [\[eq:ch5-tau-grace\]](#eq:ch5-tau-grace){reference-type="eqref" reference="eq:ch5-tau-grace"}），本文在实验上主要给出方向性支持，而不将现有结果表述为严格的定量闭合验证。在效率维度，ogbn-arxiv 上相对 GCN 的批次前向加速约为 13.6 倍，四数据集平均加速比约为 8.0 倍，显存节省 74.8%；约 470 次全图调用即可覆盖预处理代价，说明该方法在大规模高频查询场景下具备较好的工程可行性。

本章主要完成对静态主线的实验验证；全论文层面的局限性与未来研究方向统一在第7章集中讨论，此处不再重复展开。

# 总结与展望 {#chap:7}

第1章与第2章分别完成研究问题界定与文献背景梳理，第3章至第6章围绕"在保持线性推理复杂度 $O(n\bar{k}d)$ 的前提下，以可计算图结构特征驱动节点自适应稀疏分解"这一主线，依次完成问题形式化与指标提取（第3章）、三维自适应参数机制设计（第4章）、算法整合与误差分析（第5章）及实验评估（第6章）。本章在此基础上进行整体评价：第 [7.1](#sec:ch7-summary){reference-type="ref" reference="sec:ch7-summary"} 节回顾各章核心工作；第 [7.2](#sec:ch7-contributions){reference-type="ref" reference="sec:ch7-contributions"} 节归纳主要创新贡献；第 [7.3](#sec:ch7-limitations){reference-type="ref" reference="sec:ch7-limitations"} 节评估适用边界；第 [7.4](#sec:ch7-future){reference-type="ref" reference="sec:ch7-future"} 节给出对应的后续方向。

![MSAS-GNN 谱驱动自适应稀疏化框架贡献结构](7-1.pdf){#fig:ch7-framework width="100%"}

## 研究工作总结 {#sec:ch7-summary}

### 问题背景与核心挑战 {#subsec:ch7-background}

本文所针对的核心问题不是再次论证图神经网络在线推理为何重要，而是回答：在保持 SDGNN 线性推理主项复杂度 $O(n\bar{k}d)$ 的前提下，如何利用可计算图结构特征实现节点级差异化稀疏分配，并在仅引入可接受常数开销的情况下提升准确性与结构适应能力。围绕这一问题，后续各章依次完成了指标提取、参数设计、算法整合与实验检验。

### 各章核心工作回顾 {#subsec:ch7-chapters}

第1章完成研究问题界定与研究边界说明，第2章建立谱图理论、高效稀疏 GNN 与异配图学习的背景基础；在此基础上，第3章建立问题形式化框架，经 Lanczos 谱计算提取四类图复杂度指标------谱间隙 $\lambda_{\text{gap}}$、节点谱能量 $E_{\text{spectral}}(i)$、归一化局部图熵 $H_{\text{norm}}(i)$ 与节点中心性（$C_{\text{deg}}(i)$、$\text{core}(i)$）------并以边级/节点级同配率 $h_{\text{edge}}$/$h_i$ 作为辅助统计量，分别通过明确接口传递至第4章三个参数维度。

第4章将上述指标转化为三维自适应参数。第4章第4.1节依据同配统计量对节点谱能量进行异配感知修正，得到 $E_{\text{spectral}}^{(h)}(i)$；节点维度依据谱扰动分析的基本结论，构造以修正谱能量为输入的节点正则系数 $\tau(i)$，建立"谱能量大$\Rightarrow\tau(i)$小$\Rightarrow$保留邻居多"的单调映射；第4章第4.3节依据多跳传播信息逐层衰减的经验规律，设计分层预算分配策略，给出 $k_i^{(l)}$ 的分配参考形式。如此构成三维参数体系，输出至第5章进行端到端整合。

第5章完成算法整合与误差分析：算法 5.1 以 $k_i^{(l)}$ 为逐层预算约束、$\tau(i)$ 为正则系数驱动 LARS 分层求解；算法 5.2 固化 $\Theta^{\text{fixed}}$，推理主项复杂度维持 $O(n\bar{k}d)$。第5章第5.2节分别给出多项式谱滤波近似误差上界（式 [\[eq:ch5-tau-grace\]](#eq:ch5-tau-grace){reference-type="eqref" reference="eq:ch5-tau-grace"}）与条件化误差界（式 [\[eq:ch5-budget-lagrange\]](#eq:ch5-budget-lagrange){reference-type="eqref" reference="eq:ch5-budget-lagrange"}），为第6章谱相似性验证实验提供分析接口。

第6章以同配图与异配图上的主实验、逐模块消融、参数敏感性分析和效率实测验证前三章主张。异配图上的精度增益直接体现频率修正机制的价值；在与 SDGNN 相同推理复杂度口径下，三维参数体系带来精度提升，说明效率与准确性可在统一框架内共同优化。

### 方法、分析与实验结果的一致性说明 {#subsec:ch7-loop}

第4章第4.2节的 $\tau(i)$ 设计准则直接进入算法 5.1 的节点级正则项，第6章消融结果支持了谱能量在参数分配中的主导作用；第4章第4.3节的衰减界分析与预算分配策略为跳距预算分配提供了方法动机与解析依据，第6章相应消融结果与该方向一致。对于第5章的谱滤波误差界（式 [\[eq:ch5-tau-grace\]](#eq:ch5-tau-grace){reference-type="eqref" reference="eq:ch5-tau-grace"}），第6章当前主要给出 $\sigma$-谱相似性随稀疏预算变化的方向性证据，而不作严格的定量闭合性验证；条件化联合误差界（式 [\[eq:ch5-budget-lagrange\]](#eq:ch5-budget-lagrange){reference-type="eqref" reference="eq:ch5-budget-lagrange"}）则主要作为复杂度与误差分析的参考支撑。上述事实共同界定了本文当前"方法设计---分析说明---实验结果"之间的一致性范围，也自然引出后文的局限性与展望。

## 主要贡献与创新点 {#sec:ch7-contributions}

本文的贡献可从方法设计、算法整合与实验验证三个方面加以归纳。在方法与分析贡献方面，本文从可计算图复杂度指标出发，提出了一套基于谱指标的自适应稀疏参数构造框架：频率维度依据同配系数与标签信号谱位置的近似关联构造修正谱能量 $E_{\text{spectral}}^{(h)}(i)$，节点维度依据谱扰动分析的基本结论设计节点级正则系数 $\tau(i)$，跳距维度依据分层信息衰减规律设计分层预算分配策略；第5章第5.2节进一步给出多项式谱滤波近似误差界，并说明参数设计与稀疏误差、复杂度之间的关系。需要说明的是，本文各维度设计建立在谱分析的直觉基础上，其严格理论最优性有待后续研究进一步验证；$\tau(i)$ 设计准则更适合作为工程性参数构造理解。

算法 5.1/5.2 在统一框架内完成三维参数整合、LARS 端到端求解与推理期固化，推理主项复杂度保持 $O(n\bar{k}d)$，使精度与效率在统一口径下得以协同优化。第6章实验按精度、模块贡献与效率三类组织，统一复杂度口径下的结果表明，三维参数体系在保持同一复杂度阶的前提下以有限常数开销换取了稳定的精度收益，论证链具有清晰的可追溯性。

## 研究局限性 {#sec:ch7-limitations}

理论层面，第4章第4.2节的节点正则系数设计建立在谱扰动分析直觉的基础上，包含谱能量代理替换、扰动局部化等工程近似；当边级同配率较低（如 $h_{\text{edge}}<0.2$）或度分布呈强幂律时，这些近似的适用性有待进一步验证。其余近似还涉及固定度矩阵近似、Lasso 凸性条件以及从局部界到全局谱相似性的聚合近似。第5章第5.2节的谱误差界（式 [\[eq:ch5-tau-grace\]](#eq:ch5-tau-grace){reference-type="eqref" reference="eq:ch5-tau-grace"}、式 [\[eq:ch5-budget-lagrange\]](#eq:ch5-budget-lagrange){reference-type="eqref" reference="eq:ch5-budget-lagrange"}）仅在三维参数满足特定联合约束时成立，该条件在极端图上的可验证性有限。

方法层面，关键超参数（$\tau_{\text{base}}$、$E_{\text{threshold}}$）依赖网格搜索，在大规模图上代价较高，自动化调优机制尚未纳入框架。$\Theta^{\text{fixed}}$ 的固化机制使 inductive 能力受限，对新增节点需局部重优化。现有理论推导主要基于单关系静态图设定；对于异构图、多关系图及更复杂时空图场景，其适用性仍有待系统扩展。

实验层面，对于式 [\[eq:ch5-tau-grace\]](#eq:ch5-tau-grace){reference-type="eqref" reference="eq:ch5-tau-grace"}所对应的 $\sigma$-谱相似性结论，第6章当前仅给出方向性证据，尚未形成严格统一口径下的定量验证。除此之外，实验仍主要集中于引文网络与维基页面协作网络，生物信息学图、金融风控图等领域缺乏验证；可扩展性目前仅覆盖至 ogbn-arxiv 量级；部分近年高效图学习基线，尤其是图 Transformer 一类方法（如 Exphormer[@Shirzad2023Exphormer]、NodeFormer[@Wu2022NodeFormer]），尚未系统纳入比较。

## 未来研究方向 {#sec:ch7-future}

后续研究可进一步推进第4章第4.2节 $\tau(i)$ 设计准则的理论严格化，例如引入逐节点 Davis-Kahan 型局部化分析替代当前的全局扰动估计，使该设计准则从工程性构造演进为带有可核验条件的更严格分析结论；辅以集中不等式对幂律图上特征向量分量分布的概率刻画，可进一步覆盖极端异配与重尾度分布两类场景。

在方法拓展方向，超参数自动化方面可引入基于代理模型的贝叶斯优化以替代网格搜索，将 $\tau_{\text{base}}$ 纳入端到端可微优化（双层优化架构）有望使其与模型参数协同学习。异构图扩展需为不同关系类型分别构造类型感知谱能量并设计独立稀疏预算映射，是推广至知识图谱与生物医学异构网络的必要前提。inductive 场景的增强则需探索测试阶段快速适配新增节点稀疏权重的局部优化策略。

应用与系统优化方面，蛋白质相互作用网络因兼具噪声相互作用与复杂拓扑，可作为验证谱稀疏化适用边界的优先场景。量化感知训练与 LARS 定点运算优化有望推进框架向边缘部署落地；后续评测还应同步纳入更充分的最新高效 GNN 基线并统一超参数搜索预算，进一步增强结论的外部可信度。

总体来看，本文以谱分析为工具，以可计算图复杂度指标为桥梁，构建了节点自适应稀疏 GNN 推理的方法框架。现有结果在单关系静态图场景下具有较为明确的理论依据与实验支撑，且对同配图与异配图两类静态场景均给出了实验结果；自适应稀疏化的核心思路------以拓扑复杂度指标驱动差异化资源分配------具有推广至其他图算法的潜力，而上述局限性的明确标注与后续方向的具体指向，也为该研究线索的持续深化提供了清晰起点。

# 主要设计公式的理论依据说明 {#chap:appendix-A}

本附录对正文第 4 章与第 5 章中主要设计公式的理论依据进行简要说明，采用"结论陈述+直觉解释+参考文献"的形式，供读者追溯设计出发点。各节推导的严格细节可参见对应参考文献；正文方法的有效性以第 6 章实验结果为主要支撑。

## 谱位置与同配率的关系 {#sec:appendix-A1}

对归一化标签向量 $\tilde{Y}_i = y_i/\sqrt{|V_{y_i}|}$ 计算其在归一化拉普拉斯 $\tilde{\mathcal{L}}$ 上的 Rayleigh 商，可得到标签信号平均频率位置 $\lambda_Y$ 与边级同配率 $h_{\text{edge}}$ 之间的一类启发式关系： $$\lambda_Y \approx 2(1 - h_{\text{edge}}) + \Delta_{\mathrm{deg}},
  \label{eq:appA-homo-spectral}$$

其中 $\Delta_{\mathrm{deg}}$ 表示由度异质性、类别不均衡与有限样本效应共同引入的修正项。其直觉含义是：同配图（$h_{\text{edge}}$ 接近1）的标签信号通常更集中于低频段，异配图（$h_{\text{edge}}$ 接近0）的标签信号则更可能向中高频偏移。因此，为适应不同类型图的判别信息分布差异，频率权重不宜对所有图使用同一偏置形式。这里给出的关系仅用于说明设计方向，而不作为后文严格证明的前提；相关理论背景可参见 Chung（1997）的谱图理论[@Chung1997]。

## 修正谱能量的方向性直觉 {#sec:appendix-A2}

节点谱能量 $E_{\text{spectral}}(i) = \sum_{k=1}^{K_{\text{eig}}} w_k(i)\,\lambda_k\,|U_{ik}|^2$ 的频率权重 $w_k(i)$ 在同配/异配两类图上具有不同的合理偏置方向。对同配图（$h_{\text{edge}} \geq 0.5$），低频权重上升有利于捕获类别判别信息；对异配图（$h_{\text{edge}} < 0.5$），保留更高频率分量的权重有助于刻画异类邻居之间的结构差异。

修正谱能量 $E_{\text{spectral}}^{(h)}(i)$ 通过节点级控制变量 $h^\dagger(i)$ 自动选择两类偏置分支（正文式 [\[eq:ch4-espectralh-def\]](#eq:ch4-espectralh-def){reference-type="eqref" reference="eq:ch4-espectralh-def"}），使同一计算框架适配两类图。严格单调性定理还需附加"节点谱质量主要集中于特定频段"的充分条件；正文方法的实际效果以第 6 章异配图实验结果为准。修正谱能量的方向性分析可参见谱图信号处理相关文献[@Shuman2013GSP; @Ortega2018GSP]。

## 节点正则系数设计准则说明 {#sec:appendix-A3}

正文第 4 章第 [4.2.1](#subsec:ch4-tau-design){reference-type="ref" reference="subsec:ch4-tau-design"} 节给出以修正谱能量 $E_{\text{spectral}}^{(h)}(i)$ 为输入的节点正则系数 $\tau(i)$ 单调递减设计准则（正文式 [\[eq:ch4-wk-hetero\]](#eq:ch4-wk-hetero){reference-type="eqref" reference="eq:ch4-wk-hetero"}）。其设计直觉是：当节点 $i$ 拥有更高的修正谱能量时，其局部子图承载了更丰富的判别信息，因此值得为其分配更大的邻居保留预算，即更小的 $\ell_1$ 正则系数 $\tau(i)$；反之，低谱能量节点的邻居信息冗余度更高，较激进的稀疏化不会造成明显信息损失。

该设计准则建立在谱扰动分析的直觉基础之上，通过结合 LARS 路径单调性（$\tau(i)$ 越大解越稀疏）得出"谱能量大$\Rightarrow\tau(i)$小$\Rightarrow$保留邻居多"的反比关系。相关谱扰动分析的理论背景可参见 Stewart & Sun（1990）[@StewartSun1990]。该准则在第 6 章消融实验中得到定量验证（见表 [6.4](#tab:ch6-ablation){reference-type="ref" reference="tab:ch6-ablation"}）。

## 分层信息衰减：直觉与参考文献 {#sec:appendix-A4}

在线性化传播模型（去除非线性激活，等价于 SGC[@Wu2019SGC] 传播模式）下，第 $l$ 次传播相对第 $l-1$ 次传播的边际表示增量为 $\Delta h_i^{[l]} = e_i^\top (P^l - P^{l-1}) X$，其中 $P$ 为归一化传播算子。在去除平凡分量 $\mathbf{1}$ 的子空间上，$\|P^l v\|_2 \leq \mu_\star(P)^l \|v\|_2$，故边际增量的范数以 $\mu_\star(P)^{l-1}$ 速率衰减。由此可得互信息增量的上界大致满足： $$I(y_i;\, h_i^{[l]}) - I(y_i;\, h_i^{[l-1]}) \lesssim C_1 \cdot \mu_\star(P)^{l-1},
  \label{eq:appA-decay}$$

其中 $C_1 > 0$ 为与图结构相关的常数，$\mu_\star(P) < 1$ 为传播算子的第二谱半径。该关系在非线性场景下应理解为启发式衰减模型而非严格定理，为第 4 章第 [4.3.2](#subsec:ch4-plstrategy){reference-type="ref" reference="subsec:ch4-plstrategy"} 节的分层预算设计提供动机依据。谱收缩的相关分析可参考谱图理论文献[@Chung1997]。

## 跳距预算分配策略：两种方案对比 {#sec:appendix-A5}

基于第 4 章第 [4.3.1](#subsec:ch4-lemma36){reference-type="ref" reference="subsec:ch4-lemma36"} 节的衰减界分析，跳距保留率 $p_l$ 应随跳距 $l$ 单调递减。正文提出两种策略，分别对应不同的参数化方式。

均匀衰减方案（策略1）取指数衰减形式，公比 $r = (p_L/p_1)^{1/(L-1)}$ 由端点唯一确定： $$p_l = p_1 \cdot r^{l-1}, \quad r \in (0,1).
  \label{eq:appA-pl-uniform}$$

谱间隙自适应方案（策略2）以 $\hat{\lambda}_{\text{gap}}$ 为衰减速率参数，等价于策略1取 $r = \exp(-\kappa\hat{\lambda}_{\text{gap}})$ 的自动参数化版本： $$p_l = \max\!\left\{p_{\min},\; p_{\text{base}} \exp\!\bigl(-\kappa\hat{\lambda}_{\text{gap}}(l-1)\bigr)\right\}, \quad \kappa > 0.
  \label{eq:appA-pl-spectral}$$

谱间隙越大衰减越快，无需手动设置公比。在层间保留率差异适中时（$p_{\max}/p_{\min} \leq 2$），精确最优分配 $k_i^{*(l)} \propto \sqrt{p_l}$ 与工程近似 $k_i^{*(l)} \propto p_l$ 的相对偏差不超过20%，可视具体任务场景选择。两种策略的量化对比以第 6 章跳距预算消融实验（表 [6.5](#tab:ch6-hopbudget){reference-type="ref" reference="tab:ch6-hopbudget"}）为依据。

## 谱滤波误差界结论 {#sec:appendix-A6}

当稀疏图 $\hat{G}$ 满足 $\sigma$-谱相似性条件（正文式 [\[eq:ch5-sigma-loewner\]](#eq:ch5-sigma-loewner){reference-type="eqref" reference="eq:ch5-sigma-loewner"}）时，$K_{\text{poly}}$ 阶多项式谱滤波器（Chebyshev 展开系数为 $\{c_k\}$，$C_{T,k}$ 为对应有界常数）的输出误差满足： $$\left\|g(\tilde{\mathcal{L}}_G)X - g(\tilde{\mathcal{L}}_{\hat{G}})X\right\|_F
  \;\leq\; 2(\sigma-1)\cdot\left(\sum_{k=0}^{K_{\text{poly}}}|c_k|\,C_{T,k}\right)\cdot\|X\|_F.
  \label{eq:appA-filter-bound}$$

该误差界的核心含义是：当稀疏图相对原图的谱结构偏差（由参数 $\sigma \geq 1$ 刻画）越小，输出特征矩阵的偏差也越小。对 GCN（$K_{\text{poly}}=1$，$c_1=1$，$C_{T,1}=1$）而言，界简化为 $2(\sigma-1)\|X\|_F$。该结论可由算子范数不等式、Chebyshev 多项式的 Lipschitz 性质与三角不等式的组合推导得出，相关技术背景可参见 Shuman 等（2013）[@Shuman2013GSP] 的谱图信号处理框架。第 6 章实验在多个数据集上对 $\sigma$ 进行了实测验证（见图 [5.3](#fig:ch5-ablation){reference-type="ref" reference="fig:ch5-ablation"}）。

## 训练稳定性的工程经验说明 {#sec:appendix-A7}

算法 5.1 的交替优化框架在六个数据集上的实验均表现为训练曲线平稳下降，未出现系统性发散，体现了良好的训练稳定性。从工程角度看，这一稳定性主要来源于两方面：Phase $\Theta$ 的 LARS 求解每步均局部降低目标值；Phase $W$ 采用衰减学习率的随机梯度更新，在合理学习率范围内不会持续增大目标。训练稳定性的理论分析可参见块坐标下降与随机梯度相关文献[@Tseng2001; @Razaviyayn2013]，实验训练曲线见第 6 章相关图表。

# 算法流程与工程实现补充 {#chap:appendix-B}

本附录提供正文第 4 章与第 5 章中被压缩或省略的完整算法流程，包括对比学习增强路径的三阶段实现细节（附录B.1）以及基于条件化联合误差界的参数反向设计步骤（附录B.2）。两节内容均对正文主线算法形成独立补充，不改变第5章主循环接口。

## 对比学习增强 $\tau(i)$ 的完整三阶段流程 {#sec:appendix-B1}

本节对第 4 章第 [4.2.4](#subsec:ch4-grace){reference-type="ref" reference="subsec:ch4-grace"} 节所描述的对比学习增强路径给出完整实现（正文引用为"附录B.1"）。该路径是算法 [\[alg:ch4-tau-compute\]](#alg:ch4-tau-compute){reference-type="ref" reference="alg:ch4-tau-compute"}手工路径的可选增强生成器，两路输出共享 $[\tau_{\min},\tau_{\text{base}}]$ 数值口径；当残差修正量 $\delta_i=0$ 时，本节算法退化为纯手工路径，不抛弃前序章节建立的理论约束。

### 三阶段流程总览 {#subsec:appB1-overview}

增强路径在算法 [\[alg:ch4-tau-compute\]](#alg:ch4-tau-compute){reference-type="ref" reference="alg:ch4-tau-compute"}输出的手工基线 $\{\tau^{(0)}(i)\}_{i\in V}$ 上叠加由对比嵌入驱动的乘法修正因子，其实现按阶段一、阶段二与阶段三顺序衔接。阶段一承担对比预训练，采用 GRACE[@Zhu2020GRACE] 风格的双视图增强（边掩码 $p_e$、特征掩码 $p_m$ 各自独立采样）训练共享参数 GNN 编码器 $f_\omega$。实现时，编码器输出记为 $z_i^{(r)}$，投影头输出记为 $q_i^{(r)}=g_{\mathrm{proj}}(z_i^{(r)})$，对比损失作用在 $q_i^{(r)}$ 上；正文式 [\[eq:ch4-infonce\]](#eq:ch4-infonce){reference-type="eqref" reference="eq:ch4-infonce"} 为简化后的节点级记号写法，附录在实现层面将投影后的对比向量显式记出。训练结束后丢弃投影头并保留编码器权重 $\omega$。阶段二在固定 $\omega$ 下于原图提取节点嵌入 $Z=f_\omega(G)$，并先由局部对比一致性分数构造节点级代理目标 $\tau^{\mathrm{proxy}}(i)$，再训练残差 MLP $g_\tau$，使增强路径输出 $\tau(i)$ 拟合该代理目标。这样，手工路径 $\tau^{(0)}(i)$ 继续提供稳定基线，而对比嵌入仅负责给出有界的结构修正方向；相应训练目标由代理目标拟合项、残差幅度正则与参数 $L_2$ 正则三项组成（式 [\[eq:appB-loss\]](#eq:appB-loss){reference-type="eqref" reference="eq:appB-loss"}）。阶段三在固定的 $\omega$ 与 $W_\tau$ 上前向推断，将 $\delta_i$ 以乘法方式叠加于手工基线，再经 clip 将 $\tau(i)$ 限制在 $[\tau_{\min},\tau_{\text{base}}]$（式 [\[eq:appB-clip\]](#eq:appB-clip){reference-type="eqref" reference="eq:appB-clip"}）。

阶段一属于离线预训练，不侵入算法 [\[alg:ch5-main\]](#alg:ch5-main){reference-type="ref" reference="alg:ch5-main"}的在线推断热路径。阶段二涉及的 MLP 规模较小（$d+4$ 维输入、$d_h$ 维隐层），训练开销与对比预训练相比可忽略不计。若对比嵌入 $Z=f_\omega(G)$ 在离线阶段预先缓存，则在线 GNN 推理热路径复杂度与算法 [\[alg:ch4-tau-compute\]](#alg:ch4-tau-compute){reference-type="ref" reference="alg:ch4-tau-compute"}相同，不增加额外计算；否则需额外计入一次编码器前向成本。

残差修正的参数化形式为 $$\tau_{\text{raw}}(i) = \tau^{(0)}(i) \cdot \exp\!\bigl(\delta_i\bigr),\quad
  \delta_i = g_\tau\!\bigl([z_i;\,s_i];\,W_\tau\bigr),
  \label{eq:appB-residual}$$

其中 $s_i \in \mathbb{R}^4$ 为算法 [\[alg:ch4-tau-compute\]](#alg:ch4-tau-compute){reference-type="ref" reference="alg:ch4-tau-compute"}输出的四维手工特征向量，$z_i \in \mathbb{R}^d$ 为对比嵌入。$\exp(\delta_i)$ 对基线做乘法修正：$\delta_i>0$ 时增大稀疏惩罚（激进剪枝），$\delta_i<0$ 时减小稀疏惩罚（保留更多邻居）。为与手工路径保持一致的数值口径，clip 将输出写为 $$\tau(i) = \mathrm{clip}\!\left(\tau_{\text{raw}}(i),\ \tau_{\min},\ \tau_{\text{base}}\right).
  \label{eq:appB-clip}$$

为避免训练目标退化为对手工基线的平凡重构，先定义节点 $i$ 的局部对比一致性分数 $$\rho_i=\frac{1}{|\mathcal{N}(i)|}\sum_{j\in\mathcal{N}(i)}\cos(z_i,z_j),
  \qquad
  \widehat{\rho}_i=\mathrm{zscore}(\rho_i),
  \label{eq:appB-rho}$$

并据此构造代理目标 $$\tau^{\mathrm{proxy}}(i)=
  \mathrm{clip}\!\left(\tau^{(0)}(i)\cdot \exp(\beta\widehat{\rho}_i),\ \tau_{\min},\ \tau_{\text{base}}\right),
  \qquad \beta>0.
  \label{eq:appB-proxy}$$

其中 $\mathcal{N}(i)$ 为节点 $i$ 的一阶邻居集；$\mathrm{zscore}(\cdot)$ 在单图节点集 $V$ 上对 $\{\rho_i\}$ 标准化得到 $\{\widehat{\rho}_i\}$。

阶段二的训练目标定义为 $$\mathcal{L}_{\tau} = \frac{1}{|V|}\sum_{i\in V}\!\bigl(\log\tau(i)-\log\tau^{\mathrm{proxy}}(i)\bigr)^2
  + \lambda_\delta\frac{1}{|V|}\sum_{i\in V}\delta_i^2
  + \lambda_W\|W_\tau\|_2^2.
  \label{eq:appB-loss}$$

第一项在对数空间约束增强路径逼近由对比一致性分数诱导的代理目标，等价于乘法意义下对相对偏移的拟合；第二项限制 $g_\tau$ 输出的残差幅度，防止嵌入 $z_i$ 携带的噪声过度放大稀疏度偏移；第三项为参数 $L_2$ 正则。在 $W_\tau$ 收敛后，三者共同使 $\tau(i)$ 在手工路径基线附近形成有界的结构感知修正，而不会退化为对手工路径的平凡重构。

### 完整算法 {#subsec:appB1-algo}

完整的预训练与融合推断流程由算法 [\[alg:appB-grace\]](#alg:appB-grace){reference-type="ref" reference="alg:appB-grace"}统一呈现。

:::: algorithm
::: algorithmic
图 $G=(V,E,X)$，增强概率 $p_e, p_m$，手工特征向量 $\{s_i\}_{i\in V}$（来自算法 [\[alg:ch4-tau-compute\]](#alg:ch4-tau-compute){reference-type="ref" reference="alg:ch4-tau-compute"}的四维输出），手工路径基线 $\{\tau^{(0)}(i)\}_{i\in V}$（来自算法 [\[alg:ch4-tau-compute\]](#alg:ch4-tau-compute){reference-type="ref" reference="alg:ch4-tau-compute"}），嵌入维度 $d$，投影维度 $d'$，MLP隐层维度 $d_h$，温度参数 $t$，正则系数 $\lambda_\delta, \lambda_W$，对比一致性系数 $\beta>0$（式 [\[eq:appB-proxy\]](#eq:appB-proxy){reference-type="eqref" reference="eq:appB-proxy"}），剪枝上下界 $\tau_{\text{base}},\tau_{\min}$，预训练轮数 $T_{\text{pre}}$，MLP训练轮数 $T_{\text{mlp}}$ 增强后的节点级正则系数 $\{\tau(i)\}_{i\in V}$ **// Phase 1：对比预训练** $\xi_{uv}^{(1)},\xi_{uv}^{(2)}\sim\mathrm{Bernoulli}(1-p_e)$ 独立采样；$\tilde{E}_r=\{(u,v)\in E\mid\xi_{uv}^{(r)}=1\}$，$r\in\{1,2\}$ $\zeta_{ij}^{(1)},\zeta_{ij}^{(2)}\sim\mathrm{Bernoulli}(1-p_m)$ 独立采样；$\tilde{X}_{ij}^{(r)}=X_{ij}\cdot\zeta_{ij}^{(r)}$ 构造增强视图 $G_r=(V,\tilde{E}_r,\tilde{X}^{(r)})$，$r\in\{1,2\}$ $Z^{(r)} \leftarrow f_\omega(G_r)$（共享参数GNN编码器）；$q_i^{(r)} \leftarrow g_{\mathrm{proj}}(z_i^{(r)})$（线性投影头） $\mathcal{L}_{\text{total}} \leftarrow \mathrm{SymInfoNCE}(\{q^{(1)},q^{(2)}\}, t)$（对称InfoNCE损失，式 [\[eq:ch4-infonce\]](#eq:ch4-infonce){reference-type="eqref" reference="eq:ch4-infonce"}） $\omega,\, W_{\text{proj}},\, b_{\text{proj}} \leftarrow \mathrm{AdamStep}(\nabla_{\omega,\text{proj}}\mathcal{L}_{\text{total}})$ 丢弃投影头参数 $(W_{\text{proj}}, b_{\text{proj}})$，保留编码器 $\omega$ $Z \leftarrow f_\omega(G)$（原图无增强前向，$Z\in\mathbb{R}^{n\times d}$） 由式 [\[eq:appB-rho\]](#eq:appB-rho){reference-type="eqref" reference="eq:appB-rho"}计算 $\rho_i,\widehat{\rho}_i$ 由式 [\[eq:appB-proxy\]](#eq:appB-proxy){reference-type="eqref" reference="eq:appB-proxy"}计算 $\tau^{\mathrm{proxy}}(i)$ **// Phase 2：训练残差 MLP $g_\tau$** $c_i \leftarrow [z_i;\, s_i]$（拼接对比嵌入与手工特征） $\delta_i \leftarrow g_\tau(c_i;\,W_\tau)$（对数空间残差，式 [\[eq:appB-residual\]](#eq:appB-residual){reference-type="eqref" reference="eq:appB-residual"}） $\tau(i) \leftarrow \mathrm{clip}\!\bigl(\tau^{(0)}(i)\cdot\exp(\delta_i),\;\tau_{\min},\;\tau_{\text{base}}\bigr)$（式 [\[eq:appB-clip\]](#eq:appB-clip){reference-type="eqref" reference="eq:appB-clip"}） $\mathcal{L}_{\tau} \leftarrow$ 按式 [\[eq:appB-loss\]](#eq:appB-loss){reference-type="eqref" reference="eq:appB-loss"}计算（目标为 $\tau^{\mathrm{proxy}}(i)$） $W_\tau \leftarrow \mathrm{AdamStep}(\nabla_{W_\tau}\mathcal{L}_{\tau})$ **// Phase 3：推断最终 $\tau(i)$** $c_i \leftarrow [z_i;\, s_i]$ $\delta_i \leftarrow g_\tau(c_i;\,W_\tau)$ $\tau(i) \leftarrow \mathrm{clip}\!\left(\tau^{(0)}(i)\cdot\exp(\delta_i),\;\tau_{\min},\;\tau_{\text{base}}\right)$ $\{\tau(i)\}_{i\in V}$
:::
::::

### 复杂度说明与启用条件 {#subsec:appB1-complexity}

算法 [\[alg:appB-grace\]](#alg:appB-grace){reference-type="ref" reference="alg:appB-grace"}引入的额外开销均不与算法 [\[alg:ch5-main\]](#alg:ch5-main){reference-type="ref" reference="alg:ch5-main"}的在线推断热路径叠加。对比预训练阶段属于离线一次性成本，开销约为 $O(T_{\text{pre}}\cdot m d L_{\text{enc}})$，在编码器层数 $L_{\text{enc}}$ 与嵌入维度 $d$ 视为常数时与图规模 $m$ 线性相关；原图表示提取一次的开销 $O(m d L_{\text{enc}})$ 同属一次性成本；残差MLP训练的开销为 $O(T_{\text{mlp}}\cdot n(d+4)d_h)$，对应在 $(d+4)$ 维输入、$d_h$ 维隐层上的批量前向与反向。这些新增成本均集中在离线预训练与一次性全节点映射阶段，不改变第5章推理热路径的单次稀疏矩阵乘复杂度 $O(n\bar{k}d)$（式 [\[eq:ch5-tau-range\]](#eq:ch5-tau-range){reference-type="eqref" reference="eq:ch5-tau-range"}）。

对比学习增强路径适合在两类场景中优先尝试。其一，当节点标签存在较明显噪声时，手工路径的显式谱拓扑特征对噪声边较为敏感，而对比嵌入提供的隐式高阶结构先验可在一定程度上补偿谱扰动界的不稳定性；本文在当前实验设置中的观察是，噪声率升高时该路径更容易带来额外收益，但不固定写成统一的启用门限。其二，当图中存在显著隐式高阶结构（如三角形密度分布呈双峰、社区间连接模式复杂）时，纯手工特征路径对此类结构的感知能力有限，引入对比嵌入通常更为适合。若上述条件均不满足，直接使用算法 [\[alg:ch4-tau-compute\]](#alg:ch4-tau-compute){reference-type="ref" reference="alg:ch4-tau-compute"}的手工路径基线即可，两路共享 $[\tau_{\min},\tau_{\text{base}}]$ 输出口径，第5章主循环接口无需修改。两路的定量性能差异见第 6 章消融实验。

## $\tau_{\text{base}}$ 反向设计与 Lanczos 校验步骤 {#sec:appendix-B2}

本节对第 5 章第 [5.2](#sec:ch5-spectral){reference-type="ref" reference="sec:ch5-spectral"} 节式 [\[eq:ch5-budget-lagrange\]](#eq:ch5-budget-lagrange){reference-type="eqref" reference="eq:ch5-budget-lagrange"}的条件化联合误差界给出可操作的反向设计算法（正文引用为"附录B.2"）。反向设计的目标是：给定目标精度 $\varepsilon_{\text{total}}$ 与分层保留率 $\{p_l\}$，通过式 [\[eq:ch5-budget-lagrange\]](#eq:ch5-budget-lagrange){reference-type="eqref" reference="eq:ch5-budget-lagrange"}的链条反推满足精度要求的初始化参数 $\tau_{\text{base}}$，并以 Lanczos 代理量 $\widetilde{\sigma}_{\mathrm{proxy}}$ 做闭环校验。

### 反向设计问题定义 {#subsec:appB2-problem}

式 [\[eq:ch5-budget-lagrange\]](#eq:ch5-budget-lagrange){reference-type="eqref" reference="eq:ch5-budget-lagrange"}（$L$层条件化联合误差界）将三维设计参数 $(\tau_{\text{base}},\{p_l\})$ 与下游GNN输出误差 $\varepsilon_{\text{total}}$ 定量联系起来，在线性化传播假设和附加扰动控制假设成立的条件下提供工程代理参考。直接使用该链条作为初始化信号存在两个工程障碍：代理常数 $C$ 与参考阈值 $E_{\text{threshold,ref}}$ 在理论推导中未精确给出；从节点级局部扰动代理到层级谱相似性参数的聚合尚未完全严格化，直接代入可能导致过于保守的稀疏化。

为此，本节将式 [\[eq:ch5-budget-lagrange\]](#eq:ch5-budget-lagrange){reference-type="eqref" reference="eq:ch5-budget-lagrange"}转化为逐步迭代的反向设计流程：先由工程代理公式给出 $\tau_{\text{base}}$ 初始候选值，再以 $\widetilde{\sigma}_{\mathrm{proxy}}$（固定步数 $K_{\mathrm{Lanczos}}$ 的 Lanczos 近似，开销 $O(K_{\mathrm{Lanczos}}m)$；当 $K_{\mathrm{Lanczos}}$ 取定小常数时可视为与 $m$ 线性）做闭环校验，若不满足目标则将 $\tau_{\text{base}}$ 减半重试，直至达标或超出最大迭代轮次。图结构代理量 $$\alpha(G) := \frac{\displaystyle\sum_{i\in V} \dfrac{d_i/\sum_j d_j}{E_{\text{spectral}}^{(h)}(i)}}{\lambda_{\text{gap}}}
  \label{eq:appB-alpha}$$

（对应正文式 [\[eq:ch5-alphadef\]](#eq:ch5-alphadef){reference-type="eqref" reference="eq:ch5-alphadef"}）作为初始化信号，将 $\tau_{\text{base}}$ 候选值初始化为 $$\tau_{\text{base}}^{(0)} = \frac{\varepsilon_{\text{total}}}{S \cdot \alpha(G)},\qquad
  S = \|X\|_F \cdot \sum_{l=1}^{L} \frac{1}{p_l},
  \label{eq:appB-init}$$

其中 $\lambda_{\text{gap}}$ 为归一化拉普拉斯的谱间隙（可由算法 [\[alg:ch4-tau-compute\]](#alg:ch4-tau-compute){reference-type="ref" reference="alg:ch4-tau-compute"}中 Lanczos 分解输出直接复用）。$C$、$E_{\text{threshold,ref}}$ 等未被理论完全约束的代理参数统一在第 6 章实验设置中说明，不并入式 [\[eq:ch5-budget-lagrange\]](#eq:ch5-budget-lagrange){reference-type="eqref" reference="eq:ch5-budget-lagrange"}的理论主链；算法 [\[alg:appB-paramdesign\]](#alg:appB-paramdesign){reference-type="ref" reference="alg:appB-paramdesign"}中亦统一使用 $E_{\text{threshold,ref}}$ 这一记号，不再另设 $E_{\text{threshold}}$。

### 完整算法 {#subsec:appB2-algo}

:::: algorithm
::: algorithmic
图 $G=(V,E,X)$，目标误差 $\varepsilon_{\text{total}}$，分层保留率 $\{p_l\}_{l=1}^{L}$，特征矩阵 $X\in\mathbb{R}^{n\times D}$，参考阈值 $E_{\text{threshold,ref}}$，数值稳定常数 $\epsilon_{\text{num}}>0$，Lanczos 迭代步数 $K_{\mathrm{Lanczos}}$，滤波器参数 $\{K_{\text{poly}}^{(l)},\,\{c_k^{(l)}\}\}_{l=1}^{L}$（来自第 [5.2](#sec:ch5-spectral){reference-type="ref" reference="sec:ch5-spectral"} 节） $\tau_{\text{base}}$（满足目标精度要求的候选值） **Step 1**：由式 [\[eq:appB-alpha\]](#eq:appB-alpha){reference-type="eqref" reference="eq:appB-alpha"}计算 $\alpha(G)$，再依据式 [\[eq:appB-init\]](#eq:appB-init){reference-type="eqref" reference="eq:appB-init"}初始化 $\tau_{\text{base}}^{(0)}$ $\tau_{\text{base}} \leftarrow \tau_{\text{base}}^{(0)}$ **Step 2**：$\tau(i) \leftarrow \tau_{\text{base}} \cdot E_{\text{threshold,ref}} / \max\!\bigl(E_{\text{spectral}}^{(h)}(i),\epsilon_{\text{num}}\bigr)$，$\forall\,i\in V$ **Step 3**：以 $\{\tau(i)\}$ 调用算法 [\[alg:ch5-main\]](#alg:ch5-main){reference-type="ref" reference="alg:ch5-main"}与算法 [\[alg:ch5-fixed\]](#alg:ch5-fixed){reference-type="ref" reference="alg:ch5-fixed"}，得到稀疏系数矩阵 $\Theta^{\text{fixed}}$ **Step 4**：由 $\Theta^{\text{fixed}}$ 构造对称非负代理邻接矩阵 $\hat{A}=\bigl(|\Theta^{\text{fixed}}|+|\Theta^{\text{fixed}}|^\top\bigr)/2$，度矩阵 $\hat{D}_{ii}=\sum_j \hat{A}_{ij}$，以及归一化拉普拉斯 $\tilde{\mathcal{L}}_{\hat{G}} = I-\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}$ **Step 5**：$\widetilde{\sigma}_{\mathrm{proxy}} \leftarrow \operatorname{LanczosApprox}_{K_{\mathrm{Lanczos}}}(\tilde{\mathcal{L}}_G,\,\tilde{\mathcal{L}}_{\hat{G}})$（式 [\[eq:ch5-tau-mono\]](#eq:ch5-tau-mono){reference-type="eqref" reference="eq:ch5-tau-mono"}的代理估计，开销为 $O(K_{\mathrm{Lanczos}}m)$；当 $K_{\mathrm{Lanczos}}$ 取固定小常数时可简写为 $O(m)$） **break**（满足目标精度，退出迭代） $\tau_{\text{base}} \leftarrow \tau_{\text{base}} / 2$（减半重试） $\tau_{\text{base}}$
:::
::::

Step 5 的 $\widetilde{\sigma}_{\mathrm{proxy}}$ 为谱相似性代理量，其中 $\tilde{\mathcal{L}}_{\hat{G}}$ 由对称非负代理邻接矩阵 $\hat{A}$ 构造；实现中采用固定迭代步数 $K_{\mathrm{Lanczos}}$ 的 Lanczos 近似，其与严格 $\sigma$ 的偏差在第 6 章谱相似性验证部分统一报告。若场景对时间敏感可略去 Step 5，以 Step 2 的估计直接使用；当实验表明理论上界偏保守时，可引入经验放宽系数 $r\ge 1$，将最终取值写为 $\tau_{\text{base}}^{\mathrm{practical}}=\min(r\,\tau_{\text{base}}^{\mathrm{theory}},\,\tau_{\mathrm{cap}})$；其中 $\tau_{\mathrm{cap}}$ 为工程上允许的上限。

### Cora 数值示例 {#subsec:appB2-example}

表 [9.1](#tab:appB-example){reference-type="ref" reference="tab:appB-example"} 展示了在 Cora 数据集（$L=2$ 层GCN，$\gamma=1$，$(p_1,p_2)=(0.6,0.4)$，$\varepsilon_{\text{total}}=0.1$）上运行算法 [\[alg:appB-paramdesign\]](#alg:appB-paramdesign){reference-type="ref" reference="alg:appB-paramdesign"}的逐步计算过程。

::::: threeparttable
::: {#tab:appB-example}
  **参数**                                      **计算过程**                                                                                                    **结果**
  --------------------------------------------- --------------------------------------------------------------------------------------------------------------- ------------------------------------------------------------------------------------------------------------------------------
  $\alpha(G)$                                   $\bigl(\sum(d_i/m)/E_{\text{spectral}}^{(h)}(i)\bigr)/\lambda_{\text{gap}}$                                     $\approx 8.3$
  $S$                                           $\|X\|_F\cdot(1/0.6+1/0.4)$                                                                                     $\approx 4.2\|X\|_F$
  $\tau_{\text{base}}^{\text{theory}}$          $0.1\|X\|_F\cdot E_{\text{threshold,ref}}/(1\cdot 8.3\cdot 4.2\|X\|_F)$                                         $\approx 2.9\times10^{-3}\cdot E_{\text{threshold,ref}}$
  $\tau_{\text{base}}^{\text{practical}}$       $\min(r\,\tau_{\text{base}}^{\text{theory}},\;\tau_{\mathrm{cap}})$                                             经验放宽系数 $r=3$，本例取 $\tau_{\mathrm{cap}}=10^{-4}$；$\approx 8.7\times10^{-6}$（$E_{\text{threshold,ref}}=10^{-3}$时）
  验证：$\widetilde{\sigma}_{\mathrm{proxy}}$   Lanczos近似（式 [\[eq:ch5-tau-mono\]](#eq:ch5-tau-mono){reference-type="eqref" reference="eq:ch5-tau-mono"}）   $1.09$（满足目标 $\varepsilon=0.1$）

  : 算法 [\[alg:appB-paramdesign\]](#alg:appB-paramdesign){reference-type="ref" reference="alg:appB-paramdesign"}参数反向设计实例（Cora，$L=2$，$(p_1,p_2)=(0.6,0.4)$，$\varepsilon_{\text{total}}=0.1$）
:::

::: tablenotes
注：$E_{\text{threshold,ref}}$ 为工程代理参数，其精确取值归入第 6 章实验设置；$r=3$ 的放宽系数用于吸收理论代理界与实测误差之间可能存在的保守偏差，具体取值仍需结合 Lanczos 校验结果确定。
:::
:::::

表中反向设计给出的 $\tau_{\text{base}}\approx 8.7\times10^{-6}$ 与第 4 章经验推荐值（$\tau_{\text{base}}=10^{-5}$）量级吻合，两者相互印证，说明式 [\[eq:ch5-budget-lagrange\]](#eq:ch5-budget-lagrange){reference-type="eqref" reference="eq:ch5-budget-lagrange"}的条件化理论界在实际图数据上具有可信的参考意义。需要特别指出，理论代理界通常偏保守，因此取 $r\ge 1$（如 $r=1.1$ 至 $r=3$）均有合理工程依据，最终参数选择须结合 Lanczos 校验步骤与 $\tau_{\mathrm{cap}}$ 确定。

### 理论松弛说明 {#subsec:appB2-remark}

算法 [\[alg:appB-paramdesign\]](#alg:appB-paramdesign){reference-type="ref" reference="alg:appB-paramdesign"}以工程代理界为基础，通过 Lanczos 校验步骤对 $\tau_{\text{base}}$ 进行实测验证，实验结果见第 6 章。松弛因子 $r$、最大迭代轮次以及 $E_{\text{threshold,ref}}$ 等代理参数统一在第 6 章实验设置中说明，不改变本节算法的流程结构。

# 实验设置、补充结果与复现说明 {#chap:appendix-C}

本附录为第 6 章实验评估提供设置规范、补充结果与复现说明，包括超参数搜索空间与默认配置（附录C.1）、数据划分及随机性控制细节（附录C.2）、补充消融与灵敏度分析（附录C.3）以及复现清单与工程口径说明（附录C.4）。本附录用于汇总和补充正文第 6 章的实验设定，与正文共同构成实验部分的完整说明。

## 超参数搜索空间与默认配置 {#sec:appendix-C1}

本节对应正文第 6 章第 [6.1](#sec:ch6-setup){reference-type="ref" reference="sec:ch6-setup"} 节中的超参数配置子节（正文引用为"附录C.1"）。

超参数搜索采用贝叶斯优化框架（Optuna[@Akiba2019Optuna] TPE[@Bergstra2011TPE]采样器），以验证集准确率为优化目标，每个数据集独立进行50次试验（小数据集每次试验约5--10分钟，ogbn-arxiv约1--2小时）。表 [10.1](#tab:appC-hyperparams){reference-type="ref" reference="tab:appC-hyperparams"}列出了MSAS-GNN的全部关键超参数、搜索范围及Cora数据集上的最优值，供实验复现参考。各超参数均按所属模块归入对应的正文来源节，读者可据此定位相关理论背景。

::::: threeparttable
::: {#tab:appC-hyperparams}
  **超参数**                                **含义**                                       **搜索范围**                           **Cora最优值**      **来源节**
  ----------------------------------------- ---------------------------------------------- -------------------------------------- ------------------- -------------------------------------------------------------------------------------------------
  $L$                                       GNN层数                                        $\{2,\ 3,\ 4\}$                                            第 [6.1](#sec:ch6-setup){reference-type="ref" reference="sec:ch6-setup"} 节
  $d$                                       隐藏维度                                       $\{64,\ 128,\ 256\}$                                       第 [6.1](#sec:ch6-setup){reference-type="ref" reference="sec:ch6-setup"} 节
  $k$                                       稀疏预算（全图均值目标$k_i^{\text{total}}$）   $\{20,\ 30,\ 50,\ 80\}$                                    第 [4.3](#sec:ch4-hop){reference-type="ref" reference="sec:ch4-hop"} 节
  $\tau_{\text{base}}$                      节点级正则系数基础值                           $[10^{-4},10^{-2}]$（对数尺度）        $10^{-3}$           第 [4.2.1](#subsec:ch4-tau-design){reference-type="ref" reference="subsec:ch4-tau-design"} 节
  $\beta_{\mathrm{f}},\beta_{\mathrm{h}}$   同配/异配频率衰减系数                          固定默认                               $1.0,\ 1.0$         第 [4.1.2](#subsec:ch4-freq-design){reference-type="ref" reference="subsec:ch4-freq-design"} 节
  $\beta_{\tau}$                            谱能量到正则强度的衰减系数                     固定默认                                                   第 [4.2.1](#subsec:ch4-tau-design){reference-type="ref" reference="subsec:ch4-tau-design"} 节
  $\gamma,\delta,\eta$                      多因子融合权重（度/核心数/熵）                 固定默认                               $0.5,\ 0.3,\ 0.2$   第 [4.2.2](#subsec:ch4-multifactor){reference-type="ref" reference="subsec:ch4-multifactor"} 节
  $\tau_{\text{NCE}}$                       InfoNCE 温度参数（仅 CL 增强变体使用）         $\{0.03,\ 0.05,\ 0.07,\ 0.1,\ 0.2\}$                       附录 [9.1](#sec:appendix-B1){reference-type="ref" reference="sec:appendix-B1"}
  patience                                  早停窗口长度                                   固定默认                                                   第 6 章第 [6.1](#sec:ch6-setup){reference-type="ref" reference="sec:ch6-setup"} 节
  lr                                        学习率                                         $\{0.001,\ 0.005,\ 0.01\}$                                 ---
  dropout                                   Dropout率                                      $\{0.1,\ 0.3,\ 0.5\}$                                      ---

  : MSAS-GNN超参数配置（以Cora为示例数据集）
:::

::: tablenotes
注：$\tau_{\text{NCE}}$仅在附录 [9.1](#sec:appendix-B1){reference-type="ref" reference="sec:appendix-B1"}所述 CL 增强变体中使用，与节点级正则系数$\tau(i)$不是同一参数对象。$K_{\text{eig}}=50$、$\tau_{\min}=10^{-7}$等实现常数沿用第4章算法默认值。上表展示正文直接涉及的搜索项与固定默认项；完整实验配置已在论文配套代码与补充材料中归档。
:::
:::::

训练策略方面，优化器采用Adam（$\beta_1=0.9$，$\beta_2=0.999$，weight_decay=$5\times10^{-4}$），学习率调度采用CosineAnnealingLR（$T_{\max}=200$，$\eta_{\min}=10^{-5}$）。早停策略依据验证集准确率，连续20轮不提升则终止，最大训练轮数在小数据集（Cora/Citeseer/PubMed/Chameleon/Squirrel）上为200轮，在大规模数据集（ogbn-arxiv）上为50轮。图遍历策略方面，小数据集采用全图训练（full-batch）；ogbn-arxiv 采用随机节点 mini-batch（batch_size=1024），仅在采样节点上计算 Phase $W$ 的损失。

LARS稀疏化求解阶段采用第 5 章算法 [\[alg:ch5-main\]](#alg:ch5-main){reference-type="ref" reference="alg:ch5-main"}的分层 LARS + Phase $W$ 交替优化实现：对每个节点按跳距环层逐步构造稀疏权重，并在验证指标连续20轮不提升时早停。Phase $W$ 中显式参数 $W_\phi$ 以由 $(X,H^*)$ 闭式岭回归构造的线性映射热启动，小图采用 full-batch，ogbn-arxiv 采用上述随机节点 mini-batch。

## 数据划分、随机种子与显著性检验细节 {#sec:appendix-C2}

本节汇总正文第 6 章第 [6.1](#sec:ch6-setup){reference-type="ref" reference="sec:ch6-setup"} 节中涉及数据划分与随机性控制的全部细节，为实验可重复性提供完整说明。

#### 静态图数据划分

静态图实验（Cora、Citeseer、PubMed、Chameleon、Squirrel）统一采用60%/20%/20%随机划分，分别对应训练集、验证集和测试集；除明确采用官方固定划分的数据集外，所有基线方法均在相同划分与相同预处理口径下重新训练，不直接沿用文献报告数字。ogbn-arxiv采用OGB[@Hu2020OGB]官方提供的时间划分------训练节点为2017年及以前，验证节点为2018年，测试节点为2019年及以后，不另行调整，以保证与公开报告结果具有可比性。

正文第 6 章仅报告六个静态节点分类数据集上的实验结果，因此本附录不再展开动态图数据划分与 tgbl-wiki 的补充协议。

#### 随机种子序列

所有静态实验在10个不同随机种子下重复，种子序列为 $$\{42,\ 123,\ 456,\ 789,\ 2021,\ 2022,\ 2023,\ 2024,\ 2025,\ 2026\},$$ 报告均值与标准差（即"均值$\pm$标准差"格式）。上述种子同时控制数据集随机划分、参数初始化和邻居采样过程中的随机性，确保各次运行的差异来源可溯。

#### 显著性检验设置

方法间的显著性差异采用Wilcoxon符号秩检验[@Wilcoxon1945Ranking]（双侧，显著性水平$p<0.05$）。对每对方法，以10个随机种子的测试准确率序列为输入，计算双侧$p$值；所有在正文中标记为"统计显著"的结论均已通过该检验，未通过者在对应表注中以"n.s."标记。选用Wilcoxon符号秩检验而非配对$t$检验的理由在于，该方法对准确率分布不作正态假设，在小样本（$n=10$种子）条件下具有更稳健的第一类错误控制能力。

## 补充消融与灵敏度分析 {#sec:appendix-C3}

本节汇集第 6 章正文因篇幅限制未收录的补充实验结果，包括$\tau(i)$对比学习增强路径的完整对比表、预算分配参数$\xi$的细粒度消融、关键超参数灵敏度曲线的完整版本以及谱相似性代理量的补充分析图，供读者深入了解各模块的鲁棒性。除正文已定义符号外，本节新增实现层参数均在首次出现处说明，并与"符号和缩略语说明"保持一致。

#### $\tau(i)$对比学习增强路径补充表

正文第 [6.3.1](#subsec:ch6-modular){reference-type="ref" reference="subsec:ch6-modular"} 节仅在 Cora 上报告了对比学习增强路径的受控消融。此处将该验证扩展至六个数据集，作为正文结果的补充展示。

表 [10.2](#tab:appC-taupath){reference-type="ref" reference="tab:appC-taupath"}将正文表 [6.4](#tab:ch6-ablation){reference-type="ref" reference="tab:ch6-ablation"}（Cora单数据集）扩展至全部六个数据集，噪声场景统一取30%噪声注入，实验协议与正文第 [6.3.1](#subsec:ch6-modular){reference-type="ref" reference="subsec:ch6-modular"} 节一致。

::::: threeparttable
::: {#tab:appC-taupath}
  **数据集**   **噪声率**   **B5：完整版$\tau(i)$（不含CL）**   **B6：CL增强变体（附录B.1）**   **随机扰动对照**   **SDGNN（B0，参照）**
  ------------ ------------ ----------------------------------- ------------------------------- ------------------ -----------------------
               \%           ±0.7                                **88.6±0.7**                    ±0.9               ±0.9
               \%                                               **81.6**                                           
               \%           ±0.9                                **82.5±0.8**                    ±1.1               ±1.1
               \%                                               **74.2**                                           
               \%           ±0.4                                **89.7±0.4**                    ±0.5               ±0.5
               \%                                               **82.6**                                           
               \%           ±0.9                                **67.7±0.9**                    ±1.1               ±1.1
               \%                                               **59.4**                                           
               \%           ±1.2                                **57.5±1.2**                    ±1.4               ±1.4
               \%                                               **49.9**                                           
               \%           ±0.2                                **75.4±0.2**                    ±0.3               ±0.2
               \%                                               **68.7**                                           

  : $\tau(i)$对比学习增强路径六数据集完整对比（准确率%）
:::

::: tablenotes
注：B5 列对应正文表 [6.4](#tab:ch6-ablation){reference-type="ref" reference="tab:ch6-ablation"}中 B5 配置（谱能量+度中心性+k-core+局部熵+分层跳距，不含CL增强），其 Cora 值（88.3%）与正文消融表直接对齐；B6 为加入 CL 增强后的变体（附录 B.1 算法）；SDGNN（B0）列对应正文主实验基线。30%噪声行仅报告10次均值。
:::
:::::

跨数据集规律与正文Cora结论高度一致。CL增强路径（B6）相对无CL的完整版（B5）在干净图上的提升幅度在0.3--0.6个百分点之间，在30%噪声场景下放大至0.5--0.7个百分点；随机扰动对照在所有数据集上均表现低于B5基础值（-0.5至-0.9个百分点），排除了"任何形式的修正均有帮助"这一弱假设。值得注意的是，异配性图（Chameleon、Squirrel）的增强收益（干净图约+0.5--0.6%）略高于同配性图（约+0.3--0.4%），与第 [4.2.4](#subsec:ch4-grace){reference-type="ref" reference="subsec:ch4-grace"} 节关于对比学习路径可补偿纯手工谱特征之不足、在结构更复杂区域更具修正空间的叙述方向一致。

#### 预算分配参数$\xi$细粒度消融

正文第 [6.3.2](#subsec:ch6-hopbudget){reference-type="ref" reference="subsec:ch6-hopbudget"} 节报告了$\xi\in\{0.5,1.0\}$两档的对比结论。考虑到$\xi$在$[0,1]$上连续变化时各层预算的分配比例存在非线性过渡，此处以步长0.1补充$\xi\in\{0.1,0.2,\ldots,0.9,1.0\}$共10档在Cora与Chameleon上的准确率与额外计算开销曲线。

![预算分配参数$\xi$细粒度消融折线图（Cora与Chameleon）](C-1.pdf){#fig:appC-gamma width="100%"}

#### 关键超参数灵敏度曲线补充版本

正文图 [6.1](#fig:ch6-hyperparam){reference-type="ref" reference="fig:ch6-hyperparam"}展示了 Cora 数据集上 $k$、$\tau_{\text{base}}$ 与 $\tau_{\text{NCE}}$ 三个超参数的灵敏度曲线。此处补充 Chameleon 与 ogbn-arxiv 上对应的灵敏度结果，并追加多因子融合主权重 $\gamma$ 的敏感性分析。

![关键超参数灵敏度补充曲线（Chameleon与ogbn-arxiv）](C-2.pdf){#fig:appC-hyperparam width="100%"}

#### 谱相似性代理量与实测误差补充图

正文第 [6.3.2](#subsec:ch6-hopbudget){reference-type="ref" reference="subsec:ch6-hopbudget"} 节主要报告不同预算分配策略下的准确率与平均近似误差，并未单列随$k$变化的谱相似性表。为避免与正文表 [6.5](#tab:ch6-hopbudget){reference-type="ref" reference="tab:ch6-hopbudget"}混淆，此处单独补充 Citeseer 与 ogbn-arxiv 在不同 $k$ 下的谱相似性核验结果，并与实测近似误差$\varepsilon_{\text{approx}}$对照展示。

表 [10.3](#tab:appC-spectral){reference-type="ref" reference="tab:appC-spectral"}补充Citeseer与ogbn-arxiv两个数据集在相同稀疏预算梯度下的谱相似性代理量$\widetilde{\sigma}_{\mathrm{proxy}}$、按简化系数计算的工程参考值与实测近似误差，供读者与正文相关结果横向对照。

::::: threeparttable
::: {#tab:appC-spectral}
  **数据集**   **预算$k$**   **谱相似性代理量$\widetilde{\sigma}_{\mathrm{proxy}}$**   **按简化系数计算的工程参考值**   **实测$\varepsilon_{\text{approx}}$**   **节点分类准确率（%）**
  ------------ ------------- --------------------------------------------------------- -------------------------------- --------------------------------------- -------------------------
                                                                                                                                                                ±1.2
                                                                                                                                                                ±0.9
                                                                                                                                                                ±0.8
                                                                                                                                                                ±0.8
                                                                                                                                                                ±0.9
                                                                                                                                                                ±0.6
                                                                                                                                                                ±0.5
                                                                                                                                                                ±0.5
                                                                                                                                                                ±0.5
                                                                                                                                                                ±0.5

  : 谱相似性代理量$\widetilde{\sigma}_{\mathrm{proxy}}$验证补充（Citeseer与ogbn-arxiv，$K_{\text{poly}}=3$）
:::

::: tablenotes
注：第二列对大规模图统一报告 Lanczos 代理量$\widetilde{\sigma}_{\mathrm{proxy}}$，不与严格$\sigma$混用。第四列仅作为按简化系数得到的工程参考值，不作为正文误差界（式 [\[eq:ch5-tau-grace\]](#eq:ch5-tau-grace){reference-type="eqref" reference="eq:ch5-tau-grace"}）的严格上界证据；若需核验二者关系，须在统一系数口径下重算，并保证其数值不低于对应的实测$\varepsilon_{\text{approx}}$。与Cora等小规模基准相比，ogbn-arxiv的$\widetilde{\sigma}_{\mathrm{proxy}}$在相同$k$下往往偏高（如$k=50$时1.12 vs. 约1.05），说明大规模稠密图在相同绝对保留邻居数下稀疏化对谱结构的扰动更大，与预期相符；Citeseer的曲线走势与Cora接近，但准确率绝对值偏低（约73--74%），与正文主实验一致。
:::
:::::

图 [10.3](#fig:appC-spectral){reference-type="ref" reference="fig:appC-spectral"}以折线图形式将不同数据集上的$\widetilde{\sigma}_{\mathrm{proxy}}$与实测$\varepsilon_{\text{approx}}$随$k$的变化并排呈现，直观展示代理量在不同图类型上的跟踪精度。

![谱相似性代理量$\widetilde{\sigma}_{\mathrm{proxy}}$补充曲线（Citeseer与ogbn-arxiv）](C-3.pdf){#fig:appC-spectral width="100%"}

## 复现清单与工程口径说明 {#sec:appendix-C4}

本节提供完整的硬件与软件环境记录、效率统计口径约定以及复现所需的关键工程细节，以保证第 6 章所有实验数字的可重复性。

#### 硬件与软件环境

全部实验在高性能计算集群上运行。GPU 方面，小数据集（Cora、Citeseer、PubMed、Chameleon、Squirrel）采用 NVIDIA V100 32GB，大规模数据集（ogbn-arxiv）采用 NVIDIA A100 40GB；CPU 为 Intel Xeon Gold 6248R（48 核，主频 3.0GHz），内存为 256GB DDR4-3200，节点间以 100Gbps InfiniBand 互联。

软件栈为 Ubuntu 20.04 LTS，CUDA 11.8，PyTorch 2.0.1，PyTorch Geometric 2.3.1，超参数搜索使用 Optuna 3.2.0[@Akiba2019Optuna]；其余关键依赖包括 NumPy 1.24.3、SciPy 1.10.1、scikit-learn 1.2.2、NetworkX 3.1。为保证推理时间测量的可重复性，所有效率实验均在单卡单进程环境下执行，排除多卡并行带来的干扰。

#### 效率统计口径

第 6 章第 [6.4](#sec:ch6-efficiency){reference-type="ref" reference="sec:ch6-efficiency"} 节使用三类不可互换的效率统计口径。对于 Cora、Citeseer、PubMed、Chameleon 与 Squirrel，报告单次全图前向时间（ms/次前向）；对于 ogbn-arxiv，报告固定 batch_size=1024 下的批次前向时间（ms/batch）与节点吞吐量（nodes/s）；Break-even 分析使用与对应数据集一致的单次推理时间口径。除非显式说明，不再将时间归一化为"ms/节点"，以避免与正文表 [6.6](#tab:ch6-infer){reference-type="ref" reference="tab:ch6-infer"}混淆。

#### 预处理Break-Even分析口径

Break-even调用次数定义为 $$Q_{\text{be}} = \frac{t_{\text{pre}}}{t_{\text{dense}} - t_{\text{sparse}}},
  \label{eq:appC-breakeven}$$

其中$t_{\text{pre}}$为总预处理时间（含$\tau(i)$计算与LARS路径求解），$t_{\text{dense}}$与$t_{\text{sparse}}$必须在同一统计口径下测量：小数据集取单次全图前向时间，大规模图取遍历测试节点一轮的总推理时间。式 [\[eq:appC-breakeven\]](#eq:appC-breakeven){reference-type="eqref" reference="eq:appC-breakeven"}给出的$Q_{\text{be}}$表示：预处理开销摊销到单次推理收益上，需要调用模型推理至少$Q_{\text{be}}$次方可回本。第 6 章第 [6.4.2](#subsec:ch6-breakeven){reference-type="ref" reference="subsec:ch6-breakeven"} 节中各数据集的$Q_{\text{be}}$数值均按此公式在对应硬件环境下实测计算，不作理论估算代替。

#### 正文范围一致性说明

正文第 6 章仅包含六个静态节点分类数据集上的主实验、消融、效率与可视化分析，本附录亦仅对这些静态实验提供复现补充，不再单列动态图实验说明。

::: bnuindex
## 索引说明 {#索引说明 .unnumbered}

本学位论文编制了创新索引、主题索引、专有名称索引。

## 创新索引 {#创新索引 .unnumbered}

创新索引以学位论文中描述创新内容的词语为目标编制，款目采用**加粗**格式表示，按标目在论文中首次出现的顺序集中排列在索引的最前面。

**三维自适应稀疏参数体系（频率维 $\times$ 节点维 $\times$ 跳距维）**

*以节点修正谱能量 $E_{\mathrm{spectral}}^{(h)}(i)$ 为核心，从频率、节点、跳距三条线索取代 SDGNN 的全局统一正则系数 $\lambda_{\mathrm{reg}}$，在推理期保持 $O(n\bar{k}d)$ 单次稀疏矩阵乘形式。*

**同配系数感知频率权重调整机制**

*基于边级同配率 $h_{\mathrm{edge}}$ 与节点级同配率 $h_i$ 的统一控制变量，对频率打分向量 $\alpha(i)$ 进行差异化修正，消除单一低通假设在异配图上的频率响应错配。*

**修正谱能量 $E_{\mathrm{spectral}}^{(h)}(i)$**

*引入统一控制变量 $h^{\dagger}(i)$ 对各频率分量的 softmax 权重 $w_k(i)$ 做同配/异配感知修正，使同配节点侧重低频、异配节点为高频保留更大权重。第4章相关分析刻画了该量随局部同配环境变化的单调趋势。*

**节点级正则系数 $\tau(i)$ 设计准则**

*依据谱能量与邻居有效信息量的正相关关系，建立"修正谱能量高 $\Rightarrow$ $\tau(i)$ 小 $\Rightarrow$ LARS 截断晚 $\Rightarrow$ 保留邻居多"的单调映射，并融合度中心性 $C_{\mathrm{deg}}(i)$、$k$-core 指数和局部图熵形成多因子节点级参数。*

**节点---跳距联合预算分配**

*将节点级总预算 $k_i^{\mathrm{total}}$（由 $\tau(i)$ 驱动）与各跳距配额 $k_i^{(l)}$（由衰减律驱动）纳入统一框架，使节点维与跳距维的稀疏约束在 LARS 求解器层面通过 `max_iter` 接口协同生效，形成"节点粒度 $\times$ 跳距粒度"的二级稀疏预算体系。*

**跳距预算配额 $k_i^{(l)}$ 的近邻优先分配策略**

*以第二谱半径为衰减基底建立各跳距信息衰减界，依据衰减规律设计近跳密集、远跳稀疏的分层预算分配策略，使近跳关键邻居优先保留。相关分析见第4章第4.3节。*

**$\sigma$-谱相似性误差分析**

*在三维参数联合约束下讨论稀疏化后多项式谱滤波近似误差的条件型分析结果，为第4章和第5章提供跨章节评估结构变化质量的统一度量接口。*

## 主题索引 {#主题索引 .unnumbered}

主题索引以学位论文中重点论述、具有检索价值的重要主题词为标目编制，按拼音排序。

## 专有名称索引 {#专有名称索引 .unnumbered}

专有名称索引以机构名、人名、地名、文献名等专有名称为标目编制，按拼音排序。
:::

::: resume
无。
:::

::: acknowledgements
衷心感谢导师翟铮老师。
:::

[^1]: 对任意阈值 $t>0$，以 Markov 不等式可推导大误差节点集 $\mathcal{S}(t):=\{i:\|\Delta z_i\|_2\geq t\}$ 的比例上界为 $|\mathcal{S}(t)|/n \leq (\mathcal{E}_{\mathrm{approx}}/t)^2$。将表示层误差严格转化为准确率下降界还需额外给定分类头 Lipschitz 常数及各节点 margin 分布，本文在第6章做经验验证。
