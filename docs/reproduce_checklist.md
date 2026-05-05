# 完整复现检查清单

## 环境与数据
- [ ] Python 3.10, PyTorch 2.0.1, PyG 2.3.1, OGB 1.3.6
- [ ] 6个数据集下载完成
- [ ] 10组60/20/20划分生成
- [ ] 泄漏检查通过（train/val/test 互不重叠，至少检查 Chameleon/Squirrel 全部10种子）

## 预处理
- [ ] MetricBundle计算完成（data/cache/spectral/）
- [ ] Lanczos精度验证通过（误差<1e-4）
- [ ] H*缓存完整（60个文件）

## 主实验（论文表6.2/6.3）
- [ ] Cora: 88.0±0.7% (p=0.002)
- [ ] Citeseer: 82.1±0.9% (p=0.001)
- [ ] PubMed: 89.4±0.4% (p=0.048)
- [ ] ogbn-arxiv: 75.13±0.23% (p=0.009)
- [ ] Chameleon: 65.6±0.9% (p=0.003)
- [ ] Squirrel: 55.9±1.2% (p=0.011)
- [ ] GraphSAINT/NodeFormer/DIFFormer/SGFormer/NAGphormer 补充基线日志完整

## 消融（论文表6.4）
- [ ] B0→B5单调递增
- [ ] B5 > B5-frozen
- [ ] B2 > B2-RND

## 效率（论文表6.6）
- [ ] 平均加速比~8.0x（vs GCN，四数据集）

## 摊销（论文表6.7）
- [ ] Cora Q_be≈32727, PubMed Q_be≈3543, ogbn-arxiv Q_be≈464

## 可视化
- [ ] τ分布 Pearson $r_{\mathrm{Pearson}}\approx-0.68$
- [ ] t-SNE Silhouette: 约0.44/0.51/0.57
