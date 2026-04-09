# 效率测量协议

## 三类口径（严格分离，不可混用）

| 口径 | 函数 | 说明 |
|------|------|------|
| 1. 热路径诊断 | `infer_latency_sparse()` | 稀疏矩阵乘主项，不直接作为论文表6.6 |
| 2. 论文对齐 | `infer_latency_paper_protocol()` / `infer_latency_sparse_paper_protocol()` | 表6.6/6.7必须使用此口径 |
| 3. 完整端到端 | `infer_latency_end_to_end()` | 含完整链路 |

## 测量规范

- warm-up：10次
- 重复：100次，取中位数
- CUDA 同步：`torch.cuda.synchronize()`
- 显存：`torch.cuda.max_memory_allocated()`

## 论文预期值

| 数据集 | 推理时延 | 口径 |
|--------|---------|------|
| Cora | 2.1ms | ms/次全图前向 |
| ogbn-arxiv | 9.2ms | ms/batch (bs=1024) |

补充说明：

- `GCN/SGC/PPRGo/GLNN` 走稠密模型前向口径；
- `SDGNN/MSAS-GNN` 走固定 `Θ^{fixed}` 与离线缓存 `Φ̃` 的稀疏推理口径；
- 对 ogbn-arxiv，稀疏方法先测一次全图稀疏矩阵乘，再按 `ceil(n/1024)` 归一化为等效 `ms/batch`。

## Break-even 公式

```
Q_be = t_pre / (t_dense - t_sparse)
```
