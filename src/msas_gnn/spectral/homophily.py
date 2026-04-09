"""描述性边一致性统计工具。

当前正文主线不消费该统计量；保留本模块仅用于数据集描述与回溯性分析。
"""
import logging
import torch
from torch_geometric.data import Data
logger = logging.getLogger(__name__)

def compute_edge_homophily(data: Data, observed_mask=None) -> float:
    """h_edge = |{(u,v)∈E : y_u=y_v}| / |E|。边界：空图返回0。"""
    ei = data.edge_index; y = data.y
    if ei.shape[1] == 0: logger.warning("空图，h_edge=0"); return 0.0
    src, dst = ei[0], ei[1]
    valid = src != dst
    if observed_mask is not None:
        valid = valid & observed_mask[src] & observed_mask[dst]
    src, dst = src[valid], dst[valid]
    if src.numel() == 0:
        logger.warning("无可用有标签边，h_edge=0")
        return 0.0
    h = (y[src] == y[dst]).float().mean().item()
    logger.debug(f"h_edge={h:.4f}"); return h

def compute_node_homophily(data: Data, observed_mask=None):
    """h_i = |{j∈N(i):y_j=y_i}|/d_i。孤立节点h_i=0。"""
    n, ei, y = data.num_nodes, data.edge_index, data.y
    src, dst = ei[0], ei[1]
    valid = src != dst
    if observed_mask is not None:
        valid = valid & observed_mask[src] & observed_mask[dst]
    src, dst = src[valid], dst[valid]
    same = (y[src] == y[dst]).float()
    deg = torch.zeros(n); deg.scatter_add_(0, src, torch.ones(src.shape[0]))
    sc = torch.zeros(n); sc.scatter_add_(0, src, same)
    return torch.where(deg > 0, sc/deg, torch.zeros_like(sc))
