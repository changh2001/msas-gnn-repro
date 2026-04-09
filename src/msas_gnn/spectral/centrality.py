"""度中心性 C_deg(i) = d_i / (n - 1)。对应论文§3.2.2式(3.11)。"""
import logging
import torch
from torch_geometric.utils import degree
logger = logging.getLogger(__name__)

def compute_degree_centrality(data) -> torch.Tensor:
    """C_deg∈[0,1]。按论文定义忽略自环。"""
    n = data.num_nodes
    src, dst = data.edge_index
    mask = src != dst
    deg = degree(src[mask], num_nodes=n).float()
    denom = max(n - 1, 1)
    C = deg / float(denom)
    logger.debug(f"C_deg mean={C.mean():.4f}"); return C
