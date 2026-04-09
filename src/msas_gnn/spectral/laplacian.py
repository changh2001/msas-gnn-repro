"""归一化拉普拉斯 L̃ = I - D^{-1/2}AD^{-1/2}。对应论文§3.1.1式(3.1)-(3.3)。"""
import logging
import torch
from torch_geometric.utils import get_laplacian
logger = logging.getLogger(__name__)

def compute_normalized_laplacian_scipy(data):
    """计算归一化拉普拉斯（scipy格式，供Lanczos使用）。复杂度O(m)。"""
    import scipy.sparse as sp, numpy as np
    n = data.num_nodes
    lap_ei, lap_ew = get_laplacian(data.edge_index, normalization="sym", num_nodes=n)
    row, col = lap_ei[0].numpy(), lap_ei[1].numpy()
    vals = lap_ew.numpy()
    return sp.coo_matrix((vals,(row,col)), shape=(n,n)).tocsr()

def compute_normalized_laplacian(data):
    """归一化拉普拉斯（torch稀疏格式）。"""
    n = data.num_nodes
    lap_ei, lap_ew = get_laplacian(data.edge_index, normalization="sym", num_nodes=n)
    return torch.sparse_coo_tensor(lap_ei, lap_ew, (n,n)).coalesce()
