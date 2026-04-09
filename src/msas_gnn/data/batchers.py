"""NeighborLoader封装（ogbn-arxiv mini-batch）。"""
import logging
import torch
from torch_geometric.loader import NeighborLoader
logger = logging.getLogger(__name__)

def get_neighbor_loader(data, input_nodes, batch_size=1024, num_neighbors=None, num_workers=4, shuffle=False):
    if batch_size != 1024:
        logger.warning(f"batch_size={batch_size}偏离论文口径1024，效率数字不对应表6.6")
    if num_neighbors is None:
        num_neighbors = [25, 10]
    return NeighborLoader(data, num_neighbors=num_neighbors, batch_size=batch_size,
                          input_nodes=input_nodes, num_workers=num_workers, shuffle=shuffle)
