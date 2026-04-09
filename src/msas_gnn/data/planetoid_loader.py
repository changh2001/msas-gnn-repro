"""Cora/Citeseer/PubMed 加载。"""
import os, logging
from torch_geometric.datasets import Planetoid
logger = logging.getLogger(__name__)

def load_planetoid(name, root):
    canonical = {"cora":"Cora","citeseer":"CiteSeer","pubmed":"PubMed"}
    nc = canonical[name]
    ds = Planetoid(root=os.path.join(root, name), name=nc)
    data = ds[0]
    logger.info(f"[planetoid] {name}: n={data.num_nodes}, m={data.num_edges}")
    return data, ds.num_classes, ds.num_features
