"""统一数据集加载入口。"""
import logging
from typing import Tuple
from torch_geometric.data import Data
logger = logging.getLogger(__name__)

def load_dataset(name: str, root: str = "data/raw") -> Tuple[Data, int, int]:
    """按名称加载数据集，返回 (data, num_classes, num_features)。"""
    name = name.lower()
    NAMES = ["cora","citeseer","pubmed","ogbn_arxiv","chameleon","squirrel"]
    if name not in NAMES: raise ValueError(f"Unknown dataset: {name}")
    logger.info(f"[dataset_factory] Loading {name}")
    if name in ("cora","citeseer","pubmed"):
        from msas_gnn.data.planetoid_loader import load_planetoid
        return load_planetoid(name, root)
    elif name in ("chameleon","squirrel"):
        from msas_gnn.data.wikipedia_loader import load_wikipedia
        return load_wikipedia(name, root)
    elif name == "ogbn_arxiv":
        from msas_gnn.data.ogb_loader import load_ogbn_arxiv
        return load_ogbn_arxiv(root)
