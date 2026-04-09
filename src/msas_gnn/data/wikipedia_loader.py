"""Chameleon/Squirrel 加载（geom_gcn_preprocess=True）。
协议：不使用内置划分mask，由split_manager重新执行60/20/20划分。
此协议下结果不可与官方固定划分文献直接比较。
"""
import os, logging
from torch_geometric.datasets import WikipediaNetwork
logger = logging.getLogger(__name__)

def load_wikipedia(name, root, preprocess=True):
    ds = WikipediaNetwork(root=os.path.join(root, name), name=name, geom_gcn_preprocess=preprocess)
    data = ds[0]
    data.train_mask = data.val_mask = data.test_mask = None
    logger.info(f"[wikipedia] {name}: n={data.num_nodes}, geom_gcn_preprocess={preprocess}")
    return data, ds.num_classes, ds.num_features
