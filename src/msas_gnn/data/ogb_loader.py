"""ogbn-arxiv 加载（OGB官方时间划分）。"""
import os, logging
import torch
logger = logging.getLogger(__name__)

def load_ogbn_arxiv(root):
    from ogb.nodeproppred import PygNodePropPredDataset
    ds = PygNodePropPredDataset(name="ogbn-arxiv", root=os.path.join(root,"ogbn_arxiv"))
    data = ds[0]; split_idx = ds.get_idx_split()
    n = data.num_nodes
    for split_name, attr in [("train","train_mask"),("valid","val_mask"),("test","test_mask")]:
        mask = torch.zeros(n, dtype=torch.bool)
        mask[split_idx[split_name]] = True
        setattr(data, attr, mask)
    if data.y.dim()==2: data.y = data.y.squeeze(1)
    logger.info(f"[ogb] ogbn-arxiv: n={data.num_nodes}")
    return data, ds.num_classes, ds.num_node_features
