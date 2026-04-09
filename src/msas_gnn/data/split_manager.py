"""60/20/20分层随机划分管理。"""
import os, logging
import torch, numpy as np
from sklearn.model_selection import train_test_split
logger = logging.getLogger(__name__)

def stratified_split(y, train_ratio=0.6, val_ratio=0.2, seed=42):
    """按类别分层的60/20/20随机划分（论文§6.1）。"""
    n = y.shape[0]; idx = np.arange(n); y_np = y.cpu().numpy()
    test_ratio = 1.0 - train_ratio - val_ratio
    try:
        tr, tmp = train_test_split(idx, test_size=val_ratio+test_ratio, random_state=seed, stratify=y_np)
        vl, ts = train_test_split(tmp, test_size=test_ratio/(val_ratio+test_ratio), random_state=seed, stratify=y_np[tmp])
    except ValueError:
        logger.warning("分层失败，退化为随机划分")
        tr, tmp = train_test_split(idx, test_size=val_ratio+test_ratio, random_state=seed)
        vl, ts = train_test_split(tmp, test_size=test_ratio/(val_ratio+test_ratio), random_state=seed)
    tm = torch.zeros(n, dtype=torch.bool); vm = torch.zeros(n, dtype=torch.bool); tsm = torch.zeros(n, dtype=torch.bool)
    tm[tr]=True; vm[vl]=True; tsm[ts]=True
    return tm, vm, tsm

def load_or_create_split(name, y, seed, split_dir="data/splits", train_ratio=0.6, val_ratio=0.2):
    os.makedirs(split_dir, exist_ok=True)
    p = os.path.join(split_dir, f"{name}_seed{seed}.pt")
    if os.path.exists(p):
        s = torch.load(p)
        return s["train_mask"], s["val_mask"], s["test_mask"]
    tm, vm, tsm = stratified_split(y, train_ratio, val_ratio, seed)
    torch.save({"train_mask":tm,"val_mask":vm,"test_mask":tsm}, p)
    return tm, vm, tsm
