"""k-core指数（BFS剥皮算法）。对应论文§3.2.2算法3.2。复杂度O(m)。"""
import logging, collections
import torch
logger = logging.getLogger(__name__)

def compute_kcore(data) -> torch.Tensor:
    """返回每个节点的k-core指数。孤立节点core=0；完全图core=n-1。"""
    n = data.num_nodes; ei = data.edge_index
    src, dst = ei[0].tolist(), ei[1].tolist()
    adj = {i: set() for i in range(n)}
    for u, v in zip(src, dst):
        if u != v: adj[u].add(v); adj[v].add(u)
    deg = {i: len(adj[i]) for i in range(n)}
    core = [0] * n
    remaining = set(range(n))
    k = 1
    while remaining:
        changed = True
        while changed:
            changed = False
            to_remove = [v for v in remaining if deg[v] < k]
            for v in to_remove:
                core[v] = k - 1; remaining.discard(v)
                for u in adj[v]:
                    if u in remaining: deg[u] -= 1
                changed = True
        if remaining:
            for v in remaining:
                if deg[v] >= k: core[v] = k
            k += 1
        else: break
    return torch.tensor(core, dtype=torch.float32)
