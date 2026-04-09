"""候选集构造。

- `build_bfs_candidate_sets`：MSAS-GNN / 兼容 B0 使用的分层 BFS 环层候选集；
- `build_sdgnn_candidate_set`：原始 SDGNN 口径的 `K-hop 全保留 + 额外 D-hop fanout 采样`。
"""
import logging
import torch
logger = logging.getLogger(__name__)


def _build_adjacency(data):
    n = int(data.num_nodes)
    ei = data.edge_index
    src, dst = ei[0].tolist(), ei[1].tolist()
    adj = {i: [] for i in range(n)}
    for u, v in zip(src, dst):
        if u != v:
            adj[u].append(v)
    return n, adj

def build_bfs_candidate_sets(
    data,
    L=3,
    max_candidates=None,
    seed=42,
    keep_complete_hops=None,
    sampled_max_candidates=None,
):
    """构造所有节点1~L跳BFS候选集。返回List[Dict[int,List[int]]]，长度L。"""
    n, adj = _build_adjacency(data)
    rng = torch.Generator(); rng.manual_seed(seed)
    candidate_sets = []
    keep_complete_hops = L if keep_complete_hops is None else int(keep_complete_hops)
    for l in range(1, L+1):
        hop_l = {}
        for i in range(n):
            visited = {i}; frontier = [i]
            for _ in range(l-1):
                nf = []
                for u in frontier:
                    for v in adj[u]:
                        if v not in visited: visited.add(v); nf.append(v)
                frontier = nf
            cands = []
            for u in frontier:
                for v in adj[u]:
                    if v not in visited: cands.append(v); visited.add(v)
            hop_cap = None
            if l > keep_complete_hops and sampled_max_candidates is not None:
                hop_cap = int(sampled_max_candidates)
            elif max_candidates:
                hop_cap = int(max_candidates)
            if hop_cap and len(cands) > hop_cap:
                idx = torch.randperm(len(cands), generator=rng)[:hop_cap].tolist()
                cands = [cands[j] for j in idx]
            hop_l[i] = cands
        candidate_sets.append(hop_l)
    logger.debug(
        "候选集构造完成 L=%s n=%s keep_complete_hops=%s sampled_max_candidates=%s",
        L,
        n,
        keep_complete_hops,
        sampled_max_candidates,
    )
    return candidate_sets


def build_sdgnn_candidate_set(
    data,
    base_hops=2,
    extra_hops=0,
    fanouts=None,
    seed=42,
):
    """构造原始 SDGNN 口径的平坦候选集。

    规则对齐 arXiv:2410.19723v2：
    - 先保留全部 `K`-hop 内邻居；
    - 再从第 `K+1` 跳开始做 `D` 轮递归采样，每轮使用对应 fanout。
    """

    n, adj = _build_adjacency(data)
    rng = torch.Generator()
    rng.manual_seed(seed)
    fanouts = list(fanouts or [])
    candidate_set = {}

    for i in range(n):
        visited = {i}
        frontier = [i]
        candidates = []

        for _ in range(int(base_hops)):
            next_frontier = []
            for u in frontier:
                for v in adj[u]:
                    if v in visited:
                        continue
                    visited.add(v)
                    next_frontier.append(v)
                    candidates.append(v)
            frontier = next_frontier

        sampled_frontier = frontier
        for depth in range(int(extra_hops)):
            if not sampled_frontier:
                break
            fanout = None
            if fanouts:
                fanout = int(fanouts[min(depth, len(fanouts) - 1)])
            next_frontier = []
            for u in sampled_frontier:
                neighs = [v for v in adj[u] if v not in visited]
                if fanout is not None and fanout > 0 and len(neighs) > fanout:
                    idx = torch.randperm(len(neighs), generator=rng)[:fanout].tolist()
                    neighs = [neighs[j] for j in idx]
                for v in neighs:
                    if v in visited:
                        continue
                    visited.add(v)
                    next_frontier.append(v)
                    candidates.append(v)
            sampled_frontier = next_frontier

        candidate_set[i] = candidates

    logger.info(
        "SDGNN原始候选集构造完成 n=%s K=%s D=%s fanouts=%s",
        n,
        base_hops,
        extra_hops,
        fanouts,
    )
    return candidate_set
