"""推理效率测量：时延、显存、加速比。
警告：三类口径API严格分离，不可混用！
口径1 infer_latency_sparse()：热路径诊断；口径2 infer_latency_paper_protocol()：表6.6/6.7必须用此口径；
口径3 infer_latency_end_to_end()：完整端到端。
测量规范：warm-up 10次，重复100次取中位数，CUDA同步。
"""
import logging, time
import numpy as np
import torch
logger = logging.getLogger(__name__)

def infer_latency_sparse(theta_fixed, phi_tilde, warmup=10, repeat=100, device="cuda"):
    """口径1：热路径诊断，不直接作为论文表6.6。"""
    theta = theta_fixed.theta.to(device); phi = phi_tilde.to(device)
    for _ in range(warmup):
        torch.sparse.mm(theta.t(),phi)
        if device=="cuda": torch.cuda.synchronize()
    ts = []
    for _ in range(repeat):
        t0=time.perf_counter(); torch.sparse.mm(theta.t(),phi)
        if device=="cuda": torch.cuda.synchronize()
        ts.append(time.perf_counter()-t0)
    ms=[t*1000 for t in ts]
    return {"median_ms":float(np.median(ms)),"protocol":"sparse_main_term"}


def infer_latency_sparse_paper_protocol(
    theta_fixed,
    phi_tilde,
    num_nodes,
    batch_size=1024,
    warmup=10,
    repeat=100,
    is_large_graph=False,
    device="cuda",
):
    """固定Θ/Φ̃的论文对齐效率口径。

    小图直接报告一次全图稀疏矩阵乘时间；ogbn-arxiv 等大图将一次全图稀疏乘
    归一化为等效的 `ms/batch(bs=1024)`，以保持与表6.6/6.7一致的比较口径。
    """
    theta = theta_fixed.theta.to(device)
    phi = phi_tilde.to(device)
    for _ in range(warmup):
        torch.sparse.mm(theta.t(), phi)
        if device == "cuda":
            torch.cuda.synchronize()
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        torch.sparse.mm(theta.t(), phi)
        if device == "cuda":
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    median_s = float(np.median(ts))
    if is_large_graph:
        n_batches = max(int(np.ceil(float(num_nodes) / float(batch_size))), 1)
        batch_s = median_s / n_batches
        return {
            "median_ms_per_batch": batch_s * 1000.0,
            "nodes_per_sec": float(batch_size / (batch_s + 1e-10)),
            "batch_size": int(batch_size),
            "full_graph_ms": median_s * 1000.0,
            "protocol": "paper_large_graph_sparse_equivalent",
        }
    return {
        "median_ms": median_s * 1000.0,
        "protocol": "paper_small_graph_sparse",
    }

def infer_latency_paper_protocol(model, data, batch_size=1024, warmup=10, repeat=100, is_large_graph=False, device="cuda"):
    """口径2（论文对齐口径）：表6.6/6.7必须使用此口径。
    小图：ms/次全图前向；大图（ogbn-arxiv）：ms/batch(bs=1024)+nodes/s。
    """
    model=model.to(device); model.eval()
    if is_large_graph:
        from msas_gnn.data.batchers import get_neighbor_loader
        loader = get_neighbor_loader(data, data.test_mask.nonzero(as_tuple=False).squeeze(), batch_size=batch_size)
        for i,b in enumerate(loader):
            if i>=warmup: break
            with torch.no_grad(): model(b.x.to(device), b.edge_index.to(device))
            if device=="cuda": torch.cuda.synchronize()
        ts=[]
        for i,b in enumerate(loader):
            if i>=repeat: break
            b=b.to(device); t0=time.perf_counter()
            with torch.no_grad(): model(b.x, b.edge_index)
            if device=="cuda": torch.cuda.synchronize()
            ts.append(time.perf_counter()-t0)
        ms=[t*1000 for t in ts]
        return {"median_ms_per_batch":float(np.median(ms)),"nodes_per_sec":float(batch_size/(np.median(ts)+1e-10)),"batch_size":batch_size,"protocol":"paper_large_graph"}
    else:
        data=data.to(device)
        for _ in range(warmup):
            with torch.no_grad(): model(data.x,data.edge_index)
            if device=="cuda": torch.cuda.synchronize()
        ts=[]
        for _ in range(repeat):
            t0=time.perf_counter()
            with torch.no_grad(): model(data.x,data.edge_index)
            if device=="cuda": torch.cuda.synchronize()
            ts.append(time.perf_counter()-t0)
        ms=[t*1000 for t in ts]
        return {"median_ms":float(np.median(ms)),"protocol":"paper_small_graph"}

def infer_latency_end_to_end(model, data, warmup=10, repeat=100, device="cuda"):
    """口径3：完整端到端。"""
    model=model.to(device); data=data.to(device); model.eval()
    for _ in range(warmup):
        with torch.no_grad(): model(data.x,data.edge_index)
        if device=="cuda": torch.cuda.synchronize()
    ts=[]
    for _ in range(repeat):
        t0=time.perf_counter()
        with torch.no_grad(): model(data.x,data.edge_index)
        if device=="cuda": torch.cuda.synchronize()
        ts.append(time.perf_counter()-t0)
    return {"median_ms":float(np.median([t*1000 for t in ts])),"protocol":"end_to_end"}

def measure_memory(model, data, device="cuda"):
    """显存：max_memory_allocated()，含参数+前向临时张量，不含梯度缓存。"""
    if device!="cuda" or not torch.cuda.is_available(): return {"peak_memory_mb":float("nan")}
    torch.cuda.reset_peak_memory_stats()
    model=model.to(device); data=data.to(device); model.eval()
    with torch.no_grad(): model(data.x,data.edge_index)
    return {"peak_memory_mb":float(torch.cuda.max_memory_allocated()/1024/1024)}


def measure_sparse_memory(theta_fixed, phi_tilde, device="cuda"):
    """固定Θ/Φ̃下的稀疏推理显存。"""
    if device != "cuda" or not torch.cuda.is_available():
        return {"peak_memory_mb": float("nan")}
    torch.cuda.reset_peak_memory_stats()
    theta = theta_fixed.theta.to(device)
    phi = phi_tilde.to(device)
    torch.sparse.mm(theta.t(), phi)
    torch.cuda.synchronize()
    return {"peak_memory_mb": float(torch.cuda.max_memory_allocated() / 1024 / 1024)}


def count_model_parameters(model):
    """统计稠密模型参数量。"""
    total = int(sum(param.numel() for param in model.parameters()))
    trainable = int(sum(param.numel() for param in model.parameters() if param.requires_grad))
    return {
        "parameter_count": total,
        "parameter_count_trainable": trainable,
        "parameter_count_millions": float(total / 1_000_000.0),
        "parameter_count_protocol": "dense_model_parameters",
    }


def count_sparse_inference_parameters(theta_fixed, extra_dense_params=0):
    """统计稀疏推理路径保留的参数量。

    默认将固化后的稀疏权重 `Θ^{fixed}` 非零项视为主参数量，并可选叠加
    线性分类头等少量稠密参数。
    """
    theta = theta_fixed.theta
    if theta.layout == torch.sparse_csr:
        nnz = int(theta.values().numel())
    elif getattr(theta, "is_sparse", False):
        nnz = int(theta._nnz())
    else:
        nnz = int(torch.count_nonzero(theta).item())
    total = int(nnz + int(extra_dense_params))
    return {
        "parameter_count": total,
        "parameter_count_sparse_nnz": nnz,
        "parameter_count_millions": float(total / 1_000_000.0),
        "parameter_count_protocol": "fixed_sparse_theta_nnz_plus_head",
    }


def build_baseline_model(method_name, num_features, num_classes, baseline_cfg=None):
    """按配置实例化论文表口径下的基线模型。"""
    from msas_gnn.baselines.registry import get_baseline

    baseline_cfg = dict(baseline_cfg or {})
    return get_baseline(
        method_name,
        in_channels=num_features,
        out_channels=num_classes,
        hidden_channels=int(baseline_cfg.get("hidden_dim", 64)),
        dropout=float(baseline_cfg.get("dropout", 0.5)),
        num_layers=int(baseline_cfg.get("layers", 2)),
        K=int(baseline_cfg.get("K", 2)),
        alpha=float(baseline_cfg.get("alpha", 0.1)),
        structural_neighbors=int(baseline_cfg.get("structural_neighbors", 2)),
    )


def benchmark_sparse_method(
    cfg,
    ablation_id,
    seed=42,
    batch_size=1024,
    warmup=10,
    repeat=100,
    device="cuda",
):
    """训练并测量固定稀疏推理路径（SDGNN / MSAS-GNN）。"""
    from msas_gnn.config import load_experiment_config
    from msas_gnn.training.msas_trainer import MSASTrainer

    dataset = cfg.get("dataset", "cora")
    run_cfg = load_experiment_config(dataset, ablation_id=ablation_id, overrides={"seed": seed})
    artifacts = MSASTrainer(run_cfg, device=device).run_single_seed(seed, return_artifacts=True)
    is_large_graph = dataset == "ogbn_arxiv"
    latency = infer_latency_sparse_paper_protocol(
        artifacts["theta_fixed"],
        artifacts["phi_tilde"],
        num_nodes=artifacts["phi_tilde"].shape[0],
        batch_size=batch_size,
        warmup=warmup,
        repeat=repeat,
        is_large_graph=is_large_graph,
        device=device,
    )
    memory = measure_sparse_memory(artifacts["theta_fixed"], artifacts["phi_tilde"], device=device)
    num_classes = int(getattr(artifacts["data"], "num_classes", 0) or int(artifacts["data"].y.max().item()) + 1)
    head_params = int(artifacts["phi_tilde"].shape[1] * num_classes + num_classes)
    parameter_stats = count_sparse_inference_parameters(artifacts["theta_fixed"], extra_dense_params=head_params)
    stage_times = dict(artifacts.get("stage_times", {}))
    preprocess_seconds = float(
        stage_times.get("spectral_metrics", 0.0)
        + stage_times.get("adaptive_params", 0.0)
        + stage_times.get("teacher_cache", 0.0)
        + stage_times.get("alternating_opt", 0.0)
    )
    return {
        **latency,
        **memory,
        **parameter_stats,
        "k_bar": float(artifacts["theta_fixed"].k_bar),
        "ablation_id": ablation_id,
        "preprocess_seconds": preprocess_seconds,
        "preprocess_breakdown": {
            key: float(value)
            for key, value in stage_times.items()
            if key in {"spectral_metrics", "adaptive_params", "teacher_cache", "alternating_opt"}
        },
    }


def run_efficiency_benchmark(cfg, methods=None):
    """供 API 直接调用的轻量效率基准。

    说明：
    - 该入口复用与 `scripts/experiments/run_efficiency.py` 相同的论文对齐口径；
    - 默认比较正文表6.6中的六个方法；
    - 更完整的批量复现实验仍建议走 scripts 入口。
    """
    from msas_gnn.config import load_yaml
    from msas_gnn.data.dataset_factory import load_dataset
    from msas_gnn.data.transforms import add_self_loops_transform

    dataset = cfg.get("dataset", "cora")
    eff_cfg = cfg.get("efficiency", {})
    methods = methods or eff_cfg.get("methods") or [
        "gcn",
        "sgc",
        "pprgo",
        "glnn",
        "sdgnn",
        "msas_gnn",
        "graphsaint",
        "nodeformer",
        "difformer",
        "sgformer",
        "nagphormer",
    ]
    batch_size = int(cfg.get("train", {}).get("batch_size", cfg.get("batch_size", 1024)))
    warmup = int(eff_cfg.get("warmup_runs", 10))
    repeat = int(eff_cfg.get("repeat_runs", 100))
    seed = int(cfg.get("seed", 42))
    device = cfg.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    data, num_classes, num_features = load_dataset(dataset)
    data = add_self_loops_transform(data)
    is_large_graph = dataset == "ogbn_arxiv"
    results = {}
    for method_name in methods:
        if method_name in {"sdgnn", "sdgnn_compat", "b0"}:
            results[method_name] = benchmark_sparse_method(
                cfg,
                ablation_id="b0",
                seed=seed,
                batch_size=batch_size,
                warmup=warmup,
                repeat=repeat,
                device=device,
            )
        elif method_name in {"sdgnn_pure"}:
            results[method_name] = benchmark_sparse_method(
                cfg,
                ablation_id="sdgnn_pure",
                seed=seed,
                batch_size=batch_size,
                warmup=warmup,
                repeat=repeat,
                device=device,
            )
        elif method_name in {"msas_gnn", "msas_gnn_b5", "b5"}:
            results[method_name] = benchmark_sparse_method(
                cfg,
                ablation_id="b5",
                seed=seed,
                batch_size=batch_size,
                warmup=warmup,
                repeat=repeat,
                device=device,
            )
        else:
            baseline_cfg = load_yaml(f"configs/teachers/{method_name}.yaml")
            model = build_baseline_model(method_name, num_features, num_classes, baseline_cfg=baseline_cfg)
            latency = infer_latency_paper_protocol(
                model,
                data,
                batch_size=batch_size,
                warmup=warmup,
                repeat=repeat,
                is_large_graph=is_large_graph,
                device=device,
            )
            memory = measure_memory(model, data, device=device)
            parameter_stats = count_model_parameters(model)
            results[method_name] = {**latency, **memory, **parameter_stats}
        logger.info("[efficiency] %s/%s -> %s", dataset, method_name, results[method_name])
    return results
