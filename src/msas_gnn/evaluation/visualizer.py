"""结果可视化（τ分布图、t-SNE、超参敏感性折线图）。对应论文§6.5。"""
from __future__ import annotations

import copy
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)


def run_visualization(cfg, vis_type):
    if vis_type == "tsne":
        return plot_tsne(cfg)
    if vis_type == "tau_dist":
        return plot_tau_distribution(cfg)
    if vis_type == "sensitivity":
        return plot_sensitivity(cfg)
    raise ValueError(f"Unknown vis_type: {vis_type}")


def _load_vis_data(dataset: str, seed: int):
    from msas_gnn.data.dataset_factory import load_dataset
    from msas_gnn.data.split_manager import load_or_create_split
    from msas_gnn.data.transforms import add_self_loops_transform

    data, num_classes, _ = load_dataset(dataset)
    data = add_self_loops_transform(data)
    if dataset != "ogbn_arxiv":
        train_mask, val_mask, test_mask = load_or_create_split(dataset, data.y, seed)
        data.train_mask, data.val_mask, data.test_mask = train_mask, val_mask, test_mask
    return data, num_classes


def _collect_embeddings(cfg, method_name: str):
    from msas_gnn.config import load_experiment_config
    from msas_gnn.training.msas_trainer import MSASTrainer
    from msas_gnn.training.teacher_trainer import TeacherTrainer

    dataset = cfg.get("dataset", "cora")
    seed = int(cfg.get("seed", 42))
    data, num_classes = _load_vis_data(dataset, seed)
    method = method_name.lower()

    if method == "gcn":
        teacher_cfg = load_experiment_config(dataset, ablation_id="b5", overrides={"seed": seed})
        teacher_cfg["num_classes"] = num_classes
        teacher_data = copy.copy(data)
        return TeacherTrainer(teacher_cfg, device="cpu").train_and_cache(teacher_data, seed)
    if method in {"sdgnn", "sdgnn_compat", "b0"}:
        model_cfg = load_experiment_config(dataset, ablation_id="b0", overrides={"seed": seed})
        return MSASTrainer(model_cfg, device="cpu").run_single_seed(seed, return_embeddings=True)["embeddings"]
    if method in {"sdgnn_pure"}:
        model_cfg = load_experiment_config(dataset, ablation_id="sdgnn_pure", overrides={"seed": seed})
        return MSASTrainer(model_cfg, device="cpu").run_single_seed(seed, return_embeddings=True)["embeddings"]
    if method in {"msas_gnn_b5", "b5", "msas_gnn"}:
        model_cfg = load_experiment_config(dataset, ablation_id="b5", overrides={"seed": seed})
        return MSASTrainer(model_cfg, device="cpu").run_single_seed(seed, return_embeddings=True)["embeddings"]
    raise ValueError(f"Unknown visualization method: {method_name}")


def plot_tsne(cfg, output_dir="outputs/figures"):
    import inspect
    import matplotlib.pyplot as plt
    import torch
    from sklearn.manifold import TSNE

    from msas_gnn.evaluation.metrics import compute_silhouette

    os.makedirs(output_dir, exist_ok=True)
    dataset = cfg.get("dataset", "cora")
    seed = int(cfg.get("seed", 42))
    tsne_seed = int(cfg.get("tsne_seed", seed))
    perplexity = float(cfg.get("perplexity", 30))
    max_iter = int(cfg.get("max_iter", 1000))

    data, num_classes = _load_vis_data(dataset, seed)
    methods = ["gcn", "sdgnn", "msas_gnn_b5"]
    palette = [
        "#000000",
        "#E69F00",
        "#56B4E9",
        "#009E73",
        "#F0E442",
        "#0072B2",
        "#D55E00",
        "#CC79A7",
    ]
    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 5))
    if len(methods) == 1:
        axes = [axes]

    for ax, method_name in zip(axes, methods):
        embeddings = _collect_embeddings(cfg, method_name).detach().cpu().numpy()
        tsne_kwargs = {
            "n_components": 2,
            "random_state": tsne_seed,
            "perplexity": perplexity,
        }
        tsne_sig = inspect.signature(TSNE.__init__)
        if "max_iter" in tsne_sig.parameters:
            tsne_kwargs["max_iter"] = max_iter
        else:
            tsne_kwargs["n_iter"] = max_iter
        tsne = TSNE(
            **tsne_kwargs,
        )
        projected = tsne.fit_transform(embeddings)
        silhouette = compute_silhouette(torch.from_numpy(projected), data.y.cpu())
        for class_id in range(num_classes):
            mask = data.y.cpu().numpy() == class_id
            ax.scatter(
                projected[mask, 0],
                projected[mask, 1],
                s=6,
                alpha=0.6,
                color=palette[class_id % len(palette)],
            )
        ax.set_title(f"{method_name}\nSilhouette={silhouette:.3f}")
        ax.axis("off")
        logger.info("[tsne] %s silhouette=%.4f", method_name, silhouette)

    out = os.path.join(output_dir, f"tsne_{dataset}.pdf")
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("t-SNE图已保存：%s", out)
    return out


def plot_tau_distribution(cfg, output_dir="outputs/figures"):
    import matplotlib.pyplot as plt
    from torch_geometric.utils import degree

    from msas_gnn.adaptive.joint_budget import build_adaptive_params
    from msas_gnn.config import load_experiment_config
    from msas_gnn.spectral.metric_bundle import compute_metric_bundle

    os.makedirs(output_dir, exist_ok=True)
    dataset = cfg.get("dataset", "cora")
    seed = int(cfg.get("seed", 42))
    data, _ = _load_vis_data(dataset, seed)
    bundle = compute_metric_bundle(data.cpu(), K_eig=int(cfg.get("spectral", {}).get("K_eig", 50)), cache_path=f"data/cache/spectral/{dataset}_metrics.pt")
    run_cfg = load_experiment_config(dataset, ablation_id=cfg.get("ablation_id", "b5"), overrides={"seed": seed})
    tau = build_adaptive_params(bundle, run_cfg, data=data.cpu()).tau

    src, dst = data.edge_index
    deg = degree(src[src != dst], num_nodes=data.num_nodes).cpu().numpy()
    tau_np = tau.cpu().numpy()
    rho = float(np.corrcoef(np.log(deg + 1), np.log(tau_np + 1e-10))[0, 1])

    q1 = np.percentile(deg, 25)
    q3 = np.percentile(deg, 75)
    groups = {
        "低度": tau_np[deg < q1],
        "中度": tau_np[(deg >= q1) & (deg < q3)],
        "高度": tau_np[deg >= q3],
    }
    group_means = {name: float(values.mean()) for name, values in groups.items() if len(values) > 0}
    logger.info(
        "[tau_dist] %s rho=%.4f low=%.4e mid=%.4e high=%.4e",
        dataset,
        rho,
        group_means.get("低度", float("nan")),
        group_means.get("中度", float("nan")),
        group_means.get("高度", float("nan")),
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    ax_scatter, ax_box = axes
    ax_scatter.scatter(deg + 1, tau_np, alpha=0.3, s=5, color="#2196F3")
    ax_scatter.set_xscale("log")
    ax_scatter.set_yscale("log")
    ax_scatter.set_xlabel("节点度 d_i")
    ax_scatter.set_ylabel("τ(i)")
    ax_scatter.set_title(f"τ(i) vs 节点度（{dataset}）\nPearson rho={rho:.3f}")

    box_data = [groups["低度"], groups["中度"], groups["高度"]]
    ax_box.boxplot(box_data, labels=["低度", "中度", "高度"], showfliers=False)
    ax_box.set_yscale("log")
    ax_box.set_ylabel("τ(i)")
    ax_box.set_title("按节点度分组的 τ(i) 分布")
    out = os.path.join(output_dir, f"tau_distribution_{dataset}.pdf")
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("τ分布图已保存：%s", out)
    return out


def plot_sensitivity(cfg, param_name="tau_base", param_values=None, accs=None, output_dir="outputs/figures"):
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    if param_values and accs:
        ax.plot(param_values, accs, marker="o")
    ax.set_title(f"敏感性：{param_name}")
    out = os.path.join(output_dir, f"sensitivity_{param_name}.pdf")
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    return out
