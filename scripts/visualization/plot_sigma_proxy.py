"""附录C.3 图C.3（σ̃代理量补充曲线）。"""
import argparse
import json
import os


def _load_results(results_file, results_dir, datasets):
    if os.path.exists(results_file):
        with open(results_file, encoding="utf-8") as fp:
            payload = json.load(fp)
        if isinstance(payload, dict):
            return payload
    data = {}
    for dataset in datasets:
        path = os.path.join(results_dir, f"appendix_sigma_proxy_{dataset}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"文件不存在：{path}")
        with open(path, encoding="utf-8") as fp:
            data[dataset] = json.load(fp)
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", default="outputs/results/appendix_sigma_proxy.json")
    parser.add_argument("--results_dir", default="outputs/results")
    parser.add_argument("--datasets", nargs="+", default=["citeseer", "ogbn_arxiv"])
    parser.add_argument("--output_dir", default="outputs/figures/appendix")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    data = _load_results(args.results_file, args.results_dir, args.datasets)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    palette = {
        "citeseer": "#1f77b4",
        "ogbn_arxiv": "#d62728",
    }
    for dataset in args.datasets:
        rows = data.get(dataset, [])
        if not rows:
            continue
        color = palette.get(dataset, None)
        ks = [row["k"] for row in rows]
        sigma = [row.get("sigma_proxy_mean", row.get("sigma_proxy")) for row in rows]
        eps = [row.get("epsilon_approx_mean", row.get("epsilon_approx")) for row in rows]
        ref = [row.get("engineering_ref_mean", row.get("engineering_ref")) for row in rows]
        axes[0].plot(ks, sigma, marker="o", linewidth=2, label=dataset, color=color)
        axes[1].plot(ks, eps, marker="o", linewidth=2, label=f"{dataset} ε_approx", color=color)
        axes[1].plot(ks, ref, linestyle="--", linewidth=2, label=f"{dataset} ref", color=color)
        axes[1].fill_between(ks, eps, ref, alpha=0.12, color=color)

    axes[0].set_ylabel(r"$\widetilde{\sigma}_{proxy}$")
    axes[0].set_title("Spectral Similarity Proxy")
    axes[0].legend(ncol=2)
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Value")
    axes[1].set_title(r"$\varepsilon_{approx}$ vs Engineering Reference")
    axes[1].legend(ncol=2)
    plt.tight_layout()
    out = os.path.join(args.output_dir, "sigma_proxy.pdf")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"已保存：{out}")
if __name__ == "__main__": main()
