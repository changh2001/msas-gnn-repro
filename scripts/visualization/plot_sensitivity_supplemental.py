"""补充实验超参敏感性图。"""
import argparse
import json
import os

PARAMS = [("k", "k"), ("tau_base", "τ_base"), ("gamma", "γ")]
DATASETS = ["chameleon", "ogbn_arxiv"]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--results_dir",default="outputs/results")
    parser.add_argument("--output_dir",default="outputs/figures/supplemental")
    args=parser.parse_args()
    data = {}
    for ds in DATASETS:
        path = os.path.join(args.results_dir, f"sensitivity_{ds}.json")
        if not os.path.exists(path):
            print(f"文件不存在：{path}")
            return
        with open(path) as f:
            data[ds] = json.load(f)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(DATASETS), len(PARAMS), figsize=(15, 8))
    for row_idx, ds in enumerate(DATASETS):
        for col_idx, (param_id, param_label) in enumerate(PARAMS):
            ax = axes[row_idx][col_idx]
            param_data = data[ds].get(param_id, {})
            if not param_data:
                ax.set_visible(False)
                continue
            xs = sorted(param_data.keys(), key=float)
            ys = [param_data[x]["mean"] * 100 for x in xs]
            ax.plot(range(len(xs)), ys, marker="o")
            ax.set_xticks(range(len(xs)))
            ax.set_xticklabels([str(x) for x in xs], fontsize=8)
            ax.set_title(f"{param_label} ({ds})")
            ax.set_ylabel("Test Acc (%)")
    plt.tight_layout()
    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, "sensitivity_supplemental.pdf")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"已保存：{out}")
if __name__ == "__main__": main()
