"""补充实验：近邻预算几何衰减公比扫描图。"""
import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", default="outputs/results/supplemental_varrho_sweep.json")
    parser.add_argument("--output_dir", default="outputs/figures/supplemental")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if not os.path.exists(args.results_file):
        print(f"文件不存在：{args.results_file}")
        return

    import matplotlib.pyplot as plt

    with open(args.results_file, encoding="utf-8") as f:
        data = json.load(f)
    fig, ax = plt.subplots(figsize=(7, 4))
    for dataset, varrho_data in data.items():
        varrhos = sorted(varrho_data.keys(), key=float)
        means = [varrho_data[varrho]["mean"] * 100 for varrho in varrhos]
        ax.plot([float(varrho) for varrho in varrhos], means, marker="o", label=dataset)
    ax.set_xlabel("varrho")
    ax.set_ylabel("Test Accuracy (%)")
    ax.legend()
    ax.set_title("补充实验：近邻预算几何衰减公比扫描")
    out = os.path.join(args.output_dir, "varrho_sweep.pdf")
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"已保存：{out}")


if __name__ == "__main__":
    main()
