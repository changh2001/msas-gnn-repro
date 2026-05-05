"""正文超参敏感性图（beta_tau / tau_base）。"""
import argparse, json, os, shutil
PARAM_LABELS = {"beta_tau": "β_τ", "tau_base": "τ_base"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file",default="outputs/results/sensitivity_cora.json")
    parser.add_argument("--output_dir",default="outputs/figures")
    parser.add_argument("--paper_output", default="thesis/figures/6-1.pdf")
    args=parser.parse_args()
    if not os.path.exists(args.results_file): print(f"文件不存在：{args.results_file}"); return
    import matplotlib.pyplot as plt
    with open(args.results_file) as f: data=json.load(f)
    params = ["beta_tau", "tau_base"]
    fig, axes = plt.subplots(1, len(params), figsize=(10,4))
    if len(params) == 1:
        axes = [axes]
    for ax, p in zip(axes, params):
        if p not in data: continue
        vals=sorted(data[p].keys(),key=float); means=[data[p][v]["mean"]*100 for v in vals]
        ax.plot(range(len(vals)),means,marker="o"); ax.set_xticks(range(len(vals))); ax.set_xticklabels([str(v) for v in vals],fontsize=8)
        ax.set_title(f"{PARAM_LABELS.get(p, p)}（Cora）"); ax.set_ylabel("Validation Acc (%)")
    plt.tight_layout(); os.makedirs(args.output_dir,exist_ok=True)
    out=os.path.join(args.output_dir,"sensitivity_cora.pdf"); plt.savefig(out,dpi=300,bbox_inches="tight"); plt.close()
    if args.paper_output:
        os.makedirs(os.path.dirname(args.paper_output), exist_ok=True)
        shutil.copyfile(out, args.paper_output)
    print(f"已保存：{out}")
if __name__ == "__main__": main()
