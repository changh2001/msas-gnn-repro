"""正文超参敏感性三联图。"""
import argparse, json, os
PARAM_LABELS = {"tau_base": "τ_base", "k": "k", "xi_budget": "ξ"}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--results_file",default="outputs/results/sensitivity_cora.json"); parser.add_argument("--output_dir",default="outputs/figures"); args=parser.parse_args()
    if not os.path.exists(args.results_file): print(f"文件不存在：{args.results_file}"); return
    import matplotlib.pyplot as plt
    with open(args.results_file) as f: data=json.load(f)
    fig, axes = plt.subplots(1,3,figsize=(15,4))
    for ax, p in zip(axes,["tau_base","k","xi_budget"]):
        if p not in data: continue
        vals=sorted(data[p].keys(),key=float); means=[data[p][v]["mean"]*100 for v in vals]
        ax.plot(range(len(vals)),means,marker="o"); ax.set_xticks(range(len(vals))); ax.set_xticklabels([str(v) for v in vals],fontsize=8)
        ax.set_title(f"{PARAM_LABELS.get(p, p)}（Cora）"); ax.set_ylabel("Test Acc (%)")
    plt.tight_layout(); os.makedirs(args.output_dir,exist_ok=True)
    out=os.path.join(args.output_dir,"sensitivity_cora.pdf"); plt.savefig(out,dpi=300,bbox_inches="tight"); plt.close()
    print(f"已保存：{out}")
if __name__ == "__main__": main()
