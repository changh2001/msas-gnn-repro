"""补充实验 ξ 扫描图。"""
import argparse, json, os
def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--results_file",default="outputs/results/supplemental_xi_sweep.json"); parser.add_argument("--output_dir",default="outputs/figures/supplemental"); args=parser.parse_args()
    os.makedirs(args.output_dir,exist_ok=True)
    if not os.path.exists(args.results_file): print(f"文件不存在：{args.results_file}"); return
    import matplotlib.pyplot as plt
    with open(args.results_file) as f: data=json.load(f)
    fig, ax = plt.subplots(figsize=(7,4))
    for ds, xi_data in data.items():
        xis=sorted(xi_data.keys(),key=float); means=[xi_data[x]["mean"]*100 for x in xis]
        ax.plot([float(x) for x in xis],means,marker="o",label=ds)
    ax.set_xlabel("ξ"); ax.set_ylabel("Test Accuracy (%)"); ax.legend(); ax.set_title("补充实验：ξ细粒度扫描")
    out=os.path.join(args.output_dir,"xi_sweep.pdf"); plt.tight_layout(); plt.savefig(out,dpi=300,bbox_inches="tight"); plt.close()
    print(f"已保存：{out}")
if __name__ == "__main__": main()
