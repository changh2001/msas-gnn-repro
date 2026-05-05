"""t-SNE三图横排。"""
import argparse, subprocess, sys, os, shutil
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset",default="cora")
    parser.add_argument("--output_dir",default="outputs/figures")
    parser.add_argument("--paper_output", default="thesis/figures/6-3.pdf")
    args=parser.parse_args()
    subprocess.run(
        [sys.executable,"scripts/experiments/run_tsne.py","--dataset",args.dataset,"--output_dir",args.output_dir],
        check=True,
    )
    out = os.path.join(args.output_dir, f"tsne_{args.dataset}.pdf")
    if args.paper_output and os.path.exists(out):
        os.makedirs(os.path.dirname(args.paper_output), exist_ok=True)
        shutil.copyfile(out, args.paper_output)
if __name__ == "__main__": main()
