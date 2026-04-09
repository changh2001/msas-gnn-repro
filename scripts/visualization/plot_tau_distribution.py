"""τ(i)分布图。"""
import argparse, subprocess, sys, os
def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--dataset",default="cora"); parser.add_argument("--output_dir",default="outputs/figures"); args=parser.parse_args()
    subprocess.run(
        [sys.executable,"scripts/experiments/run_tau_distribution.py","--dataset",args.dataset,"--output_dir",args.output_dir],
        check=True,
    )
if __name__ == "__main__": main()
