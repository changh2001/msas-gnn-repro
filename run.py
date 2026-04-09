"""统一任务调度入口。"""
import argparse, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from msas_gnn.api import run_task

def build_parser():
    p = argparse.ArgumentParser(description="MSAS-GNN 统一运行入口")
    p.add_argument("--task", choices=["smoke","train","ablation","efficiency","visualize"], required=True)
    p.add_argument("--dataset", choices=["cora","citeseer","pubmed","ogbn_arxiv","chameleon","squirrel"], default="cora")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ablation", choices=["b0","sdgnn_pure","b1","b2","b3","b4","b5","b5_frozen","b2_rnd"], default="b5")
    p.add_argument("--config", default=None)
    p.add_argument("--vis_type", choices=["tsne","tau_dist","sensitivity"], default="tsne")
    p.add_argument("--methods", nargs="+", default=None)
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--no_cache", action="store_true")
    return p

if __name__ == "__main__":
    run_task(build_parser().parse_args())
