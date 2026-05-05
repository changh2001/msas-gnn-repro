"""t-SNE节点表示可视化（论文图6.x）。

默认对三种表示做对照：
- GCN 教师嵌入 H*
- SDGNN(B0) 的稀疏近似表示 Ĥ
- MSAS-GNN(B5-full) 的稀疏近似表示 Ĥ
"""
import argparse, logging, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cora")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tsne_seed", type=int, default=42)
    parser.add_argument("--perplexity", type=float, default=30)
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--output_dir", default="outputs/figures")
    args = parser.parse_args()
    from msas_gnn.config import load_experiment_config
    from msas_gnn.evaluation.visualizer import plot_tsne

    cfg = load_experiment_config(
        args.dataset,
        ablation_id="b5",
        overrides={
            "seed": args.seed,
            "tsne_seed": args.tsne_seed,
            "perplexity": args.perplexity,
            "max_iter": args.max_iter,
        },
    )
    out = plot_tsne(cfg, output_dir=args.output_dir)
    logger.info("t-SNE图已保存：%s", out)

if __name__ == "__main__":
    main()
