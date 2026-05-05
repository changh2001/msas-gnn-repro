"""tau(i)分布可视化（论文图6.x）。预期：r_Pearson(log tau, log 度) 约 -0.68。"""
import argparse, logging, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cora")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="outputs/figures")
    args = parser.parse_args()
    from msas_gnn.config import load_experiment_config
    from msas_gnn.evaluation.visualizer import plot_tau_distribution

    cfg = load_experiment_config(args.dataset, ablation_id="b5", overrides={"seed": args.seed})
    out = plot_tau_distribution(cfg, output_dir=args.output_dir)
    logger.info("已保存：%s", out)

if __name__ == "__main__":
    main()
