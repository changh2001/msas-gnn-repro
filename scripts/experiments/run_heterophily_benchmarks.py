"""主实验：异配性图（论文表6.3）。"""
import argparse, logging, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))
logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
SEEDS=[42,123,456,789,2021,2022,2023,2024,2025,2026]
DEFAULT_METHODS = ["gcn", "sgc", "pprgo", "geom_gcn", "h2gcn", "glnn", "b0", "b5"]
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["chameleon","squirrel"])
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--config", default="configs/models/msas_gnn_b5.yaml")
    parser.add_argument("--log_dir", default="outputs/results")
    parser.add_argument("--allow_partial", action="store_true")
    args = parser.parse_args()
    logger.info("注意：Chameleon/Squirrel使用60/20/20重划分，不可与官方划分文献直接比较")
    from scripts.experiments.run_main_benchmarks import run_dataset
    for ds in args.datasets:
        run_dataset(
            ds,
            args.seeds,
            args.config,
            args.log_dir,
            methods=args.methods,
            allow_partial=args.allow_partial,
        )
if __name__ == "__main__": main()
