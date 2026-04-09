"""补充实验：Chameleon 与 ogbn-arxiv 的超参敏感性补充。"""
import sys, os, argparse, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../src"))
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets",nargs="+",default=["chameleon","ogbn_arxiv"])
    parser.add_argument("--params",nargs="+",default=["k","tau_base","gamma"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42,123,456,789,2021,2022,2023,2024,2025,2026])
    parser.add_argument("--log_dir", default="outputs/results")
    args = parser.parse_args()
    from msas_gnn.config import load_experiment_config
    from scripts.experiments.run_sensitivity import run_sensitivity_param
    os.makedirs(args.log_dir, exist_ok=True)
    for ds in args.datasets:
        dataset_results = {}
        for p in args.params:
            base = load_experiment_config(ds, ablation_id="b5")
            dataset_results[p] = run_sensitivity_param(p, ds, base, args.seeds)
        with open(os.path.join(args.log_dir, f"sensitivity_{ds}.json"), "w") as f:
            json.dump(dataset_results, f, indent=2)
if __name__ == "__main__": main()
