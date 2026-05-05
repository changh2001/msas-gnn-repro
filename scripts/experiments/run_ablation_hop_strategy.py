"""跳距策略消融（论文表6.5）。"""
import argparse
from datetime import datetime
import json
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

STRAT = {
    "uniform": "configs/ablations/hop_uniform.yaml",
    "near_engineering": "configs/ablations/hop_near_engineering.yaml",
    "spectral_gap_reference": "configs/ablations/hop_spectral_gap_reference.yaml",
    "reverse": "configs/ablations/hop_reverse.yaml",
}
ABLATION_BY_STRATEGY = {
    "uniform": "hop_uniform",
    "near_engineering": "hop_near_engineering",
    "spectral_gap_reference": "hop_spectral_gap_reference",
    "reverse": "hop_reverse",
}
SEEDS = [42, 123, 456, 789, 2021, 2022, 2023, 2024, 2025, 2026]


def _summary(results):
    accs = [row["test_acc"] for row in results]
    eps = [row["epsilon_approx"] for row in results if row.get("epsilon_approx") is not None]
    compute = [
        row.get("stage_times", {}).get("alternating_opt")
        for row in results
        if row.get("stage_times", {}).get("alternating_opt") is not None
    ]
    return {
        "mean_acc": float(np.mean(accs)),
        "std_acc": float(np.std(accs)),
        "mean_epsilon_approx": float(np.mean(eps)) if eps else None,
        "std_epsilon_approx": float(np.std(eps)) if eps else None,
        "mean_alternating_opt_seconds": float(np.mean(compute)) if compute else None,
        "std_alternating_opt_seconds": float(np.std(compute)) if compute else None,
        "per_seed": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["cora", "chameleon"])
    parser.add_argument("--strategies", nargs="+", default=list(STRAT.keys()))
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--allow_partial", action="store_true")
    parser.add_argument("--log_dir", default="outputs/results")
    parser.add_argument("--max_outer_iter", type=int, default=None)
    args = parser.parse_args()

    from msas_gnn.config import load_experiment_config
    from msas_gnn.training.msas_trainer import MSASTrainer

    dataset_summaries = {}
    for dataset in args.datasets:
        dataset_summaries[dataset] = {}
        for strategy in args.strategies:
            cfg = load_experiment_config(
                dataset,
                ablation_id=ABLATION_BY_STRATEGY[strategy],
                override_path=STRAT[strategy],
                overrides={
                    "alternating_opt": (
                        {"max_outer_iter": args.max_outer_iter}
                        if args.max_outer_iter is not None
                        else {}
                    )
                },
            )
            cfg.setdefault("lars", {})["k"] = args.k
            trainer = MSASTrainer(cfg)
            results = []
            failures = []
            for seed in args.seeds:
                try:
                    results.append(trainer.run_single_seed(seed))
                except Exception as exc:
                    failures.append({"seed": seed, "error": str(exc)})
                    logger.warning("  [%s/%s/%s] %s", dataset, strategy, seed, exc)
            if failures and not args.allow_partial:
                failed = ", ".join(str(item["seed"]) for item in failures)
                raise RuntimeError(f"{dataset}/{strategy} 缺少完整10种子结果，失败seed: {failed}")
            if not results:
                continue
            dataset_summaries[dataset][strategy] = {
                **_summary(results),
                "dataset": dataset,
                "strategy": strategy,
                "failed_seeds": failures,
            }
            logger.info(
                "  [%s] %s: %.1f±%.1f%% ε=%s alt=%.2fs",
                dataset,
                strategy,
                dataset_summaries[dataset][strategy]["mean_acc"] * 100,
                dataset_summaries[dataset][strategy]["std_acc"] * 100,
                f"{dataset_summaries[dataset][strategy]['mean_epsilon_approx']:.3f}" if dataset_summaries[dataset][strategy]["mean_epsilon_approx"] is not None else "--",
                dataset_summaries[dataset][strategy]["mean_alternating_opt_seconds"] or float("nan"),
            )

    uniform_costs = [
        dataset_summaries[dataset]["uniform"]["mean_alternating_opt_seconds"]
        for dataset in args.datasets
        if dataset_summaries.get(dataset, {}).get("uniform", {}).get("mean_alternating_opt_seconds") is not None
    ]
    uniform_mean_cost = float(np.mean(uniform_costs)) if uniform_costs else None

    table_rows = {}
    for strategy in args.strategies:
        row = {"strategy": strategy, "k": args.k}
        costs = []
        for dataset in args.datasets:
            summary = dataset_summaries.get(dataset, {}).get(strategy)
            if not summary:
                continue
            row[f"{dataset}_mean_acc"] = summary["mean_acc"]
            row[f"{dataset}_std_acc"] = summary["std_acc"]
            if dataset == "cora":
                row["mean_epsilon_approx"] = summary["mean_epsilon_approx"]
                row["std_epsilon_approx"] = summary["std_epsilon_approx"]
            if summary["mean_alternating_opt_seconds"] is not None:
                costs.append(summary["mean_alternating_opt_seconds"])
        mean_cost = float(np.mean(costs)) if costs else None
        row["mean_alternating_opt_seconds"] = mean_cost
        row["relative_compute_overhead"] = (
            float(mean_cost / uniform_mean_cost - 1.0)
            if uniform_mean_cost is not None and mean_cost is not None
            else None
        )
        table_rows[strategy] = row

    os.makedirs(args.log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(args.log_dir, f"hop_strategy_{timestamp}.json"), "w", encoding="utf-8") as handle:
        json.dump({"per_dataset": dataset_summaries, "table_rows": table_rows}, handle, indent=2)


if __name__ == "__main__":
    main()
