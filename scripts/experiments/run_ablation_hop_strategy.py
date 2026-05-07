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
    "near_priority_engineering": "configs/ablations/hop_near_engineering.yaml",
    "spectral_gap_reference": "configs/ablations/hop_spectral_gap_reference.yaml",
    "near_priority_reference": "configs/ablations/hop_spectral_gap_reference.yaml",
    "reverse": "configs/ablations/hop_reverse.yaml",
}
ABLATION_BY_STRATEGY = {
    "uniform": "hop_uniform",
    "near_engineering": "hop_near_engineering",
    "near_priority_engineering": "hop_near_engineering",
    "spectral_gap_reference": "hop_spectral_gap_reference",
    "near_priority_reference": "hop_spectral_gap_reference",
    "reverse": "hop_reverse",
}
STRATEGY_ALIASES = {
    "near_priority_engineering": "near_engineering",
    "near_priority_reference": "spectral_gap_reference",
}
SEEDS = [42, 123, 456, 789, 2021, 2022, 2023, 2024, 2025, 2026]


def _canonical_strategy(strategy):
    return STRATEGY_ALIASES.get(str(strategy), str(strategy))


def _summary(results):
    accs = [row["test_acc"] for row in results]
    eps = [row["epsilon_approx"] for row in results if row.get("epsilon_approx") is not None]
    sigmas = [row["sigma_proxy"] for row in results if row.get("sigma_proxy") is not None]
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
        "mean_sigma_proxy": float(np.mean(sigmas)) if sigmas else None,
        "std_sigma_proxy": float(np.std(sigmas)) if sigmas else None,
        "sigma_proxy": float(np.mean(sigmas)) if sigmas else None,
        "mean_alternating_opt_seconds": float(np.mean(compute)) if compute else None,
        "std_alternating_opt_seconds": float(np.std(compute)) if compute else None,
        "per_seed": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["cora", "chameleon"])
    parser.add_argument("--strategies", nargs="+", default=["uniform", "near_engineering", "spectral_gap_reference", "reverse"])
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--allow_partial", action="store_true")
    parser.add_argument("--log_dir", default="outputs/results")
    parser.add_argument("--max_outer_iter", type=int, default=None)
    parser.add_argument("--compute_sigma_proxy", dest="compute_sigma_proxy", action="store_true", default=True)
    parser.add_argument("--no_compute_sigma_proxy", dest="compute_sigma_proxy", action="store_false")
    args = parser.parse_args()

    from msas_gnn.config import load_experiment_config
    from msas_gnn.evaluation.spectral_similarity import compute_sigma_proxy
    from msas_gnn.training.msas_trainer import MSASTrainer

    strategies = []
    for strategy in args.strategies:
        canonical = _canonical_strategy(strategy)
        if canonical not in strategies:
            strategies.append(canonical)

    dataset_summaries = {}
    for dataset in args.datasets:
        dataset_summaries[dataset] = {}
        for strategy in strategies:
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
                    need_sigma = bool(args.compute_sigma_proxy and dataset == "cora")
                    result = trainer.run_single_seed(seed, return_artifacts=need_sigma)
                    if need_sigma:
                        sigma_stats = compute_sigma_proxy(
                            result["data"],
                            result["theta_fixed"],
                            k_lanczos=50,
                            seed=seed,
                        )
                        result["sigma_proxy"] = float(sigma_stats["sigma_proxy"])
                        result["sigma_proxy_stats"] = sigma_stats
                        result.pop("theta_fixed", None)
                        result.pop("phi_tilde", None)
                        result.pop("data", None)
                    else:
                        result["sigma_proxy"] = None
                    result["hop_budget_strategy"] = strategy
                    result["e_approx"] = result.get("epsilon_approx")
                    result["pruning_rate"] = result.get("sparsity")
                    result["candidate_pruning_rate"] = result.get("sparsity")
                    results.append(result)
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
                "hop_budget_strategy": strategy,
                "solver_mode": str(cfg.get("lars", {}).get("theta_solver_mode", cfg.get("lars", {}).get("scheme", "residual_cascade"))),
                "theta_solver_mode": str(cfg.get("lars", {}).get("theta_solver_mode", cfg.get("lars", {}).get("scheme", "residual_cascade"))),
                "failed_seeds": failures,
            }
            logger.info(
                "  [%s] %s: %.1f±%.1f%% ε=%s σ̃=%s alt=%.2fs",
                dataset,
                strategy,
                dataset_summaries[dataset][strategy]["mean_acc"] * 100,
                dataset_summaries[dataset][strategy]["std_acc"] * 100,
                f"{dataset_summaries[dataset][strategy]['mean_epsilon_approx']:.3f}" if dataset_summaries[dataset][strategy]["mean_epsilon_approx"] is not None else "--",
                f"{dataset_summaries[dataset][strategy]['mean_sigma_proxy']:.3f}" if dataset_summaries[dataset][strategy]["mean_sigma_proxy"] is not None else "--",
                dataset_summaries[dataset][strategy]["mean_alternating_opt_seconds"] or float("nan"),
            )

    uniform_costs = [
        dataset_summaries[dataset]["uniform"]["mean_alternating_opt_seconds"]
        for dataset in args.datasets
        if dataset_summaries.get(dataset, {}).get("uniform", {}).get("mean_alternating_opt_seconds") is not None
    ]
    uniform_mean_cost = float(np.mean(uniform_costs)) if uniform_costs else None

    table_rows = {}
    for strategy in strategies:
        row = {"strategy": strategy, "hop_budget_strategy": strategy, "k": args.k}
        costs = []
        for dataset in args.datasets:
            summary = dataset_summaries.get(dataset, {}).get(strategy)
            if not summary:
                continue
            row.setdefault("solver_mode", summary.get("solver_mode"))
            row.setdefault("theta_solver_mode", summary.get("theta_solver_mode"))
            row[f"{dataset}_mean_acc"] = summary["mean_acc"]
            row[f"{dataset}_std_acc"] = summary["std_acc"]
            if dataset == "cora":
                row["mean_epsilon_approx"] = summary["mean_epsilon_approx"]
                row["std_epsilon_approx"] = summary["std_epsilon_approx"]
                row["e_approx"] = summary["mean_epsilon_approx"]
                row["sigma_proxy"] = summary["mean_sigma_proxy"]
                row["sigma_proxy_mean"] = summary["mean_sigma_proxy"]
                row["sigma_proxy_std"] = summary["std_sigma_proxy"]
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
