"""补充实验：σ̃ 代理量验证。"""
import argparse
import copy
import json
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../src"))
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SEEDS = [42, 123, 456, 789, 2021, 2022, 2023, 2024, 2025, 2026]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["citeseer", "ogbn_arxiv"])
    parser.add_argument("--k_values", nargs="+", type=int, default=[20, 30, 50, 80, 100])
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--log_dir", default="outputs/results")
    parser.add_argument("--num_eigs", type=int, default=50)
    parser.add_argument("--poly_coeff_sum", type=float, default=1.0)
    args = parser.parse_args()
    from msas_gnn.config import load_experiment_config
    from msas_gnn.evaluation.spectral_similarity import (
        compute_engineering_reference,
        compute_sigma_proxy,
    )
    from msas_gnn.training.msas_trainer import MSASTrainer
    os.makedirs(args.log_dir, exist_ok=True)
    all_results = {}
    for ds in args.datasets:
        base = load_experiment_config(ds, ablation_id="b5")
        results = []
        for k in args.k_values:
            per_seed = []
            failures = []
            for seed in args.seeds:
                cfg = copy.deepcopy(base)
                cfg.setdefault("lars", {})["k"] = k
                try:
                    run = MSASTrainer(cfg).run_single_seed(seed, return_artifacts=True)
                    sigma_stats = compute_sigma_proxy(run["data"], run["theta_fixed"], num_eigs=args.num_eigs)
                    engineering_ref = compute_engineering_reference(
                        sigma_stats["sigma_proxy"],
                        run["data"].x,
                        poly_coeff_sum=args.poly_coeff_sum,
                    )
                    per_seed.append(
                        {
                            "seed": seed,
                            "acc": float(run["test_acc"]),
                            "epsilon_approx": float(run["epsilon_approx"]),
                            "sigma_proxy": float(sigma_stats["sigma_proxy"]),
                            "engineering_ref": float(engineering_ref),
                            "rayleigh_min": float(sigma_stats["rayleigh_min"]),
                            "rayleigh_max": float(sigma_stats["rayleigh_max"]),
                            "num_modes": int(sigma_stats["num_modes"]),
                        }
                    )
                except Exception as exc:
                    failures.append({"seed": seed, "error": str(exc)})
                    logger.warning("  [%s/k=%s/seed=%s] %s", ds, k, seed, exc)
            if failures:
                raise RuntimeError(f"{ds}/k={k} 缺少完整种子结果：{failures}")
            row = {
                "k": k,
                "acc": float(np.mean([item["acc"] for item in per_seed])),
                "acc_mean": float(np.mean([item["acc"] for item in per_seed])),
                "acc_std": float(np.std([item["acc"] for item in per_seed])),
                "epsilon_approx": float(np.mean([item["epsilon_approx"] for item in per_seed])),
                "epsilon_approx_mean": float(np.mean([item["epsilon_approx"] for item in per_seed])),
                "epsilon_approx_std": float(np.std([item["epsilon_approx"] for item in per_seed])),
                "sigma_proxy": float(np.mean([item["sigma_proxy"] for item in per_seed])),
                "sigma_proxy_mean": float(np.mean([item["sigma_proxy"] for item in per_seed])),
                "sigma_proxy_std": float(np.std([item["sigma_proxy"] for item in per_seed])),
                "engineering_ref": float(np.mean([item["engineering_ref"] for item in per_seed])),
                "engineering_ref_mean": float(np.mean([item["engineering_ref"] for item in per_seed])),
                "engineering_ref_std": float(np.std([item["engineering_ref"] for item in per_seed])),
                "rayleigh_min": float(np.mean([item["rayleigh_min"] for item in per_seed])),
                "rayleigh_max": float(np.mean([item["rayleigh_max"] for item in per_seed])),
                "num_modes": int(round(np.mean([item["num_modes"] for item in per_seed]))),
                "per_seed": per_seed,
            }
            results.append(row)
            logger.info(
                "  [%s] k=%s: acc=%.1f±%.1f%% σ̃=%.3f ε=%.3f ref=%.3f",
                ds,
                k,
                row["acc_mean"] * 100.0,
                row["acc_std"] * 100.0,
                row["sigma_proxy"],
                row["epsilon_approx"],
                row["engineering_ref"],
            )
        all_results[ds] = results
        with open(os.path.join(args.log_dir, f"supplemental_sigma_proxy_{ds}.json"), "w") as f:
            json.dump(results, f, indent=2)
    with open(os.path.join(args.log_dir, "supplemental_sigma_proxy.json"), "w") as f:
        json.dump(all_results, f, indent=2)
if __name__ == "__main__": main()
