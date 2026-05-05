"""补充实验：近邻预算几何衰减公比 varrho 扫描。"""
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

VARRHO_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SEEDS = [42, 123, 456, 789, 2021, 2022, 2023, 2024, 2025, 2026]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["cora", "chameleon"])
    parser.add_argument("--varrho_values", nargs="+", type=float, default=VARRHO_VALUES)
    parser.add_argument("--retention_values", dest="varrho_values", nargs="+", type=float)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--log_dir", default="outputs/results")
    args = parser.parse_args()

    from msas_gnn.config import load_experiment_config
    from msas_gnn.training.msas_trainer import MSASTrainer

    all_results = {}
    for dataset in args.datasets:
        base = load_experiment_config(dataset, ablation_id="b5")
        dataset_results = {}
        for varrho in args.varrho_values:
            cfg = copy.deepcopy(base)
            cfg.setdefault("hop_dim", {}).update({"varrho": varrho, "strategy": "near_engineering"})
            accs = []
            failures = []
            for seed in args.seeds:
                try:
                    accs.append(MSASTrainer(cfg).run_single_seed(seed)["test_acc"])
                except Exception as exc:
                    failures.append({"seed": seed, "error": str(exc)})
                    logger.warning("  [varrho=%.2f] %s", varrho, exc)
            if failures:
                raise RuntimeError(f"{dataset}/varrho={varrho} 缺少完整种子结果：{failures}")
            if accs:
                dataset_results[str(varrho)] = {
                    "mean": float(np.mean(accs)),
                    "std": float(np.std(accs)),
                    "strategy": "near_engineering",
                }
                logger.info("  [%s] varrho=%.2f: %.1f%%", dataset, varrho, np.mean(accs) * 100)
        all_results[dataset] = dataset_results

    os.makedirs(args.log_dir, exist_ok=True)
    output_path = os.path.join(args.log_dir, "supplemental_varrho_sweep.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    logger.info("已写入：%s", output_path)


if __name__ == "__main__":
    main()
