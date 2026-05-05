"""逐模块消融（论文表6.4，B0到B5逐步累加）。"""
import argparse
from datetime import datetime
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SEEDS = [42, 123, 456, 789, 2021, 2022, 2023, 2024, 2025, 2026]
CFG = {
    "sdgnn_pure": "configs/ablations/sdgnn_pure.yaml",
    "b0": "configs/ablations/b0_sdgnn.yaml",
    "b1": "configs/ablations/b1_plus_spectral.yaml",
    "b2": "configs/ablations/b2_plus_centrality.yaml",
    "b3": "configs/ablations/b3_plus_kcore.yaml",
    "b4": "configs/ablations/b4_plus_entropy.yaml",
    "b5": "configs/ablations/b5_full_main.yaml",
    "b2_rnd": "configs/ablations/b2_rnd_control.yaml",
    "b5_frozen": "configs/ablations/b5_frozen.yaml",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["cora"])
    parser.add_argument("--ablations", nargs="+", default=list(CFG.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--log_dir", default="outputs/results")
    parser.add_argument("--allow_partial", action="store_true")
    parser.add_argument("--noise_mode", choices=["add", "delete", "flip"], default="add")
    parser.add_argument("--noise_ratio", type=float, default=0.3)
    parser.add_argument("--infer_warmup", type=int, default=1)
    parser.add_argument("--infer_repeat", type=int, default=5)
    parser.add_argument("--max_outer_iter", type=int, default=None)
    args = parser.parse_args()

    from msas_gnn.config import load_experiment_config
    from msas_gnn.evaluation.ablation_runner import run_single_ablation

    for dataset in args.datasets:
        summaries = {}
        failures = []
        for ablation_id in args.ablations:
            cfg = load_experiment_config(
                dataset,
                ablation_id=ablation_id,
                override_path=CFG.get(ablation_id),
                overrides={
                    "seeds": args.seeds,
                    "allow_partial_results": args.allow_partial,
                    "measurement": {
                        "measure_inference": True,
                        "infer_warmup": args.infer_warmup,
                        "infer_repeat": args.infer_repeat,
                        "infer_device": "cpu",
                    },
                    "noise_eval": {
                        "enabled": True,
                        "mode": args.noise_mode,
                        "ratio": args.noise_ratio,
                    },
                    "alternating_opt": (
                        {"max_outer_iter": args.max_outer_iter}
                        if args.max_outer_iter is not None
                        else {}
                    ),
                },
            )
            try:
                summary = run_single_ablation(cfg)
                summaries[ablation_id] = summary
                os.makedirs(args.log_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_name = f"ablation_{ablation_id}_{dataset}_{timestamp}.json"
                with open(os.path.join(args.log_dir, out_name), "w", encoding="utf-8") as handle:
                    json.dump(summary, handle, indent=2, default=str)
            except Exception as exc:
                failures.append({"ablation_id": ablation_id, "error": str(exc)})
                logger.error("[%s/%s] 失败：%s", ablation_id, dataset, exc)

        if failures and not args.allow_partial:
            raise RuntimeError(f"{dataset} 消融实验不完整：{failures}")

        logger.info("\n消融汇总（%s）：", dataset)
        for ablation_id, summary in summaries.items():
            logger.info(
                "  %-15s clean=%.1f±%.1f%% noise=%s sparsity=%s infer=%sms",
                ablation_id,
                summary["mean_acc"] * 100,
                summary["std_acc"] * 100,
                f"{summary['noise_mean_acc'] * 100:.1f}%" if summary.get("noise_mean_acc") is not None else "--",
                f"{summary['mean_sparsity'] * 100:.1f}%" if summary.get("mean_sparsity") is not None else "--",
                f"{summary['mean_inference_ms']:.2f}" if summary.get("mean_inference_ms") is not None else "--",
            )


if __name__ == "__main__":
    main()
