"""消融实验批量运行器（正文 B0~B5 与对照变体统一调度）。对应论文§6.3。"""
from __future__ import annotations

from copy import deepcopy
import logging

import numpy as np

from msas_gnn.constants import DEFAULT_SEEDS
from msas_gnn.evaluation.protocols import build_protocol_metadata

logger = logging.getLogger(__name__)


def _summarize_numeric(results, key):
    values = [float(row[key]) for row in results if row.get(key) is not None]
    if not values:
        return None, None
    return float(np.mean(values)), float(np.std(values))


def _augment_with_inference_metrics(cfg, result):
    from msas_gnn.evaluation.efficiency import infer_latency_sparse_paper_protocol

    measurement_cfg = cfg.get("measurement", {})
    efficiency_cfg = cfg.get("efficiency", {})
    latency = infer_latency_sparse_paper_protocol(
        result["theta_fixed"],
        result["phi_tilde"],
        num_nodes=result["phi_tilde"].shape[0],
        batch_size=int(cfg.get("train", {}).get("batch_size", cfg.get("batch_size", 1024))),
        warmup=int(measurement_cfg.get("infer_warmup", efficiency_cfg.get("warmup_runs", 10))),
        repeat=int(measurement_cfg.get("infer_repeat", efficiency_cfg.get("repeat_runs", 100))),
        is_large_graph=cfg.get("dataset") == "ogbn_arxiv",
        device=str(measurement_cfg.get("infer_device", cfg.get("device", "cpu"))),
    )
    result["inference_latency"] = latency
    result["inference_ms"] = float(
        latency.get("median_ms")
        or latency.get("full_graph_ms")
        or latency.get("median_ms_per_batch")
        or 0.0
    )
    result["inference_time_ms"] = result["inference_ms"]
    return result


def summarize_seed_results(cfg, results, failures, extra=None):
    acc_mean, acc_std = _summarize_numeric(results, "test_acc")
    if acc_mean is None:
        raise RuntimeError(f"{cfg['ablation_id']}/{cfg['dataset']} 未产生任何有效结果")
    eps_mean, eps_std = _summarize_numeric(results, "epsilon_approx")
    sparsity_mean, sparsity_std = _summarize_numeric(results, "sparsity")
    infer_mean, infer_std = _summarize_numeric(results, "inference_ms")
    kbar_mean, kbar_std = _summarize_numeric(results, "k_bar")
    support_mean, support_std = _summarize_numeric(results, "support_total")
    candidate_mean, candidate_std = _summarize_numeric(results, "candidate_total")
    alt_mean, alt_std = _summarize_numeric(
        [
            {"alternating_opt_seconds": row.get("stage_times", {}).get("alternating_opt")}
            for row in results
        ],
        "alternating_opt_seconds",
    )
    summary = {
        "ablation_id": cfg["ablation_id"],
        "dataset": cfg["dataset"],
        "mean_acc": acc_mean,
        "std_acc": acc_std,
        "mean_epsilon_approx": eps_mean,
        "std_epsilon_approx": eps_std,
        "mean_e_approx": eps_mean,
        "std_e_approx": eps_std,
        "mean_sparsity": sparsity_mean,
        "std_sparsity": sparsity_std,
        "mean_pruning_rate": sparsity_mean,
        "std_pruning_rate": sparsity_std,
        "mean_candidate_pruning_rate": sparsity_mean,
        "std_candidate_pruning_rate": sparsity_std,
        "mean_inference_ms": infer_mean,
        "std_inference_ms": infer_std,
        "mean_inference_time_ms": infer_mean,
        "std_inference_time_ms": infer_std,
        "mean_k_bar": kbar_mean,
        "std_k_bar": kbar_std,
        "mean_support_total": support_mean,
        "std_support_total": support_std,
        "mean_candidate_total": candidate_mean,
        "std_candidate_total": candidate_std,
        "mean_alternating_opt_seconds": alt_mean,
        "std_alternating_opt_seconds": alt_std,
        "per_seed": results,
        "failed_seeds": failures,
        "protocols": build_protocol_metadata(cfg),
        "solver_mode": str(cfg.get("lars", {}).get("theta_solver_mode", cfg.get("lars", {}).get("scheme", "residual_cascade"))),
        "theta_solver_mode": str(cfg.get("lars", {}).get("theta_solver_mode", cfg.get("lars", {}).get("scheme", "residual_cascade"))),
        "hop_budget_strategy": str(cfg.get("hop_dim", {}).get("strategy", "spectral_gap_reference")),
    }
    if extra:
        summary.update(extra)
    return summary


def _run_noise_eval(cfg, seeds, allow_partial):
    from msas_gnn.training.msas_trainer import MSASTrainer

    noise_eval_cfg = cfg.get("noise_eval", {})
    if not noise_eval_cfg.get("enabled", False):
        return {}

    noisy_cfg = deepcopy(cfg)
    noisy_cfg["noise"] = {
        "enabled": True,
        "mode": noise_eval_cfg.get("mode", "add"),
        "ratio": float(noise_eval_cfg.get("ratio", 0.3)),
    }
    noisy_cfg.pop("noise_eval", None)
    noisy_cfg.setdefault("measurement", {})["measure_inference"] = False
    trainer = MSASTrainer(noisy_cfg)
    results = []
    failures = []
    for seed in seeds:
        try:
            results.append(trainer.run_single_seed(seed))
        except Exception as exc:
            failures.append({"seed": seed, "error": str(exc)})
            logger.error("  noise seed=%s 失败：%s", seed, exc)
    if failures and not allow_partial:
        failed = ", ".join(str(item["seed"]) for item in failures)
        raise RuntimeError(f"{cfg['ablation_id']}/{cfg['dataset']} 噪声评测缺少完整结果，失败seed: {failed}")
    noise_mean, noise_std = _summarize_numeric(results, "test_acc")
    return {
        "noise_mode": noisy_cfg["noise"]["mode"],
        "noise_ratio": noisy_cfg["noise"]["ratio"],
        "noise_mean_acc": noise_mean,
        "noise_std_acc": noise_std,
        "noise_failed_seeds": failures,
    }


def run_single_ablation(cfg):
    from msas_gnn.training.msas_trainer import MSASTrainer

    trainer = MSASTrainer(cfg)
    seeds = cfg.get("seeds", DEFAULT_SEEDS)
    results = []
    failures = []
    allow_partial = bool(cfg.get("allow_partial_results", False))
    measure_inference = bool(cfg.get("measurement", {}).get("measure_inference", False))
    for seed in seeds:
        try:
            result = trainer.run_single_seed(seed, return_artifacts=measure_inference)
            if measure_inference:
                result = _augment_with_inference_metrics(cfg, result)
                result.pop("theta_fixed", None)
                result.pop("phi_tilde", None)
                result.pop("data", None)
            results.append(result)
            logger.info("  seed=%s acc=%.4f", seed, result["test_acc"])
        except Exception as exc:
            failures.append({"seed": seed, "error": str(exc)})
            logger.error("  seed=%s 失败：%s", seed, exc)
    if failures and not allow_partial:
        failed = ", ".join(str(item["seed"]) for item in failures)
        raise RuntimeError(f"{cfg['ablation_id']}/{cfg['dataset']} 缺少完整10种子结果，失败seed: {failed}")
    if not results:
        raise RuntimeError(f"{cfg['ablation_id']}/{cfg['dataset']} 未产生任何有效结果")
    if failures:
        logger.warning("仅基于部分种子汇总：成功=%s 失败=%s", len(results), len(failures))
    extra = _run_noise_eval(cfg, seeds, allow_partial)
    summary = summarize_seed_results(cfg, results, failures, extra=extra)
    logger.info(
        "  汇总：%.1f±%.1f%% | ε=%s | 稀疏度=%s | 推理=%s ms",
        summary["mean_acc"] * 100,
        summary["std_acc"] * 100,
        f"{summary['mean_epsilon_approx']:.3f}" if summary["mean_epsilon_approx"] is not None else "--",
        f"{summary['mean_sparsity'] * 100:.1f}%" if summary["mean_sparsity"] is not None else "--",
        f"{summary['mean_inference_ms']:.2f}" if summary["mean_inference_ms"] is not None else "--",
    )
    return summary
