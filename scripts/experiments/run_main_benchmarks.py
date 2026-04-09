"""主实验准确率：主线方法与对照基线统一调度。"""
import argparse, json, logging, os, sys, time, numpy as np
from datetime import datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))
logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
SEEDS = [42,123,456,789,2021,2022,2023,2024,2025,2026]
DEFAULT_METHODS = ["gcn", "sgc", "pprgo", "glnn", "b0", "b5"]
METHOD_ALIASES = {
    "sdgnn": "b0",
    "msas_gnn": "b5",
    "msas_gnn_b5": "b5",
}


def normalize_method_id(method_id):
    method_id = str(method_id).lower()
    return METHOD_ALIASES.get(method_id, method_id)


def _build_runner(ds, method_id, cfg_path):
    from msas_gnn.config import load_experiment_config, load_yaml
    from msas_gnn.training.baseline_trainer import BaselineTrainer
    from msas_gnn.training.msas_trainer import MSASTrainer

    method_id = normalize_method_id(method_id)
    if method_id in {"b0", "b5"}:
        override_path = cfg_path if method_id == "b5" else None
        cfg = load_experiment_config(ds, ablation_id=method_id, override_path=override_path)
        return method_id, MSASTrainer(cfg)

    baseline_cfg = load_yaml(f"configs/teachers/{method_id}.yaml")
    base_cfg = load_experiment_config(ds, ablation_id="b5")
    return method_id, BaselineTrainer(base_cfg, method_id, baseline_cfg=baseline_cfg)


def run_method(ds, method_id, seeds, cfg_path, log_dir, allow_partial=False):
    method_id, trainer = _build_runner(ds, method_id, cfg_path)
    results = []
    failures = []
    for seed in seeds:
        try:
            r = trainer.run_single_seed(seed)
            results.append(r)
            logger.info("[%s/%s][seed=%s] acc=%.4f", ds, method_id, seed, r["test_acc"])
        except Exception as e:
            failures.append({"seed": seed, "error": str(e)})
            logger.error("[%s/%s][seed=%s] 失败：%s", ds, method_id, seed, e)
    if failures and not allow_partial:
        failed = ", ".join(str(item["seed"]) for item in failures)
        raise RuntimeError(f"{ds}/{method_id} 主实验缺少完整10种子结果，失败seed: {failed}")
    if results:
        accs = [r["test_acc"] for r in results]
        logger.info("[%s/%s] %.1f±%.1f%%", ds, method_id, np.mean(accs) * 100, np.std(accs) * 100)
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        payload = {
            "dataset": ds,
            "method_id": method_id,
            "mean_acc": float(np.mean(accs)),
            "std_acc": float(np.std(accs)),
            "per_seed": results,
            "failed_seeds": failures,
        }
        if method_id in {"b0", "b5"}:
            payload["ablation_id"] = method_id
            if results and results[0].get("protocols") is not None:
                payload["protocols"] = results[0]["protocols"]
        with open(os.path.join(log_dir, f"main_{method_id}_{ds}_{ts}.json"), "w") as f:
            json.dump(payload, f, indent=2, default=str)
    if failures and allow_partial:
        logger.warning("[%s/%s] 仅基于部分种子汇总：成功=%s 失败=%s", ds, method_id, len(results), len(failures))
    return results


def run_dataset(ds, seeds, cfg_path, log_dir, methods=None, allow_partial=False):
    summaries = {}
    for method_id in methods or DEFAULT_METHODS:
        summaries[normalize_method_id(method_id)] = run_method(
            ds,
            method_id,
            seeds,
            cfg_path,
            log_dir,
            allow_partial=allow_partial,
        )
    return summaries

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["cora","citeseer","pubmed","ogbn_arxiv"])
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--config", default="configs/models/msas_gnn_b5.yaml")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--log_dir", default="outputs/results")
    parser.add_argument("--allow_partial", action="store_true")
    args = parser.parse_args()
    t0 = time.time()
    for ds in args.datasets:
        run_dataset(
            ds,
            args.seeds,
            args.config,
            args.log_dir,
            methods=args.methods,
            allow_partial=args.allow_partial,
        )
    logger.info(f"完成，总耗时 {time.time()-t0:.0f}s")
if __name__ == "__main__": main()
