"""对外高层接口。"""
import argparse, logging, os
from typing import Dict, Any

logger = logging.getLogger(__name__)

def _load_config(args) -> Dict[str, Any]:
    from msas_gnn.config import load_experiment_config

    cfg = load_experiment_config(
        dataset=args.dataset,
        ablation_id=args.ablation,
        override_path=args.config,
        overrides={"seed": args.seed},
    )
    if getattr(args, "no_cache", False):
        cfg["no_cache"] = True
    return cfg

def run_task(args) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")
    import sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))
    cfg = _load_config(args)
    if args.task == "smoke": _smoke(cfg)
    elif args.task == "train": _train(cfg)
    elif args.task == "ablation":
        from msas_gnn.evaluation.ablation_runner import run_single_ablation
        run_single_ablation(cfg)
    elif args.task == "efficiency":
        from msas_gnn.evaluation.efficiency import run_efficiency_benchmark
        run_efficiency_benchmark(cfg, methods=getattr(args, "methods", None))
    elif args.task == "visualize":
        from msas_gnn.evaluation.visualizer import run_visualization
        run_visualization(cfg, getattr(args, "vis_type", "tsne"))

def _smoke(cfg):
    from msas_gnn.training.msas_trainer import MSASTrainer
    r = MSASTrainer(cfg).run_single_seed(cfg["seed"])
    logger.info(f"[smoke] acc={r['test_acc']:.4f} ε={r['epsilon_approx']:.4f}")
    assert r["test_acc"] > 0.78, f"冒烟测试失败：acc={r['test_acc']:.4f}"
    logger.info("[smoke] PASSED")

def _train(cfg):
    from msas_gnn.training.msas_trainer import MSASTrainer
    r = MSASTrainer(cfg).run_single_seed(cfg["seed"])
    logger.info(f"[train] acc={r['test_acc']:.4f} ε={r['epsilon_approx']:.4f} k̄={r['k_bar']:.1f}")
