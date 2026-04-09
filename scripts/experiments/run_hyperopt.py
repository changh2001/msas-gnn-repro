"""Optuna超参搜索（小数据集n_trials=50，ogbn-arxiv默认跳过）。"""
import argparse, json, logging, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))
logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets",nargs="+",default=["cora","citeseer"])
    parser.add_argument("--n_trials",type=int,default=50); parser.add_argument("--skip",nargs="+",default=["ogbn_arxiv"])
    parser.add_argument("--log_dir",default="outputs/results/hyperopt")
    args = parser.parse_args()
    try: import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError: logger.error("未安装optuna"); sys.exit(1)
    import copy
    from msas_gnn.config import load_experiment_config
    from msas_gnn.training.msas_trainer import MSASTrainer
    for ds in args.datasets:
        if ds in args.skip: logger.info(f"[{ds}] 跳过（使用预设值）"); continue
        base = load_experiment_config(ds, ablation_id="b5")
        def obj(t):
            cfg=copy.deepcopy(base)
            cfg.setdefault("node_dim",{})["tau_base"]=t.suggest_float("tau_base",1e-5,1e-1,log=True)
            cfg.setdefault("lars",{})["k"]=t.suggest_int("k",20,100)
            cfg.setdefault("train",{})["lr"]=t.suggest_float("lr",1e-4,1e-1,log=True)
            cfg.setdefault("train",{})["dropout"]=t.suggest_float("dropout",0.0,0.7)
            return MSASTrainer(cfg).run_single_seed(42)["val_acc"]
        study=optuna.create_study(direction="maximize"); study.optimize(obj,n_trials=args.n_trials)
        logger.info(f"[{ds}] 最优：{study.best_params}")
        os.makedirs(args.log_dir,exist_ok=True)
        with open(os.path.join(args.log_dir,f"hyperopt_{ds}.json"),"w") as f:
            json.dump({"dataset":ds,"best_params":study.best_params,"best_value":study.best_value},f,indent=2)
if __name__ == "__main__": main()
