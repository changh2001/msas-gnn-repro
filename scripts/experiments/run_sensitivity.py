"""超参敏感性分析（图6-1，beta_tau / tau_base；k/gamma 保留为补充入口）。"""
import argparse, json, logging, os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
RANGES = {
    "beta_tau": [0.6, 0.8, 1.0, 1.2, 1.4],
    "tau_base": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
    "k": [20, 30, 50, 70, 100],
    "gamma": [0.2, 0.35, 0.5, 0.65, 0.8],
}
PARAM_BASE_ABLATION = {}
SEEDS=[42,123,456,789,2021,2022,2023,2024,2025,2026]


def _set_gamma_weights(cfg, gamma):
    node_dim = cfg.setdefault("node_dim", {})
    gamma = float(gamma)
    remain = max(1.0 - gamma, 0.0)
    node_dim["gamma"] = gamma
    node_dim["delta"] = remain * 3.0 / 5.0
    node_dim["omega_h"] = remain * 2.0 / 5.0

def run_sensitivity_param(param, ds, base_cfg, seeds):
    from msas_gnn.training.msas_trainer import MSASTrainer; import copy
    results={}
    for val in RANGES.get(param,[]):
        cfg=copy.deepcopy(base_cfg)
        if param=="beta_tau": cfg.setdefault("node_dim",{})["beta_tau"]=val
        elif param=="tau_base": cfg.setdefault("node_dim",{})["tau_base"]=val
        elif param=="k": cfg.setdefault("lars",{})["k"]=int(val)
        elif param=="gamma":
            _set_gamma_weights(cfg, val)
        accs=[]; failures=[]
        for seed in seeds:
            try:
                accs.append(MSASTrainer(cfg).run_single_seed(seed)["val_acc"])
            except Exception as e:
                failures.append({"seed": seed, "error": str(e)})
                logger.warning(f"  [{param}={val}] {e}")
        if failures:
            raise RuntimeError(f"{ds}/{param}={val} 缺少完整种子结果：{failures}")
        if accs:
            results[val]={"mean":float(np.mean(accs)),"std":float(np.std(accs))}
            logger.info(f"  {param}={val}: {np.mean(accs)*100:.1f}%")
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",default="cora"); parser.add_argument("--params",nargs="+",default=["beta_tau","tau_base"])
    parser.add_argument("--log_dir",default="outputs/results")
    args = parser.parse_args()
    from msas_gnn.config import load_experiment_config
    base = load_experiment_config(args.dataset, ablation_id="b5")
    all_r={}
    for p in args.params:
        ablation_id = PARAM_BASE_ABLATION.get(p, "b5")
        logger.info(f"敏感性：{p} (ablation={ablation_id})")
        base = load_experiment_config(args.dataset, ablation_id=ablation_id)
        all_r[p]=run_sensitivity_param(p,args.dataset,base,SEEDS)
    os.makedirs(args.log_dir,exist_ok=True)
    with open(os.path.join(args.log_dir,f"sensitivity_{args.dataset}.json"),"w") as f:
        json.dump(all_r,f,indent=2)
if __name__ == "__main__": main()
