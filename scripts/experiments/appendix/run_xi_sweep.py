"""附录C.3：ξ细粒度扫描（10档）。"""
import argparse, json, logging, os, sys, numpy as np, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../src"))
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
XI=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]; SEEDS=[42,123,456,789,2021,2022,2023,2024,2025,2026]
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets",nargs="+",default=["cora","chameleon"])
    parser.add_argument("--xi_values",nargs="+",type=float,default=XI)
    parser.add_argument("--seeds",nargs="+",type=int,default=SEEDS)
    parser.add_argument("--log_dir",default="outputs/results")
    args = parser.parse_args()
    from msas_gnn.config import load_experiment_config
    from msas_gnn.training.msas_trainer import MSASTrainer; all_r={}
    for ds in args.datasets:
        base = load_experiment_config(ds, ablation_id="b5"); xi_r={}
        for xi in args.xi_values:
            cfg=copy.deepcopy(base); cfg.setdefault("hop_dim",{}).update({"xi_budget":xi,"strategy":"xi_budget"})
            accs=[]; failures=[]
            for seed in args.seeds:
                try:
                    accs.append(MSASTrainer(cfg).run_single_seed(seed)["test_acc"])
                except Exception as e:
                    failures.append({"seed": seed, "error": str(e)})
                    logger.warning(f"  [ξ={xi}] {e}")
            if failures:
                raise RuntimeError(f"{ds}/xi={xi} 缺少完整种子结果：{failures}")
            if accs: xi_r[xi]={"mean":float(np.mean(accs)),"std":float(np.std(accs))}; logger.info(f"  [{ds}] ξ={xi:.1f}: {np.mean(accs)*100:.1f}%")
        all_r[ds]=xi_r
    os.makedirs(args.log_dir,exist_ok=True)
    with open(os.path.join(args.log_dir,"appendix_xi_sweep.json"),"w") as f: json.dump(all_r,f,indent=2)
if __name__ == "__main__": main()
