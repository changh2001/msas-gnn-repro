"""效率分析（论文表6.6，推理时延+显存+加速比）。须使用论文对齐口径。"""
import argparse, json, logging, os, sys, time
from datetime import datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))
logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
ALL_DS=["cora","citeseer","pubmed","ogbn_arxiv"]
METHODS=["gcn","sgc","pprgo","glnn","sdgnn","msas_gnn"]
LARGE={"ogbn_arxiv"}
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all",action="store_true"); parser.add_argument("--datasets",nargs="+",default=["cora","ogbn_arxiv"])
    parser.add_argument("--methods",nargs="+",default=METHODS)
    parser.add_argument("--warmup",type=int,default=10); parser.add_argument("--repeat",type=int,default=100)
    parser.add_argument("--batch_size",type=int,default=1024); parser.add_argument("--log_dir",default="outputs/results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    import torch; device="cuda" if torch.cuda.is_available() else "cpu"
    if device=="cpu": logger.warning("CPU模式，效率数字不对应论文表6.6")
    datasets = ALL_DS if args.all else args.datasets
    from msas_gnn.config import load_yaml
    from msas_gnn.data.dataset_factory import load_dataset
    from msas_gnn.data.transforms import add_self_loops_transform
    from msas_gnn.evaluation.efficiency import (
        build_baseline_model,
        benchmark_sparse_method,
        count_model_parameters,
        infer_latency_paper_protocol,
        measure_memory,
    )
    all_r={}
    for ds in datasets:
        is_large = ds in LARGE
        data, nc, nf = load_dataset(ds); data = add_self_loops_transform(data)
        ds_r={}
        for m in args.methods:
            try:
                if m in {"sdgnn", "b0"}:
                    cfg = {"dataset": ds}
                    ds_r[m] = benchmark_sparse_method(
                        cfg,
                        ablation_id="b0",
                        seed=args.seed,
                        batch_size=args.batch_size,
                        warmup=args.warmup,
                        repeat=args.repeat,
                        device=device,
                    )
                elif m in {"msas_gnn", "msas_gnn_b5", "b5"}:
                    cfg = {"dataset": ds}
                    ds_r[m] = benchmark_sparse_method(
                        cfg,
                        ablation_id="b5",
                        seed=args.seed,
                        batch_size=args.batch_size,
                        warmup=args.warmup,
                        repeat=args.repeat,
                        device=device,
                    )
                else:
                    baseline_cfg = load_yaml(f"configs/teachers/{m}.yaml")
                    model = build_baseline_model(m, nf, nc, baseline_cfg=baseline_cfg)
                    lat=infer_latency_paper_protocol(model,data,batch_size=args.batch_size,warmup=args.warmup,repeat=args.repeat,is_large_graph=is_large,device=device)
                    mem=measure_memory(model,data,device=device)
                    params = count_model_parameters(model)
                    ds_r[m]={**lat,**mem,**params}
                k="median_ms_per_batch" if is_large else "median_ms"
                logger.info(f"  {m}: {ds_r[m].get(k,'N/A'):.2f}ms {ds_r[m].get('peak_memory_mb','N/A'):.0f}MB")
            except Exception as e: logger.error(f"  {m} 失败：{e}")
        if "gcn" in ds_r:
            dense_key = "median_ms_per_batch" if is_large else "median_ms"
            gcn_ms = float(ds_r["gcn"][dense_key])
            for method_name, payload in ds_r.items():
                if dense_key in payload:
                    payload["speedup_vs_gcn"] = float(gcn_ms / (float(payload[dense_key]) + 1e-10))
        all_r[ds]=ds_r
    summary = {}
    for method_name in args.methods:
        speeds = []
        for ds in datasets:
            payload = all_r.get(ds, {}).get(method_name)
            if payload and "speedup_vs_gcn" in payload:
                speeds.append(float(payload["speedup_vs_gcn"]))
        if speeds:
            summary[method_name] = {"avg_speedup_vs_gcn": float(sum(speeds) / len(speeds))}
    os.makedirs(args.log_dir,exist_ok=True)
    ts=datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(args.log_dir,f"efficiency_{ts}.json"),"w") as f: json.dump({"per_dataset": all_r, "summary": summary, "seed": args.seed},f,indent=2)
if __name__ == "__main__": main()
