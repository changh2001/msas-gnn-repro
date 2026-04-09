"""批量计算六个数据集的图复杂度指标（MetricBundle）。对应论文第3章。"""
import argparse, logging, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))
logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
ALL = ["cora","citeseer","pubmed","chameleon","squirrel","ogbn_arxiv"]
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true"); parser.add_argument("--dataset", default=None)
    parser.add_argument("--K_eig", type=int, default=50); parser.add_argument("--raw_root", default="data/raw")
    parser.add_argument("--cache_dir", default="data/cache/spectral")
    args = parser.parse_args(); os.makedirs(args.cache_dir, exist_ok=True)
    datasets = ALL if args.all else ([args.dataset] if args.dataset else ["cora"])
    from msas_gnn.data.dataset_factory import load_dataset
    from msas_gnn.data.transforms import add_self_loops_transform
    from msas_gnn.spectral.metric_bundle import compute_metric_bundle
    for ds in datasets:
        t0 = time.time(); data, nc, nf = load_dataset(ds, root=args.raw_root)
        data = add_self_loops_transform(data)
        logger.info(f"[{ds}] n={data.num_nodes} m={data.num_edges} d={nf}")
        cache = os.path.join(args.cache_dir, f"{ds}_metrics.pt")
        bundle = compute_metric_bundle(data, K_eig=args.K_eig, cache_path=cache)
        logger.info(f"[{ds}] 完成 {time.time()-t0:.1f}s | h_edge={bundle.h_edge:.4f} | λ_gap={bundle.eigenvalues[1]:.6f}")
if __name__ == "__main__": main()
