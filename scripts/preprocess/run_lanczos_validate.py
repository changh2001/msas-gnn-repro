"""验证Lanczos近似精度。"""
import argparse, logging, os, sys; import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["cora","citeseer"])
    parser.add_argument("--K_eig", type=int, default=50); parser.add_argument("--raw_root", default="data/raw")
    parser.add_argument("--max_error_threshold", type=float, default=1e-4)
    args = parser.parse_args()
    from msas_gnn.data.dataset_factory import load_dataset
    from msas_gnn.data.transforms import add_self_loops_transform
    from msas_gnn.spectral.laplacian import compute_normalized_laplacian_scipy
    from msas_gnn.spectral.lanczos import lanczos_eigenpairs
    import scipy.linalg as sla
    for ds in args.datasets:
        data, _, _ = load_dataset(ds, root=args.raw_root); data = add_self_loops_transform(data)
        if data.num_nodes > 5000: logger.info(f"[{ds}] n>5000，跳过全谱对照"); continue
        L = compute_normalized_laplacian_scipy(data)
        ev, _ = lanczos_eigenpairs(L, K_eig=args.K_eig)
        ref = sla.eigh(L.toarray(), eigvals_only=True)[:args.K_eig]
        err = np.max(np.abs(ev.numpy()-ref))
        logger.info(f"[{ds}] 最大误差={err:.2e} {'[OK]' if err<args.max_error_threshold else '[FAIL]'}")
if __name__ == "__main__": main()
