"""验证Chameleon/Squirrel无 split leakage（train/val/test 互斥）。"""
import argparse, logging, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["chameleon","squirrel"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42,123,456,789,2021,2022,2023,2024,2025,2026])
    args = parser.parse_args()
    from msas_gnn.data.dataset_factory import load_dataset
    from msas_gnn.data.split_manager import load_or_create_split
    from msas_gnn.data.leakage_guard import check_feature_leakage
    all_ok = True
    for ds in args.datasets:
        data, _, _ = load_dataset(ds)
        for seed in args.seeds:
            tm, vm, tsm = load_or_create_split(ds, data.y, seed)
            ok = check_feature_leakage(data.x, tm, tsm, f"{ds}/seed={seed}", val_mask=vm)
            all_ok = all_ok and ok
    if not all_ok:
        raise SystemExit(1)
if __name__ == "__main__": main()
