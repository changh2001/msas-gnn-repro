"""生成10组60/20/20随机划分。"""
import argparse, logging, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42,123,456,789,2021,2022,2023,2024,2025,2026])
    parser.add_argument("--root", default="data/raw")
    parser.add_argument("--split_dir", default="data/splits")
    parser.add_argument("--datasets", nargs="+", default=["cora","citeseer","pubmed","chameleon","squirrel"])
    args = parser.parse_args(); os.makedirs(args.split_dir, exist_ok=True)
    from msas_gnn.data.dataset_factory import load_dataset
    from msas_gnn.data.split_manager import load_or_create_split
    for ds in args.datasets:
        data, _, _ = load_dataset(ds, root=args.root)
        for s in args.seeds: load_or_create_split(ds, data.y, s, split_dir=args.split_dir)
        print(f"[{ds}] {len(args.seeds)}组划分已生成")
if __name__ == "__main__": main()
