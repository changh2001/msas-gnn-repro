"""批量训练教师GNN并缓存H*。"""
import argparse, logging, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))
logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
ALL=["cora","citeseer","pubmed","chameleon","squirrel","ogbn_arxiv"]
SEEDS=[42,123,456,789,2021,2022,2023,2024,2025,2026]
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all",action="store_true"); parser.add_argument("--dataset",default=None)
    parser.add_argument("--seeds",nargs="+",type=int,default=SEEDS)
    parser.add_argument("--teacher_config",default="configs/teachers/gcn.yaml")
    parser.add_argument("--raw_root",default="data/raw"); parser.add_argument("--cache_dir",default="data/cache/teachers")
    args = parser.parse_args()
    import yaml
    with open(args.teacher_config) as f: tc = yaml.safe_load(f)
    datasets = ALL if args.all else ([args.dataset] if args.dataset else ["cora"])
    os.makedirs(args.cache_dir, exist_ok=True)
    from msas_gnn.data.dataset_factory import load_dataset
    from msas_gnn.data.transforms import add_self_loops_transform
    from msas_gnn.data.split_manager import load_or_create_split
    from msas_gnn.training.teacher_trainer import TeacherTrainer
    for ds in datasets:
        data, nc, _ = load_dataset(ds, root=args.raw_root); data = add_self_loops_transform(data)
        for seed in args.seeds:
            if ds != "ogbn_arxiv":
                tm,vm,tsm = load_or_create_split(ds, data.y, seed)
                data.train_mask,data.val_mask,data.test_mask = tm,vm,tsm
            cfg = {"dataset":ds,"num_classes":nc,"teacher":tc,"train":tc}
            h = TeacherTrainer(cfg).train_and_cache(data, seed, cache_dir=args.cache_dir)
            logger.info(f"  [{ds}][seed={seed}] H* shape={h.shape}")
if __name__ == "__main__": main()
