"""下载全部6个数据集。"""
import argparse, logging, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/raw")
    parser.add_argument("--datasets", nargs="+", default=["cora","citeseer","pubmed","chameleon","squirrel","ogbn_arxiv"])
    args = parser.parse_args(); os.makedirs(args.root, exist_ok=True)
    ds = set(args.datasets)
    if ds & {"cora","citeseer","pubmed"}:
        from torch_geometric.datasets import Planetoid
        for n in ["Cora","CiteSeer","PubMed"]:
            logger.info(f"下载 {n}..."); Planetoid(root=os.path.join(args.root,n.lower()),name=n)
    if ds & {"chameleon","squirrel"}:
        from torch_geometric.datasets import WikipediaNetwork
        for n in ["chameleon","squirrel"]:
            logger.info(f"下载 {n}..."); WikipediaNetwork(root=os.path.join(args.root,n),name=n,geom_gcn_preprocess=True)
    if "ogbn_arxiv" in ds:
        try:
            from ogb.nodeproppred import PygNodePropPredDataset
            logger.info("下载 ogbn-arxiv...")
            PygNodePropPredDataset(name="ogbn-arxiv", root=os.path.join(args.root,"ogbn_arxiv"))
        except ImportError: logger.warning("OGB未安装")
    logger.info("下载完成！")
if __name__ == "__main__": main()
