"""打包全部复现结果。"""
import argparse, logging, os, tarfile
from datetime import datetime
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger=logging.getLogger(__name__)
def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--output_dir",default="."); parser.add_argument("--include",nargs="+",default=["outputs/results","outputs/figures","outputs/logs"]); args=parser.parse_args()
    ts=datetime.now().strftime("%Y%m%d_%H%M%S")
    path=os.path.join(args.output_dir,f"msas_gnn_repro_bundle_{ts}.tar.gz")
    logger.info(f"打包 → {path}")
    with tarfile.open(path,"w:gz") as tar:
        for p in args.include:
            if os.path.exists(p): tar.add(p,arcname=os.path.basename(p)); logger.info(f"  + {p}")
    logger.info(f"完成：{path} ({os.path.getsize(path)/1024/1024:.1f}MB)")
if __name__ == "__main__": main()
