"""预处理摊销分析（论文表6.7，Q_be计算）。"""
import argparse
import json
import logging
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets",nargs="+",default=["cora","pubmed","ogbn_arxiv"])
    parser.add_argument("--log_dir",default="outputs/results")
    parser.add_argument("--results_dir", default="outputs/results")
    parser.add_argument("--efficiency_file", default=None)
    parser.add_argument("--dense_method", default="gcn")
    parser.add_argument("--sparse_method", default="msas_gnn")
    args = parser.parse_args()
    from msas_gnn.data.dataset_factory import load_dataset
    from msas_gnn.evaluation.break_even import (
        compute_break_even,
        extract_break_even_inputs,
        find_latest_efficiency_result,
        load_json,
    )

    efficiency_file = args.efficiency_file or find_latest_efficiency_result(args.results_dir)
    logger.info("使用效率结果：%s", efficiency_file)
    efficiency_payload = load_json(efficiency_file)
    results={}
    for ds in args.datasets:
        data, _, _ = load_dataset(ds)
        inputs = extract_break_even_inputs(
            efficiency_payload,
            ds,
            dense_method=args.dense_method,
            sparse_method=args.sparse_method,
            num_nodes=int(data.num_nodes),
        )
        r = compute_break_even(inputs["t_pre"], inputs["t_dense"], inputs["t_sparse"], ds)
        r["dense_method"] = args.dense_method
        r["sparse_method"] = args.sparse_method
        r["efficiency_file"] = efficiency_file
        r["preprocess_breakdown"] = inputs["sparse_payload"].get("preprocess_breakdown", {})
        results[ds] = r
        logger.info(
            "  %s: t_pre=%.2fs t_dense=%.4fs t_sparse=%.4fs Q_be=%.0f",
            ds,
            r["t_pre"],
            r["t_dense"],
            r["t_sparse"],
            r["Q_be"],
        )
    os.makedirs(args.log_dir,exist_ok=True)
    with open(os.path.join(args.log_dir,"breakeven_analysis.json"),"w") as f: json.dump(results,f,indent=2)
if __name__ == "__main__": main()
