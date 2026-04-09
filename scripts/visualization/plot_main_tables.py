"""生成表6.2/6.3 LaTeX输出。"""
import argparse, glob, json, logging, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
HOMOPHILY_ROWS = [
    ("gcn", "GCN"),
    ("sgc", "SGC"),
    ("pprgo", "PPRGo"),
    ("glnn", "GLNN"),
    ("b0", "SDGNN"),
    ("b5", "MSAS-GNN"),
]
HETEROPHILY_ROWS = [
    ("gcn", "GCN"),
    ("sgc", "SGC"),
    ("pprgo", "PPRGo"),
    ("geom_gcn", "\\textit{Geom-GCN}"),
    ("h2gcn", "\\textit{H2GCN}"),
    ("glnn", "GLNN"),
    ("b0", "SDGNN"),
    ("b5", "MSAS-GNN"),
]


def _payload_method_id(payload):
    method_id = payload.get("method_id")
    if method_id:
        return str(method_id)
    ablation_id = payload.get("ablation_id")
    if ablation_id in ("b0", "b5"):
        return str(ablation_id)
    return None


def _load_latest_tables(files):
    table = {}
    for path in sorted(files, key=os.path.getmtime):
        try:
            with open(path, encoding="utf-8") as fp:
                payload = json.load(fp)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("跳过损坏结果文件 %s: %s", path, exc)
            continue
        dataset = payload.get("dataset")
        mean_acc = payload.get("mean_acc")
        method_id = _payload_method_id(payload)
        if dataset is None or mean_acc is None or method_id is None:
            continue
        key = (method_id, dataset)
        table[key] = {
            "mean": float(mean_acc) * 100.0,
            "std": float(payload.get("std_acc", 0.0)) * 100.0,
        }
    return table


def _print_table(title, table, datasets, rows):
    print(title)
    print("方法 & " + " & ".join(datasets) + " \\\\")
    for method_id, method_name in rows:
        row = [method_name]
        for dataset in datasets:
            value = table.get((method_id, dataset))
            row.append(f"{value['mean']:.1f}\\pm{value['std']:.1f}" if value else "--")
        print(" & ".join(row) + " \\\\")
    print()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--results_dir",default="outputs/results"); args = parser.parse_args()
    files = glob.glob(os.path.join(args.results_dir,"**","*.json"), recursive=True)
    table = _load_latest_tables(files)
    _print_table("表6.2 引文网络/大规模图", table, ["cora", "citeseer", "pubmed", "ogbn_arxiv"], HOMOPHILY_ROWS)
    _print_table("表6.3 网页图", table, ["chameleon", "squirrel"], HETEROPHILY_ROWS)
if __name__ == "__main__": main()
