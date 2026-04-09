"""Wilcoxon显著性检验。"""
import argparse, glob, json, logging, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
EXP_P={"cora":0.002,"citeseer":0.001,"pubmed":0.048,"ogbn_arxiv":0.009,"chameleon":0.003,"squirrel":0.011}


def _read_json(path):
    with open(path, encoding="utf-8") as fp:
        return json.load(fp)


def _collect_seed_scores(files, dataset):
    paired = {"b5": {}, "b0": {}}
    for path in files:
        try:
            payload = _read_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("跳过损坏结果文件 %s: %s", path, exc)
            continue
        if payload.get("dataset") != dataset:
            continue
        ablation_id = payload.get("ablation_id") or payload.get("method_id")
        if ablation_id in ("b5", "msas_gnn_b5"):
            bucket = paired["b5"]
        elif ablation_id in ("b0", "sdgnn"):
            bucket = paired["b0"]
        else:
            continue
        for row in payload.get("per_seed", []):
            seed = row.get("seed")
            score = row.get("test_acc")
            if seed is not None and score is not None:
                bucket[int(seed)] = float(score)
    common_seeds = sorted(set(paired["b5"]) & set(paired["b0"]))
    return common_seeds, [paired["b5"][s] for s in common_seeds], [paired["b0"][s] for s in common_seeds]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp",default="main"); parser.add_argument("--results_dir",default="outputs/results")
    parser.add_argument("--output",default="outputs/results/significance_summary.json")
    args = parser.parse_args()
    from msas_gnn.evaluation.significance import run_wilcoxon
    files = sorted(
        glob.glob(os.path.join(args.results_dir, "**", "*.json"), recursive=True),
        key=os.path.getmtime,
    )
    all_r={}
    for ds in ["cora","citeseer","pubmed","ogbn_arxiv","chameleon","squirrel"]:
        seeds, b5, b0 = _collect_seed_scores(files, ds)
        if len(seeds) >= 5:
            r=run_wilcoxon(b5, b0, ds); all_r[ds]=r
            logger.info(
                "  %s: p=%.4f (预期≈%s) 配对种子=%s %s",
                ds,
                r["p_value"],
                EXP_P.get(ds),
                seeds,
                "[OK]" if r["significant"] else "[NOT SIG]",
            )
        else:
            logger.warning("  %s: 可配对种子不足，跳过显著性检验", ds)
    os.makedirs(os.path.dirname(args.output),exist_ok=True)
    with open(args.output,"w") as f: json.dump(all_r,f,indent=2)
if __name__ == "__main__": main()
