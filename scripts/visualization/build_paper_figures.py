"""一键生成论文图与 LaTeX 表格。"""
import argparse
import logging
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MAIN = [
    ("scripts/visualization/build_paper_tables.py", ["--main"]),
    ("scripts/visualization/plot_sensitivity_cora.py", []),
    ("scripts/visualization/plot_tau_distribution.py", ["--dataset", "cora"]),
    ("scripts/visualization/plot_tsne.py", ["--dataset", "cora"]),
]
ABLATION = [
    ("scripts/visualization/build_paper_tables.py", ["--ablation"]),
]
EFFICIENCY = [
    ("scripts/visualization/build_paper_tables.py", ["--efficiency"]),
]
SUPPLEMENTAL = [
    ("scripts/visualization/build_paper_tables.py", ["--supplemental"]),
    ("scripts/visualization/plot_retention_sweep.py", []),
    ("scripts/visualization/plot_sensitivity_supplemental.py", []),
    ("scripts/visualization/plot_sigma_proxy.py", []),
]


def run(script, extra_args):
    proc = subprocess.run([sys.executable, script] + extra_args, capture_output=True, text=True)
    if proc.stdout:
        logger.info(proc.stdout.strip())
    if proc.returncode != 0:
        if proc.stderr:
            logger.error(proc.stderr.strip())
        raise RuntimeError(f"图表脚本失败: {script}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", action="store_true")
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--efficiency", action="store_true")
    parser.add_argument("--supplemental", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    scripts = (
        (MAIN if args.main or args.all else [])
        + (ABLATION if args.ablation or args.all else [])
        + (EFFICIENCY if args.efficiency or args.all else [])
        + (SUPPLEMENTAL if args.supplemental or args.all else [])
    )
    if not scripts:
        parser.print_help()
        return
    logger.info("生成%s个论文产物...", len(scripts))
    for script, extra_args in scripts:
        run(script, extra_args)
    logger.info("完成！outputs/figures/、outputs/figures/supplemental/、outputs/tables/")
if __name__ == "__main__": main()
