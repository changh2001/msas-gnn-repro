.PHONY: reproduce-main reproduce-ablation reproduce-efficiency reproduce-supplemental \
        reproduce-all smoke-test-cora download-data check-env export test

check-env:
	python scripts/setup/verify_env.py

download-data: check-env
	python scripts/data/download_all.py
	python scripts/data/build_splits.py --seeds 42 123 456 789 2021 2022 2023 2024 2025 2026

smoke-test-cora:
	python run.py --task smoke --dataset cora --seed 42 --ablation b5

reproduce-main: check-env download-data
	@echo "=== [1/4] 图指标预处理（约2.2h）==="
	python scripts/preprocess/compute_graph_metrics.py --all
	@echo "=== [2/4] 教师训练与H*缓存（约4h）==="
	python scripts/preprocess/cache_teacher_reprs.py --all
	@echo "=== [3/4] 主实验双GPU并行（约18h）==="
	bash scripts/run_parallel_main.sh
	@echo "=== [4/4] 显著性检验与图表生成（约0.8h）==="
	python scripts/experiments/run_significance.py --exp main
	python scripts/visualization/build_paper_figures.py --main

reproduce-ablation:
	python scripts/experiments/run_ablation_modular.py --datasets cora
	python scripts/experiments/run_ablation_hop_strategy.py --datasets cora chameleon
	python scripts/visualization/build_paper_figures.py --ablation

reproduce-efficiency:
	python scripts/experiments/run_efficiency.py --all
	python scripts/experiments/run_breakeven.py --datasets cora pubmed ogbn_arxiv
	python scripts/visualization/build_paper_figures.py --efficiency

reproduce-supplemental:
	python scripts/experiments/supplemental/run_retention_sweep.py --datasets cora chameleon
	python scripts/experiments/supplemental/run_sensitivity_supplemental.py --datasets chameleon ogbn_arxiv
	python scripts/experiments/supplemental/run_spectral_proxy.py --datasets citeseer ogbn_arxiv
	python scripts/visualization/build_paper_figures.py --supplemental

reproduce-all: reproduce-main reproduce-ablation reproduce-efficiency reproduce-supplemental
	python scripts/visualization/build_paper_figures.py --all

export:
	python scripts/release/export_repro_bundle.py

test:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

test-all:
	pytest tests/ -v

clean-cache:
	rm -rf data/cache/* outputs/*
