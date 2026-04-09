#!/bin/bash
set -e
echo "[INFO] 启动双GPU并行主实验..."
SEEDS=$(python -c "import yaml; print(' '.join(map(str, yaml.safe_load(open('configs/global/seeds.yaml'))['seeds'])))")
CONFIG="configs/models/msas_gnn_b5.yaml"
CUDA_VISIBLE_DEVICES=0 python scripts/experiments/run_main_benchmarks.py --datasets cora citeseer pubmed --config $CONFIG --seeds $SEEDS --log_dir outputs/results/ &
PID_GPU0=$!
CUDA_VISIBLE_DEVICES=1 python scripts/experiments/run_heterophily_benchmarks.py --datasets chameleon squirrel --config $CONFIG --seeds $SEEDS --log_dir outputs/results/
CUDA_VISIBLE_DEVICES=1 python scripts/experiments/run_main_benchmarks.py --datasets ogbn_arxiv --config $CONFIG --seeds $SEEDS --log_dir outputs/results/
wait $PID_GPU0
echo "[INFO] 双GPU主实验完成！"
