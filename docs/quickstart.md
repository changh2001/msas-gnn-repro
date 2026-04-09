# 快速开始

```bash
conda env create -f environment.yml
conda activate msas-gnn
pip install -e .
python scripts/setup/verify_env.py
make download-data
make smoke-test-cora
```
