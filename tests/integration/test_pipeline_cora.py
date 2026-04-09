"""Cora完整流水线冒烟测试（单seed=42，约5分钟）。"""
import pytest, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))
@pytest.mark.skipif(not os.path.exists("data/raw/cora"), reason="数据集未下载")
def test_cora_smoke():
    import yaml
    from msas_gnn.training.msas_trainer import MSASTrainer
    with open("configs/models/msas_gnn_b5.yaml") as f: cfg=yaml.safe_load(f)
    with open("configs/datasets/cora.yaml") as f: cfg.update(yaml.safe_load(f))
    cfg.update({"dataset":"cora","ablation_id":"b5"})
    r=MSASTrainer(cfg).run_single_seed(42)
    assert r["test_acc"]>0.80, f"准确率过低：{r['test_acc']:.4f}"
    assert r["epsilon_approx"]<0.5
    print(f"\n冒烟测试通过：acc={r['test_acc']:.4f} ε={r['epsilon_approx']:.4f}")
