"""B0基线可运行性验证。"""
import pytest, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))
@pytest.mark.skipif(not os.path.exists("data/raw/cora"), reason="数据集未下载")
def test_b0():
    import yaml
    from msas_gnn.training.msas_trainer import MSASTrainer
    with open("configs/ablations/b0_sdgnn.yaml") as f: cfg=yaml.safe_load(f) or {}
    with open("configs/datasets/cora.yaml") as f: cfg.update(yaml.safe_load(f))
    cfg.update({"dataset":"cora","ablation_id":"b0"})
    r=MSASTrainer(cfg).run_single_seed(42)
    assert r["test_acc"]>0.70
    print(f"\nB0验证通过：acc={r['test_acc']:.4f}")
