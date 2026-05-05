"""配置装配与论文口径回归测试。"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def test_cora_b5_defaults_align_with_current_paper_defaults():
    from msas_gnn.config import load_experiment_config

    cfg = load_experiment_config("cora", ablation_id="b5")
    assert cfg["train"]["lr"] == 0.005
    assert cfg["train"]["dropout"] == 0.3
    assert cfg["teacher"]["layers"] == 3
    assert cfg["teacher"]["hidden_dim"] == 64
    assert cfg["teacher"]["lr"] == 0.005
    assert cfg["teacher"]["dropout"] == 0.3
    assert cfg["node_dim"]["omega_h"] == 0.2
    assert "eta" not in cfg["node_dim"]
    assert cfg["hop_dim"]["varrho"] == 0.7
    assert cfg["hop_dim"]["rho"] == 1.0


def test_cora_b0_uses_explicit_gcn_teacher_defaults():
    from msas_gnn.config import load_experiment_config

    cfg = load_experiment_config("cora", ablation_id="b0")
    assert cfg["teacher_name"] == "gcn"
    assert cfg["teacher"]["layers"] == 3
    assert cfg["teacher"]["hidden_dim"] == 64
    assert cfg["train"]["lr"] == 0.005
    assert cfg["train"]["dropout"] == 0.3


def test_cora_sdgnn_pure_defaults_use_gcn_teacher_and_original_protocol():
    from msas_gnn.config import load_experiment_config

    cfg = load_experiment_config("cora", ablation_id="sdgnn_pure")
    assert cfg["teacher_name"] == "gcn"
    assert cfg["alternating_opt"]["protocol"] == "sdgnn_orig"
    assert cfg["sdgnn_pure"]["base_hops"] == 2
    assert cfg["sdgnn_pure"]["fanouts"] == [10, 10]


def test_b6_is_removed_from_public_ablation_registry():
    from msas_gnn.config import ABLATION_CONFIGS, BASE_MODEL_BY_ABLATION

    assert "b6" not in ABLATION_CONFIGS
    assert "b6" not in BASE_MODEL_BY_ABLATION
