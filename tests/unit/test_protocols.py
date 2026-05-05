"""协议与工程口径测试。"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))


def test_leakage_guard_detects_pairwise_overlap():
    from msas_gnn.data.leakage_guard import check_feature_leakage

    x = torch.randn(5, 3)
    train_mask = torch.tensor([1, 1, 0, 0, 0], dtype=torch.bool)
    val_mask = torch.tensor([0, 1, 1, 0, 0], dtype=torch.bool)
    test_mask = torch.tensor([0, 0, 0, 1, 1], dtype=torch.bool)
    assert check_feature_leakage(x, train_mask, test_mask, "toy", val_mask=val_mask) is False


def test_sparse_paper_protocol_normalizes_large_graph_to_batch_time():
    from msas_gnn.evaluation.efficiency import infer_latency_sparse_paper_protocol
    from msas_gnn.typing import ThetaFixed

    theta = torch.eye(8).to_sparse_csr()
    phi = torch.randn(8, 4)
    theta_fixed = ThetaFixed(theta=theta, k_bar=1.0, sparsity=0.875)
    stats = infer_latency_sparse_paper_protocol(
        theta_fixed,
        phi,
        num_nodes=8,
        batch_size=4,
        warmup=1,
        repeat=2,
        is_large_graph=True,
        device="cpu",
    )
    assert "median_ms_per_batch" in stats
    assert "nodes_per_sec" in stats
    assert "full_graph_ms" in stats


def test_baseline_registry_ignores_unsupported_kwargs_for_gcn():
    from msas_gnn.baselines.gcn import GCN
    from msas_gnn.baselines.registry import get_baseline

    model = get_baseline(
        "gcn",
        in_channels=16,
        hidden_channels=8,
        out_channels=3,
        dropout=0.5,
        num_layers=2,
        K=10,
        alpha=0.2,
    )
    assert isinstance(model, GCN)


def test_baseline_registry_supports_sage():
    from msas_gnn.baselines.sage import GraphSAGE
    from msas_gnn.baselines.registry import get_baseline

    model = get_baseline(
        "sage",
        in_channels=16,
        hidden_channels=8,
        out_channels=3,
        dropout=0.5,
        num_layers=2,
    )
    assert isinstance(model, GraphSAGE)


def test_recent_baselines_are_instantiable_and_forward():
    from msas_gnn.baselines.registry import get_baseline

    x = torch.randn(6, 8)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 0, 2], [1, 2, 3, 4, 5, 0, 2, 0]],
        dtype=torch.long,
    )
    for name in ["graphsaint", "nodeformer", "difformer", "sgformer", "nagphormer"]:
        model = get_baseline(
            name,
            in_channels=8,
            hidden_channels=4,
            out_channels=3,
            dropout=0.1,
            num_layers=1,
            K=2,
        )
        out = model(x, edge_index)
        assert out.shape == (6, 3)


def test_break_even_extracts_large_graph_seconds_from_efficiency_payload():
    from msas_gnn.evaluation.break_even import extract_break_even_inputs

    payload = {
        "per_dataset": {
            "ogbn_arxiv": {
                "gcn": {"median_ms_per_batch": 10.0, "batch_size": 1024},
                "msas_gnn": {
                    "median_ms_per_batch": 2.0,
                    "batch_size": 1024,
                    "preprocess_seconds": 50.0,
                    "preprocess_breakdown": {"alternating_opt": 40.0},
                },
            }
        }
    }
    stats = extract_break_even_inputs(payload, "ogbn_arxiv", num_nodes=2500)
    assert stats["t_dense"] == 0.03
    assert stats["t_sparse"] == 0.006
    assert stats["t_pre"] == 50.0


def test_dense_parameter_count_reports_model_numel():
    import torch.nn as nn

    from msas_gnn.evaluation.efficiency import count_model_parameters

    model = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2))
    stats = count_model_parameters(model)
    expected = sum(param.numel() for param in model.parameters())
    assert stats["parameter_count"] == expected
    assert stats["parameter_count_trainable"] == expected
    assert stats["parameter_count_millions"] == expected / 1_000_000.0
    assert stats["parameter_count_protocol"] == "dense_model_parameters"


def test_sparse_parameter_count_uses_theta_nnz_plus_head():
    from msas_gnn.evaluation.efficiency import count_sparse_inference_parameters
    from msas_gnn.typing import ThetaFixed

    theta = torch.tensor(
        [
            [1.0, 0.0, 2.0],
            [0.0, 0.0, 0.0],
            [3.0, 4.0, 0.0],
        ]
    ).to_sparse_csr()
    theta_fixed = ThetaFixed(theta=theta, k_bar=4.0 / 3.0, sparsity=5.0 / 9.0)
    stats = count_sparse_inference_parameters(theta_fixed, extra_dense_params=7)
    assert stats["parameter_count_sparse_nnz"] == 4
    assert stats["parameter_count"] == 11
    assert stats["parameter_count_millions"] == 11 / 1_000_000.0
    assert stats["parameter_count_protocol"] == "fixed_sparse_theta_nnz_plus_head"


def test_linear_feature_transform_ridge_init_recovers_known_mapping():
    from msas_gnn.training.feature_transform import (
        compute_phi_tilde,
        initialize_linear_feature_transform,
    )

    torch.manual_seed(0)
    x = torch.randn(32, 4)
    w_true = torch.tensor(
        [
            [1.0, 0.0, 0.5],
            [0.0, 1.0, -0.5],
            [0.5, -0.2, 0.0],
            [0.0, 0.3, 1.0],
        ]
    )
    h_star = x @ w_true
    transform = initialize_linear_feature_transform(x, h_star, ridge=1e-6, device="cpu")
    phi = transform(x)
    assert torch.allclose(phi, h_star, atol=1e-4)
    phi_tilde = compute_phi_tilde(transform, x)
    assert phi_tilde.shape == h_star.shape


def test_ablation_summary_includes_sparsity_latency_and_noise_fields():
    import pytest

    from msas_gnn.evaluation.ablation_runner import summarize_seed_results

    cfg = {"ablation_id": "b1", "dataset": "cora"}
    results = [
        {
            "test_acc": 0.87,
            "epsilon_approx": 0.18,
            "sparsity": 0.75,
            "inference_ms": 1.9,
            "k_bar": 3.8,
            "support_total": 90,
            "candidate_total": 360,
            "stage_times": {"alternating_opt": 10.0},
        },
        {
            "test_acc": 0.89,
            "epsilon_approx": 0.16,
            "sparsity": 0.73,
            "inference_ms": 2.1,
            "k_bar": 3.6,
            "support_total": 92,
            "candidate_total": 368,
            "stage_times": {"alternating_opt": 12.0},
        },
    ]
    summary = summarize_seed_results(cfg, results, [], extra={"noise_mean_acc": 0.8})
    assert summary["mean_acc"] == pytest.approx(0.88)
    assert summary["mean_sparsity"] == pytest.approx(0.74)
    assert summary["mean_inference_ms"] == pytest.approx(2.0)
    assert summary["mean_alternating_opt_seconds"] == pytest.approx(11.0)
    assert summary["mean_support_total"] == pytest.approx(91.0)
    assert summary["mean_candidate_total"] == pytest.approx(364.0)
    assert summary["noise_mean_acc"] == pytest.approx(0.8)
    assert summary["protocols"]["sparsity_protocol"] == "candidate_pruning_rate"
