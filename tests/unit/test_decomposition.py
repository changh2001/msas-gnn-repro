"""分解模块单元测试。"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))


def test_candidate_builder_supports_keep_then_sample():
    from types import SimpleNamespace

    from msas_gnn.decomposition.candidate_builder import build_bfs_candidate_sets

    edge_index = torch.tensor(
        [
            [0, 0, 1, 1, 2, 2],
            [1, 2, 3, 4, 5, 6],
        ],
        dtype=torch.long,
    )
    data = SimpleNamespace(num_nodes=7, edge_index=edge_index)
    cands = build_bfs_candidate_sets(
        data,
        L=2,
        seed=0,
        keep_complete_hops=1,
        sampled_max_candidates=1,
    )
    assert set(cands[0][0]) == {1, 2}
    assert len(cands[1][0]) == 1


def test_sdgnn_candidate_builder_supports_khop_plus_recursive_sampling():
    from types import SimpleNamespace

    from msas_gnn.decomposition.candidate_builder import build_sdgnn_candidate_set

    edge_index = torch.tensor(
        [
            [0, 0, 1, 1, 2, 3],
            [1, 2, 3, 4, 5, 6],
        ],
        dtype=torch.long,
    )
    data = SimpleNamespace(num_nodes=7, edge_index=edge_index)
    cands = build_sdgnn_candidate_set(
        data,
        base_hops=1,
        extra_hops=1,
        fanouts=[1],
        seed=0,
    )
    assert set(cands[0][:2]) == {1, 2}
    assert len(cands[0]) == 4


def test_phase_theta_subset_keeps_residual_cascade():
    from msas_gnn.decomposition.theta_optimizer import run_phase_theta
    from msas_gnn.typing import AdaptiveParamSet

    h_star = torch.tensor(
        [
            [1.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    phi_tilde = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    params = AdaptiveParamSet(
        tau=torch.zeros(3),
        k_budget=torch.tensor([[1, 1], [0, 0], [0, 0]], dtype=torch.long),
        freq_weights=torch.ones(2),
    )
    candidate_sets = [
        {0: [1], 1: [], 2: []},
        {0: [2], 1: [], 2: []},
    ]
    theta_fixed = run_phase_theta(
        h_star,
        phi_tilde,
        params,
        candidate_sets,
        cfg={},
        node_indices=[0],
    )
    theta_dense = theta_fixed.theta.to_dense()
    assert theta_dense[1, 0] != 0
    assert theta_dense[2, 0] != 0
    assert torch.allclose(theta_dense[:, 1:], torch.zeros_like(theta_dense[:, 1:]))


def test_phase_theta_writes_self_channel_diagonal():
    from msas_gnn.decomposition.theta_optimizer import run_phase_theta
    from msas_gnn.typing import AdaptiveParamSet

    h_star = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    phi_tilde = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    params = AdaptiveParamSet(
        tau=torch.zeros(2),
        k_budget=torch.zeros((2, 1), dtype=torch.long),
        freq_weights=torch.ones(1),
    )
    theta_fixed = run_phase_theta(
        h_star,
        phi_tilde,
        params,
        candidate_sets=[{0: [], 1: []}],
        cfg={},
        node_indices=[0],
    )
    theta_dense = theta_fixed.theta.to_dense()
    assert theta_dense[0, 0] != 0
    assert theta_fixed.candidate_total == 1


def test_phase_theta_shared_target_only_changes_layer_target_path():
    from msas_gnn.decomposition.theta_optimizer import run_phase_theta
    from msas_gnn.typing import AdaptiveParamSet

    h_star = torch.tensor([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    phi_tilde = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    params = AdaptiveParamSet(
        tau=torch.zeros(3),
        k_budget=torch.tensor([[1, 1], [0, 0], [0, 0]], dtype=torch.long),
        freq_weights=torch.ones(2),
    )
    candidate_sets = [{0: [1], 1: [], 2: []}, {0: [2], 1: [], 2: []}]

    residual = run_phase_theta(
        h_star,
        phi_tilde,
        params,
        candidate_sets,
        cfg={"lars": {"theta_solver_mode": "residual_cascade"}},
        node_indices=[0],
    ).theta.to_dense()
    shared = run_phase_theta(
        h_star,
        phi_tilde,
        params,
        candidate_sets,
        cfg={"lars": {"theta_solver_mode": "shared_target"}},
        node_indices=[0],
    ).theta.to_dense()

    assert residual[1, 0] != 0
    assert residual[2, 0] == 0
    assert shared[1, 0] != 0
    assert shared[2, 0] != 0


def test_phase_theta_sdgnn_uses_flat_candidate_pool():
    from msas_gnn.decomposition.theta_optimizer import run_phase_theta_sdgnn
    from msas_gnn.typing import AdaptiveParamSet

    h_star = torch.tensor(
        [
            [1.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    phi_tilde = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    params = AdaptiveParamSet(
        tau=torch.zeros(3),
        k_budget=torch.tensor([[2], [2], [2]], dtype=torch.long),
        freq_weights=torch.ones(1),
    )
    candidate_set = {0: [1, 2], 1: [], 2: []}
    theta_fixed = run_phase_theta_sdgnn(
        h_star,
        phi_tilde,
        params,
        candidate_set,
        cfg={},
        node_indices=[0],
    )
    theta_dense = theta_fixed.theta.to_dense()
    assert theta_dense[1, 0] != 0
    assert theta_dense[2, 0] != 0
    assert theta_fixed.support_total == 2
    assert theta_fixed.candidate_total == 2


def test_phase_theta_b0_uses_flat_bfs_pool():
    from msas_gnn.decomposition.theta_optimizer import run_phase_theta_bfs_flat
    from msas_gnn.typing import AdaptiveParamSet

    h_star = torch.tensor([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    phi_tilde = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    params = AdaptiveParamSet(
        tau=torch.zeros(3),
        k_budget=torch.zeros((3, 2), dtype=torch.long),
        freq_weights=torch.ones(1),
    )
    candidate_sets = [{0: [1], 1: [], 2: []}, {0: [2], 1: [], 2: []}]
    theta_fixed = run_phase_theta_bfs_flat(
        h_star,
        phi_tilde,
        params,
        candidate_sets,
        cfg={"lars": {"k": 3}},
        node_indices=[0],
    )
    theta_dense = theta_fixed.theta.to_dense()
    assert theta_dense[1, 0] != 0
    assert theta_dense[2, 0] != 0
    assert theta_fixed.candidate_total == 3
