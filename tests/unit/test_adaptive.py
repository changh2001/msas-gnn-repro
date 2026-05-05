"""自适应参数单元测试。"""
import pytest, torch, sys, os
from torch_geometric.data import Data
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))
from msas_gnn.typing import MetricBundle
def bundle(n=50,K=20):
    return MetricBundle(spectral_energy=torch.abs(torch.randn(n)),h_norm=torch.rand(n),c_deg=torch.rand(n),
                        core=torch.randint(1,5,(n,)).float(),h_edge=0.5,eigenvalues=torch.linspace(0,2,K),eigenvectors=torch.randn(n,K))
def test_tau_domain():
    from msas_gnn.adaptive.tau_builder import build_tau
    b=bundle(); tau_base,tau_min=1e-3,1e-7; tau=build_tau(b,tau_base=tau_base,tau_min=tau_min)
    assert not tau.isnan().any() and not tau.isinf().any()
    assert tau.min()>=tau_min-1e-10 and tau.max()<=tau_base+1e-10
def test_budget_conservation():
    from msas_gnn.adaptive.hop_budget import allocate_hop_budget_for_candidates
    n,k,L=100,50,3; tau=torch.full((n,),1e-3)
    candidate_sets = [
        {i: list(range(2)) for i in range(n)},
        {i: list(range(5)) for i in range(n)},
        {i: list(range(11)) for i in range(n)},
    ]
    budget=allocate_hop_budget_for_candidates(
        candidate_sets,
        tau,
        cfg={"hop_dim": {"strategy": "spectral_gap_reference", "p_base": 0.6, "p_min": 0.1, "kappa": 2.0}},
        lambda_gap=0.5,
    )
    capacities = torch.tensor([2, 5, 11]).repeat(n, 1)
    assert budget.shape == (n, L)
    assert budget.min() >= 0
    assert (budget <= capacities).all()
    assert not (budget.sum(dim=1) == k).all()


def test_varrho_decay_and_rho_scale_are_distinct():
    from msas_gnn.adaptive.hop_budget import allocate_hop_budget_for_candidates

    n = 4
    tau = torch.full((n,), 1e-3)
    candidate_sets = [
        {i: list(range(10)) for i in range(n)},
        {i: list(range(10)) for i in range(n)},
        {i: list(range(10)) for i in range(n)},
    ]
    base = allocate_hop_budget_for_candidates(
        candidate_sets,
        tau,
        cfg={"hop_dim": {"strategy": "near_engineering", "p_base": 0.6, "p_min": 0.1, "varrho": 0.5, "rho": 1.0}},
    )
    scaled = allocate_hop_budget_for_candidates(
        candidate_sets,
        tau,
        cfg={"hop_dim": {"strategy": "near_engineering", "p_base": 0.6, "p_min": 0.1, "varrho": 0.5, "rho": 0.5}},
    )
    assert torch.equal(base[0], torch.tensor([6, 3, 2]))
    assert (scaled <= base).all()
    assert scaled[0, 0] < base[0, 0]
def test_freq_weights_shape_and_threshold():
    from msas_gnn.adaptive.joint_budget import build_adaptive_params
    b=bundle()
    cfg={"ablation_id":"b5","node_dim":{"tau_base":1e-3,"tau_min":1e-7,"c_e":1.0},"hop_dim":{"L":3},"lars":{"k":50}}
    params = build_adaptive_params(b, cfg)
    assert params.freq_weights.shape[0] == b.eigenvalues.shape[0]


def test_energy_threshold_uses_spectral_energy_median():
    from msas_gnn.adaptive.tau_builder import resolve_energy_threshold

    b = bundle(n=5, K=4)
    b = b._replace(spectral_energy=torch.tensor([1.0, 2.0, 4.0, 8.0, 16.0]))
    assert resolve_energy_threshold(b, c_e=0.5) == 2.0

def test_b2_rnd_is_tau_perturbation_not_factor_drop():
    from msas_gnn.adaptive.joint_budget import build_adaptive_params
    b = bundle()
    base_cfg = {
        "seed": 42,
        "node_dim": {"tau_base": 1e-3, "tau_min": 1e-7, "c_e": 1.0, "gamma": 0.5},
        "hop_dim": {"L": 3},
        "lars": {"k": 50},
    }
    p_b2 = build_adaptive_params(b, {**base_cfg, "ablation_id": "b2", "node_dim": {**base_cfg["node_dim"], "use_spectral_energy": True, "use_centrality": True, "use_kcore": False, "use_entropy": False}})
    p_rnd = build_adaptive_params(b, {**base_cfg, "ablation_id": "b2_rnd", "node_dim": {**base_cfg["node_dim"], "use_spectral_energy": True, "use_centrality": True, "use_kcore": False, "use_entropy": False, "random_tau_perturbation": True}})
    assert not torch.allclose(p_b2.tau, p_rnd.tau)
    assert p_rnd.tau.min() >= 1e-7 - 1e-10
    assert p_rnd.tau.max() <= 1e-3 + 1e-10


def test_sdgnn_pure_uses_global_lambda_without_metric_bundle():
    from msas_gnn.adaptive.joint_budget import build_adaptive_params

    data = Data(x=torch.randn(4, 3), edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long))
    cfg = {"ablation_id": "sdgnn_pure", "lambda_reg": 1e-5, "lars": {"k": 12}}
    params = build_adaptive_params(None, cfg, data=data)
    assert torch.allclose(params.tau, torch.full((4,), 1e-5))
    assert params.k_budget.shape == (4, 1)
    assert (params.k_budget == 12).all()
