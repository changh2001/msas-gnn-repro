"""LARS求解器单元测试。"""
import torch, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))
def test_basic():
    from msas_gnn.decomposition.lars_solver import lars_lasso_single
    th=lars_lasso_single(torch.randn(32),torch.randn(20,32),tau=1e-3,budget=5)
    assert th.shape[0]==20 and (th.abs()>1e-8).sum().item()<=5
def test_empty(): 
    from msas_gnn.decomposition.lars_solver import lars_lasso_single
    assert lars_lasso_single(torch.randn(16),torch.zeros(0,16),tau=1e-3,budget=5).shape[0]==0
def test_zero_budget():
    from msas_gnn.decomposition.lars_solver import lars_lasso_single
    assert lars_lasso_single(torch.randn(16),torch.randn(10,16),tau=1e-3,budget=0).abs().max()<1e-8

def test_identity_design_matches_soft_threshold():
    from msas_gnn.decomposition.lars_solver import lars_lasso_single

    r = torch.tensor([1.0, 0.0], dtype=torch.float32)
    phi_candidates = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    th = lars_lasso_single(r, phi_candidates, tau=0.2, budget=2)
    assert torch.allclose(th, torch.tensor([0.8, 0.0]), atol=1e-3)
