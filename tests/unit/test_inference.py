"""推理口径一致性测试。"""
import torch, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))
def test_shape():
    from msas_gnn.decomposition.inference import infer_h_hat
    from msas_gnn.typing import ThetaFixed
    n,d=50,32; th=torch.randn(n,n)*0.1; tf=ThetaFixed(theta=th.to_sparse_csr(),k_bar=5.0,sparsity=0.9)
    assert infer_h_hat(tf,torch.randn(n,d)).shape==(n,d)
def test_consistency():
    from msas_gnn.decomposition.inference import infer_h_hat
    from msas_gnn.typing import ThetaFixed
    n,d=30,16; th=torch.randn(n,n)*0.1; phi=torch.randn(n,d)
    h_d=th.t()@phi; tf=ThetaFixed(theta=th.to_sparse_csr(),k_bar=10.0,sparsity=0.5)
    assert torch.allclose(h_d,infer_h_hat(tf,phi),atol=1e-4)
