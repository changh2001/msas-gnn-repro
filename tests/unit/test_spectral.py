"""谱计算模块单元测试。"""
import pytest, torch, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))
from torch_geometric.data import Data
def make_graph(n=20):
    from msas_gnn.data.transforms import add_self_loops_transform
    row=list(range(n))+list(range(1,n))+[0]; col=list(range(1,n))+[0]+list(range(n))
    ei=torch.tensor([row[:2*n],col[:2*n]],dtype=torch.long)
    data=Data(x=torch.randn(n,8),y=torch.randint(0,3,(n,)),edge_index=ei,num_nodes=n)
    return add_self_loops_transform(data)
def test_lanczos_ordering():
    from msas_gnn.spectral.laplacian import compute_normalized_laplacian_scipy
    from msas_gnn.spectral.lanczos import lanczos_eigenpairs
    data=make_graph(20); L=compute_normalized_laplacian_scipy(data)
    evals,evecs=lanczos_eigenpairs(L,K_eig=10)
    assert (evals[1:]-evals[:-1]>=-1e-6).all(), "特征值应升序"
    assert abs(evals[0].item())<0.1, "eigenvalues[0]应≈0"
    assert evals[1].item()>0, "λ_gap=eigenvalues[1]应>0"
    assert evecs.shape==(20,10)
def test_spectral_energy_nonneg():
    from msas_gnn.spectral.laplacian import compute_normalized_laplacian_scipy
    from msas_gnn.spectral.lanczos import lanczos_eigenpairs
    from msas_gnn.spectral.spectral_energy import compute_spectral_energy
    data=make_graph(20); L=compute_normalized_laplacian_scipy(data)
    ev,evec=lanczos_eigenpairs(L,K_eig=10); E=compute_spectral_energy(ev,evec)
    assert E.shape[0]==20 and (E>=0).all()
def test_entropy_range():
    from msas_gnn.spectral.entropy import compute_local_entropy
    data=make_graph(20); H=compute_local_entropy(data)
    assert (H>=0).all() and (H<=1.0+1e-6).all()
def test_entropy_independent_of_labels():
    from msas_gnn.spectral.entropy import compute_local_entropy
    data=make_graph(20)
    h1 = compute_local_entropy(data)
    data.y = torch.roll(data.y, shifts=3)
    h2 = compute_local_entropy(data)
    assert torch.allclose(h1, h2)
def test_centrality_formula():
    from msas_gnn.spectral.centrality import compute_degree_centrality
    data=make_graph(20); C=compute_degree_centrality(data)
    assert C.max().item()==pytest.approx(2/19,abs=1e-6)


def test_sigma_proxy_is_close_to_one_for_identical_graph():
    from msas_gnn.evaluation.spectral_similarity import compute_sigma_proxy
    from msas_gnn.typing import ThetaFixed

    data = make_graph(20)
    values = torch.ones(data.edge_index.shape[1], dtype=torch.float32)
    theta = torch.sparse_coo_tensor(
        data.edge_index,
        values,
        size=(data.num_nodes, data.num_nodes),
    ).coalesce().to_sparse_csr()
    theta_fixed = ThetaFixed(theta=theta, k_bar=1.0, sparsity=0.0)
    stats = compute_sigma_proxy(data, theta_fixed, num_eigs=10)
    assert stats["sigma_proxy"] == pytest.approx(1.0, abs=5e-2)


def test_proxy_adjacency_symmetrizes_directed_support_and_drops_self_loops():
    from msas_gnn.evaluation.spectral_similarity import build_proxy_adjacency
    from msas_gnn.typing import ThetaFixed

    indices = torch.tensor([[0, 2, 1], [1, 2, 1]], dtype=torch.long)
    values = torch.tensor([0.7, 1.0, 2.0], dtype=torch.float32)
    theta = torch.sparse_coo_tensor(indices, values, size=(3, 3)).coalesce().to_sparse_csr()
    adjacency = build_proxy_adjacency(ThetaFixed(theta=theta, k_bar=1.0, sparsity=0.0))
    dense = adjacency.toarray()
    assert dense[0, 1] == pytest.approx(1.0)
    assert dense[1, 0] == pytest.approx(1.0)
    assert dense[1, 1] == pytest.approx(0.0)
    assert dense[2, 2] == pytest.approx(0.0)
