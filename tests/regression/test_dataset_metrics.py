"""数据集统计回归测试（需下载数据集）。"""
import pytest, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))
EXPECTED={"cora":0.81,"citeseer":0.74,"pubmed":0.80,"chameleon":0.23,"squirrel":0.22}
@pytest.mark.skipif(not os.path.exists("data/raw/cora"), reason="数据集未下载")
def test_cora_homophily():
    from msas_gnn.data.dataset_factory import load_dataset
    from msas_gnn.data.transforms import add_self_loops_transform
    from msas_gnn.spectral.homophily import compute_edge_homophily
    data,_,_=load_dataset("cora"); data=add_self_loops_transform(data)
    assert abs(compute_edge_homophily(data)-EXPECTED["cora"])<0.05
