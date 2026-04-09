"""评估指标测试。"""
import torch, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))
def test_accuracy():
    from msas_gnn.evaluation.metrics import compute_accuracy
    n=100; logits=torch.zeros(n,3); labels=torch.randint(0,3,(n,))
    for i in range(n): logits[i,labels[i]]=10.0
    assert abs(compute_accuracy(logits,labels,torch.ones(n,dtype=torch.bool))-1.0)<1e-6
def test_epsilon():
    from msas_gnn.evaluation.metrics import compute_epsilon_approx
    h=torch.randn(50,16); assert abs(compute_epsilon_approx(h,h))<1e-6
