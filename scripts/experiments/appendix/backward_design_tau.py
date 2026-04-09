"""附录B.2：τ_base反向设计（预期：α≈8.3，τ_practical≈8.7e-6，σ̃_proxy=1.09）。"""
import argparse, logging, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../src"))
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",default="cora"); parser.add_argument("--epsilon_total",type=float,default=0.1)
    parser.add_argument("--L",type=int,default=2)
    args = parser.parse_args()
    import torch
    from msas_gnn.data.dataset_factory import load_dataset
    from msas_gnn.data.transforms import add_self_loops_transform
    from torch_geometric.utils import degree
    data, _, _ = load_dataset(args.dataset); data = add_self_loops_transform(data)
    n = data.num_nodes; X_norm = float(torch.norm(data.x,"fro").item())
    deg = degree(data.edge_index[0],num_nodes=n)
    alpha_G = float(deg.max().item()/(deg.mean().item()+1e-10))
    S = alpha_G*X_norm/n
    tau_p = args.epsilon_total/(3*args.L*S+1e-20)
    sigma = tau_p*alpha_G/1e-3
    logger.info(f"=== 附录B.2 τ_base反向设计（{args.dataset}）===")
    logger.info(f"  α(G) ≈ {alpha_G:.2f}（预期≈8.3）")
    logger.info(f"  τ_base^practical ≈ {tau_p:.3e}（预期≈8.7e-6）")
    logger.info(f"  σ̃_proxy = {sigma:.3f}（预期≈1.09）")
if __name__ == "__main__": main()
