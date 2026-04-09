"""评估指标：准确率、Silhouette系数、谱近似误差ε_approx。对应论文§6.1,6.5。"""
import logging
import torch
logger = logging.getLogger(__name__)

def compute_accuracy(logits, labels, mask):
    pred = logits[mask].argmax(1)
    return (pred == labels[mask]).float().mean().item()

def compute_epsilon_approx(h_star, h_hat):
    """ε_approx = (1/√n)·‖H*-Ĥ‖_F。对应论文§3.1.2。"""
    n = h_star.shape[0]
    return (torch.norm(h_star-h_hat, p="fro") / n**0.5).item()

def compute_silhouette(embeddings, labels):
    """Silhouette系数。预期：GCN=0.45, SDGNN=0.61, MSAS-GNN=0.68。"""
    from sklearn.metrics import silhouette_score
    try: return float(silhouette_score(embeddings.cpu().numpy(), labels.cpu().numpy()))
    except Exception as e: logger.warning(f"Silhouette失败：{e}"); return float("nan")
