"""损失函数：表示逼近+任务分类。对应论文§3.1.2式(3.6)-(3.8)。"""
import torch.nn.functional as F

def representation_approximation_loss(h_star, h_hat):
    """L_approx = (1/2n)‖H*-Ĥ‖_F²。对应式(3.6)。"""
    return 0.5 * F.mse_loss(h_hat, h_star)

def task_classification_loss(logits, labels, mask):
    """L_task = CE(logits[mask], y[mask])。对应式(3.7)。"""
    return F.cross_entropy(logits[mask], labels[mask])

def combined_loss(h_star, h_hat, logits, labels, mask, alpha=0.5):
    """L = α·L_approx + (1-α)·L_task。对应式(3.8)。"""
    return alpha*representation_approximation_loss(h_star,h_hat) + (1-alpha)*task_classification_loss(logits,labels,mask)
