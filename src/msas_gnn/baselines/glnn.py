"""GLNN（无图推理MLP+知识蒸馏）。Zhang et al., NeurIPS 2022"""
import torch.nn as nn, torch.nn.functional as F
class GLNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5, **kwargs):
        super().__init__()
        self.dropout = dropout
        layers = [nn.Linear(in_channels, hidden_channels), nn.ReLU()]
        for _ in range(num_layers-2): layers += [nn.Linear(hidden_channels, hidden_channels), nn.ReLU()]
        layers.append(nn.Linear(hidden_channels, out_channels))
        self.mlp = nn.Sequential(*layers)
    def forward(self, x, edge_index=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.mlp(x)
