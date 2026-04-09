"""H2GCN简化实现（仅补充参照）。Zhu et al., NeurIPS 2020"""
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GCNConv
class H2GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, K=2, dropout=0.5, **kwargs):
        super().__init__()
        self.K = K; self.dropout = dropout
        self.ego_lin = nn.Linear(in_channels, hidden_channels)
        self.neighbor_convs = nn.ModuleList([GCNConv(in_channels, hidden_channels) for _ in range(K)])
        self.classifier = nn.Linear(hidden_channels*(K+1), out_channels)
    def forward(self, x, edge_index):
        parts = [F.relu(self.ego_lin(x))]
        h = x
        for conv in self.neighbor_convs:
            h = F.relu(conv(h, edge_index)); parts.append(h)
        out = torch.cat(parts, dim=-1)
        return self.classifier(F.dropout(out, p=self.dropout, training=self.training))
