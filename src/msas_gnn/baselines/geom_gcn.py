"""Geom-GCN简化实现（仅补充参照）。Pei et al., ICLR 2020"""
import torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GCNConv
class GeomGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5, **kwargs):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout
    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(F.dropout(x, p=self.dropout, training=self.training), edge_index)
