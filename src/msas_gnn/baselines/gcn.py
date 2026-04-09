"""标准GCN（2层），同时作为教师模型。Kipf & Welling, ICLR 2017"""
import torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5, num_layers=2):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers-2): self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index)); x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)
    def get_embedding(self, x, edge_index):
        for conv in self.convs[:-1]: x = F.relu(conv(x, edge_index))
        return x
