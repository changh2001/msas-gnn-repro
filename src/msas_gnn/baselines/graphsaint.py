"""GraphSAINT-style self-contained baseline.

GraphSAINT is primarily a sampler/training protocol. For this repository's
controlled baseline interface we use a GraphSAGE-style encoder that can be
trained by the existing mini-batch loaders and is reported as the local
GraphSAINT reproduction protocol.
"""
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAINT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5, **kwargs):
        super().__init__()
        self.dropout = float(dropout)
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(max(int(num_layers) - 2, 0)):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)

    def get_embedding(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
        return x
