"""GraphSAGE 基线。用于对齐原始 SDGNN 论文中的目标 GNN 设定。"""

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5, num_layers=2, **kwargs):
        super().__init__()
        num_layers = max(int(num_layers), 2)
        dims = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        self.convs = nn.ModuleList(
            [SAGEConv(dims[i], dims[i + 1]) for i in range(num_layers)]
        )
        self.dropout = float(dropout)

    def get_embedding(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)

    def forward(self, x, edge_index):
        return self.get_embedding(x, edge_index)
