"""PPRGo。Bojchevski et al., KDD 2020"""
import torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import APPNP
class PPRGo(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, K=10, alpha=0.1, dropout=0.5, **kwargs):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.prop = APPNP(K=K, alpha=alpha); self.dropout = dropout
    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.prop(self.lin2(x), edge_index)
