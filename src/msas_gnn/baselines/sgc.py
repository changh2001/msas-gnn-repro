"""SGC。Wu et al., ICML 2019"""
import torch.nn as nn
from torch_geometric.nn import SGConv
class SGC(nn.Module):
    def __init__(self, in_channels, out_channels, K=2, **kwargs):
        super().__init__()
        self.conv = SGConv(in_channels, out_channels, K=K, cached=True)
    def forward(self, x, edge_index): return self.conv(x, edge_index)
