"""Self-contained recent graph Transformer style baselines.

These implementations are lightweight local reproductions for the thesis
comparison pipeline. They preserve each method family's main computation style
while sharing the repository's standard training and evaluation protocol.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import APPNP, GCNConv, SGConv


class LinearGlobalAttention(nn.Module):
    def __init__(self, hidden_channels: int, dropout: float = 0.0):
        super().__init__()
        self.q = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.k = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.v = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.out = nn.Linear(hidden_channels, hidden_channels)
        self.dropout = float(dropout)

    def forward(self, x):
        q = F.elu(self.q(x)) + 1.0
        k = F.elu(self.k(x)) + 1.0
        v = self.v(x)
        kv = k.t() @ v
        z = 1.0 / (q @ k.sum(dim=0).clamp_min(1e-6))
        h = (q @ kv) * z.unsqueeze(-1)
        h = self.out(h)
        return F.dropout(h, p=self.dropout, training=self.training)


class NodeFormer(nn.Module):
    """Kernelized global interaction plus a lightweight structural residual."""

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5, **kwargs):
        super().__init__()
        self.dropout = float(dropout)
        self.input = nn.Linear(in_channels, hidden_channels)
        self.attn_layers = nn.ModuleList(
            [LinearGlobalAttention(hidden_channels, dropout=dropout) for _ in range(max(int(num_layers), 1))]
        )
        self.struct = GCNConv(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.input(x))
        for attn in self.attn_layers:
            x = F.layer_norm(x + attn(x), x.shape[-1:])
        x = F.relu(self.struct(x, edge_index) + x)
        return self.classifier(F.dropout(x, p=self.dropout, training=self.training))


class DIFFormer(nn.Module):
    """Diffusion-induced Transformer approximation."""

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5, K=10, alpha=0.1, **kwargs):
        super().__init__()
        self.dropout = float(dropout)
        self.input = nn.Linear(in_channels, hidden_channels)
        self.attn = LinearGlobalAttention(hidden_channels, dropout=dropout)
        self.diffusion = APPNP(K=int(K), alpha=float(alpha))
        self.layers = int(max(num_layers, 1))
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.input(x))
        for _ in range(self.layers):
            global_h = self.attn(x)
            diff_h = self.diffusion(x, edge_index)
            x = F.layer_norm(x + global_h + diff_h, x.shape[-1:])
        return self.classifier(F.dropout(x, p=self.dropout, training=self.training))


class SGFormer(nn.Module):
    """Simplified graph Transformer: linear global branch + graph branch."""

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5, **kwargs):
        super().__init__()
        self.dropout = float(dropout)
        self.input = nn.Linear(in_channels, hidden_channels)
        self.global_layers = nn.ModuleList(
            [LinearGlobalAttention(hidden_channels, dropout=dropout) for _ in range(max(int(num_layers), 1))]
        )
        self.graph_branch = GCNConv(hidden_channels, hidden_channels)
        self.mix = nn.Linear(hidden_channels * 2, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.input(x))
        h_global = x
        for layer in self.global_layers:
            h_global = F.layer_norm(h_global + layer(h_global), h_global.shape[-1:])
        h_graph = F.relu(self.graph_branch(x, edge_index))
        h = F.relu(self.mix(torch.cat([h_global, h_graph], dim=-1)))
        return self.classifier(F.dropout(h, p=self.dropout, training=self.training))


class NAGphormer(nn.Module):
    """Hop2Token-style local reproduction with a short token Transformer."""

    def __init__(self, in_channels, hidden_channels, out_channels, K=3, dropout=0.5, num_layers=1, **kwargs):
        super().__init__()
        self.dropout = float(dropout)
        self.K = int(K)
        self.input = nn.Linear(in_channels, hidden_channels)
        self.hop_props = nn.ModuleList(
            [SGConv(hidden_channels, hidden_channels, K=hop, cached=False) for hop in range(1, self.K + 1)]
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_channels,
            nhead=1,
            dim_feedforward=hidden_channels * 2,
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(int(num_layers), 1))
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        h0 = F.relu(self.input(x))
        tokens = [h0]
        for prop in self.hop_props:
            tokens.append(F.relu(prop(h0, edge_index)))
        seq = torch.stack(tokens, dim=1)
        seq = self.encoder(seq)
        h = seq.mean(dim=1)
        return self.classifier(F.dropout(h, p=self.dropout, training=self.training))
