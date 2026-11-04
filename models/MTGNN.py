import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.loaders import (
    adaptive_adj,
    attach_model_config,
    select_seeds,
)


class MixProp(nn.Module):
    def __init__(self, channels, gcn_depth=2, alpha=0.05, dropout=0.0):
        super().__init__()
        self.gcn_depth = gcn_depth
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Conv2d((gcn_depth + 1) * channels, channels, kernel_size=1)

    def forward(self, x, adj):
        adj = adj / (adj.sum(dim=1, keepdim=True) + 1e-8)
        states = [x]
        h = x
        for _ in range(self.gcn_depth):
            h = self.alpha * x + (1 - self.alpha) * torch.einsum(
                "ij,bcjt->bcit", adj, h
            )
            states.append(h)
        return self.mlp(self.dropout(torch.cat(states, dim=1)))


class TemporalGatedConv(nn.Module):
    def __init__(self, channels, kernel_size=2, dilation=1, dropout=0.0):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        conv = dict(
            kernel_size=(1, kernel_size),
            dilation=(1, dilation),
            padding=(0, pad),
        )
        self.filter_conv = nn.Conv2d(channels, channels, **conv)
        self.gate_conv = nn.Conv2d(channels, channels, **conv)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = torch.tanh(self.filter_conv(x)) * torch.sigmoid(self.gate_conv(x))
        if h.size(-1) > x.size(-1):
            h = h[..., -x.size(-1):]
        return self.dropout(h)


class MTGNNBlock(nn.Module):
    def __init__(
        self, channels, kernel_size=2, dilation=1, gcn_depth=2,
        propalpha=0.05, dropout=0.0,
    ):
        super().__init__()
        self.temporal = TemporalGatedConv(channels, kernel_size, dilation, dropout)
        self.mixprop = MixProp(channels, gcn_depth, propalpha, dropout)
        self.residual = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x, adj):
        h = self.norm(self.mixprop(self.temporal(x), adj))
        return F.relu(h + self.residual(x))


class MTGNN(nn.Module):
    def __init__(
        self,
        adj_mx,
        num_nodes,
        num_blocks=3,
        kernel_size=2,
        dilation_base=2,
        gcn_depth=2,
        propalpha=0.05,
        **kw,
    ):
        super().__init__()
        kw.setdefault("loader", "classic")
        attach_model_config(self, adj_mx, num_nodes, kw)

        self.input_proj = nn.Conv2d(self.input_dim, self.hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList(
            MTGNNBlock(
                self.hidden_dim,
                kernel_size,
                dilation_base ** i,
                gcn_depth,
                propalpha,
                self.dropout.p,
            )
            for i in range(num_blocks)
        )
        self.end_proj1 = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1)
        self.end_proj2 = nn.Conv2d(
            self.hidden_dim, self.horizon * self.output_dim, kernel_size=1
        )
        self.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=5, factor=0.5
        )

    def forward(self, batch):
        x = batch.x.to(self.device, non_blocking=True).float()
        y = batch.y.to(self.device, non_blocking=True).float()

        h = self.input_proj(x.permute(0, 3, 2, 1))
        adj = adaptive_adj(self)
        for block in self.blocks:
            h = block(h, adj)

        h = self.end_proj2(self.dropout(F.relu(self.end_proj1(h[..., -1:]))))
        B = h.size(0)
        h = h.squeeze(-1).view(B, self.horizon, self.output_dim, self.num_nodes)
        pred = h.permute(0, 1, 3, 2).contiguous()
        return select_seeds(pred, y, getattr(batch, "seed_mask", None))