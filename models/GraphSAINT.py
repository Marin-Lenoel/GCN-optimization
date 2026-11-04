import torch
import torch.nn as nn

from torch_geometric.nn import GraphConv
from torch_geometric.nn.models.basic_gnn import BasicGNN

from shared.loaders import (
    _BASE_DEFAULTS,
    attach_model_config,
    batch_edge_index,
    select_seeds,
    to_forecast,
)


class GraphSAINT(BasicGNN):
    use_learned_adj = False

    def __init__(
        self,
        adj_mx,
        num_nodes,
        num_blocks=1,
        walk_length=2,
        num_steps=8,
        lastlayer="NONE",
        **kw,
    ):
        kw.setdefault("lr", 5e-3)
        kw.setdefault("weight_decay", 1e-5)
        kw.setdefault("eta_min", 1e-6)
        kw.setdefault("loader", "graphsaint")
        kw["loader_kwargs"] = {
            **(kw.get("loader_kwargs") or {}),
            "walk_length": walk_length,
            "num_steps": num_steps,
        }

        cfg = dict(_BASE_DEFAULTS)
        cfg.update(kw)

        self.num_nodes = num_nodes
        self.input_dim = cfg["input_dim"]
        self.seq_len = cfg["seq_len"]
        self.hidden_dim = cfg["hidden_dim"]
        self.horizon = cfg["horizon"]
        self.output_dim = cfg["output_dim"]

        self.lastlayer_type = str(lastlayer).upper()
        out_size = self.output_dim * self.horizon

        super().__init__(
            in_channels=self.input_dim,
            hidden_channels=self.hidden_dim,
            num_layers=num_blocks,
            out_channels=self.hidden_dim,
            dropout=cfg["dropout"],
        )

        attach_model_config(self, adj_mx, num_nodes, cfg)

        self.supports_edge_weight = False
        self.supports_edge_attr = False

        self._graph_per_timestep = True

        if self.lastlayer_type == "GRU":
            self.temporal_layer = nn.GRU(
                input_size=self.hidden_dim,
                hidden_size=self.hidden_dim,
                num_layers=1,
                batch_first=True,
            )
            self.h0 = nn.Parameter(
                torch.zeros(1, 1, self.hidden_dim)
            )
            nn.init.orthogonal_(self.h0)

        elif self.lastlayer_type == "LSTM":
            self.temporal_layer = nn.LSTM(
                input_size=self.hidden_dim,
                hidden_size=self.hidden_dim,
                num_layers=1,
                batch_first=True,
            )
            self.h0 = nn.Parameter(
                torch.zeros(1, 1, self.hidden_dim)
            )
            self.c0 = nn.Parameter(
                torch.zeros(1, 1, self.hidden_dim)
            )
            nn.init.orthogonal_(self.h0)
            nn.init.orthogonal_(self.c0)

        elif self.lastlayer_type in {"LINEAR", "NONE"}:
            self.temporal_layer = nn.Identity()

        else:
            raise ValueError(
                f"Unknown lastlayer: {lastlayer}. "
                "Use 'GRU', 'LSTM', 'LINEAR' or 'NONE'."
            )

        self.to_forecast = nn.Linear(
            self.hidden_dim,
            out_size,
        )

        self.to(self.device)

    def init_conv(self, in_channels, out_channels, **kwargs):
        return GraphConv(
            in_channels,
            out_channels,
            aggr="add",
        )

    def forward(self, batch):
        x = batch.x.to(self.device, non_blocking=True).float()
        y = batch.y.to(self.device, non_blocking=True).float()
        edge_index = batch_edge_index(self, batch.edge_index, x)


        B, T, N, F = x.shape


        h = x.reshape(B * T * N, F)
        spatial = super().forward(h, edge_index)
        spatial = spatial.reshape(B, T, N, self.hidden_dim)

        temporal = spatial.permute(0, 2, 1, 3).contiguous()
        temporal = temporal.reshape(B * N, T, self.hidden_dim)

        if self.lastlayer_type == "GRU":
            h0 = self.h0.expand(
                1, B * N, self.hidden_dim
            ).contiguous()
            temporal_output, _ = self.temporal_layer(
                temporal,
                h0,
            )
            out = temporal_output[:, -1, :]

        elif self.lastlayer_type == "LSTM":
            h0 = self.h0.expand(
                1, B * N, self.hidden_dim
            ).contiguous()
            c0 = self.c0.expand(
                1, B * N, self.hidden_dim
            ).contiguous()
            temporal_output, _ = self.temporal_layer(
                temporal,
                (h0, c0),
            )
            out = temporal_output[:, -1, :]

        elif self.lastlayer_type in {"LINEAR", "NONE"}:
            out = temporal[:, -1, :]


        out = self.to_forecast(out)

        pred = to_forecast(
            out,
            B,
            N,
            self.horizon,
            self.output_dim,
        )

        return select_seeds(pred, y, getattr(batch, "seed_mask", None))