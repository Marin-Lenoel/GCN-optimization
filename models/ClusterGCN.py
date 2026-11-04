from torch_geometric.nn import ClusterGCNConv
from torch_geometric.nn.models.basic_gnn import BasicGNN

from shared.loaders import (
    _BASE_DEFAULTS,
    attach_model_config,
    batch_edge_index,
    flatten_nodes,
    select_seeds,
    to_forecast,
)


class ClusterGCN(BasicGNN):
    use_learned_adj = False

    def __init__(
        self,
        adj_mx,
        num_nodes,
        num_blocks=2,
        num_clusters=10,
        cluster_batch_size=1,
        diag_lambda=0.0,
        **kw,
    ):
        kw.setdefault("lr", 5e-3)
        kw.setdefault("weight_decay", 1e-5)
        kw.setdefault("eta_min", 1e-6)
        kw.setdefault("loader", "cluster")
        kw["loader_kwargs"] = {
            **(kw.get("loader_kwargs") or {}),
            "num_parts": num_clusters,
            "cluster_batch_size": cluster_batch_size,
        }

        cfg = dict(_BASE_DEFAULTS)
        cfg.update(kw)

        in_size = cfg["input_dim"] * cfg["seq_len"]
        out_size = cfg["output_dim"] * cfg["horizon"]

        self._diag_lambda = diag_lambda

        super().__init__(
            in_channels=in_size,
            hidden_channels=cfg["hidden_dim"],
            num_layers=num_blocks + 1,
            out_channels=out_size,
            dropout=cfg["dropout"],
        )

        attach_model_config(self, adj_mx, num_nodes, cfg)
        self.supports_edge_weight = False
        self.supports_edge_attr = False

    def init_conv(self, in_channels, out_channels, **kwargs):
        return ClusterGCNConv(
            in_channels, out_channels, diag_lambda=self._diag_lambda
        )

    def forward(self, batch):
        x = batch.x.to(self.device, non_blocking=True).float()
        y = batch.y.to(self.device, non_blocking=True).float()
        edge_index = batch_edge_index(self, batch.edge_index, x)

        h = flatten_nodes(x)
        out = super().forward(h, edge_index)
        pred = to_forecast(
            out, x.shape[0], x.shape[2], self.horizon, self.output_dim
        )
        return select_seeds(pred, y, getattr(batch, "seed_mask", None))