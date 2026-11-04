import torch.nn as nn

from shared.loaders import attach_model_config, select_seeds


class RepeatLast(nn.Module):
    use_learned_adj = False

    def __init__(self, adj_mx, num_nodes, **kw):
        super().__init__()
        kw.setdefault("epochs", 1)
        kw.setdefault("patience", 1)
        kw.setdefault("loader", "classic")
        attach_model_config(self, adj_mx, num_nodes, kw)
        self.requires_training = False

    def forward(self, batch):
        x = batch.x.to(self.device, non_blocking=True).float()
        y = batch.y.to(self.device, non_blocking=True).float()
        pred = x[:, -1:, :, : self.output_dim].repeat(1, self.horizon, 1, 1)
        return select_seeds(pred, y, getattr(batch, "seed_mask", None))