from itertools import cycle
from pathlib import Path
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data
from torch_geometric.loader import (
    ClusterData,
    ClusterLoader,
    GraphSAINTNodeSampler,
    GraphSAINTRandomWalkSampler,
    NeighborLoader,
)
from torch_geometric.utils import sort_edge_index

torch.backends.cudnn.benchmark = True

H5_GROUPS = {"metr-la": "df", "pems-bay": "speed"}
LOADER_NAMES = (
    "classic",
    "full",
    "neighbor",
    "cluster",
    "graphsaint",
    "graphsaint-node",
)

_dataset_cache = {}


class TrafficDataset(Dataset):
    def __init__(self, data, seq_len, horizon):
        self.data = torch.from_numpy(np.asarray(data, dtype=np.float32))
        self.seq_len = seq_len
        self.horizon = horizon
        self.num_samples = len(self.data) - seq_len - horizon + 1
        if self.num_samples <= 0:
            raise ValueError(
                f"Pas assez de données: T={len(self.data)}, "
                f"seq_len={seq_len}, horizon={horizon}."
            )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        x_end = index + self.seq_len
        return self.data[index:x_end], self.data[x_end : x_end + self.horizon]


def load_dataset_arrays(dataset_name, dataset_path):
    key = (dataset_name.strip().lower(), str(dataset_path))
    if key in _dataset_cache:
        return _dataset_cache[key]

    name = key[0]
    if name not in H5_GROUPS:
        raise ValueError(
            f"Dataset '{name}' non supporté. "
            f"Utiliser: {', '.join(H5_GROUPS)}."
        )

    dataset_path = Path(dataset_path)
    h5_path = dataset_path / f"{name}.h5"
    adj_path = dataset_path / f"{name}-adj_mx.npy"
    if not h5_path.exists():
        raise FileNotFoundError(f"Série introuvable: {h5_path}")
    if not adj_path.exists():
        raise FileNotFoundError(f"Adjacence introuvable: {adj_path}")

    with h5py.File(h5_path, "r") as f:
        data = np.asarray(f[H5_GROUPS[name]]["block0_values"][:], np.float32)
    adj_mx = np.asarray(np.load(adj_path), dtype=np.float32)

    if data.ndim == 2:
        data = data[..., np.newaxis]
    if data.ndim != 3:
        raise ValueError(f"Série attendue (T, N, C), reçu: {data.shape}")
    if adj_mx.ndim != 2 or adj_mx.shape[0] != adj_mx.shape[1]:
        raise ValueError(f"Adjacence non carrée: {adj_mx.shape}")
    if data.shape[1] != adj_mx.shape[0]:
        raise ValueError(
            f"Série: {data.shape[1]} nœuds, adjacence: {adj_mx.shape}."
        )

    _dataset_cache[key] = (data, adj_mx)
    return data, adj_mx


def prepare_dataloaders(
    dataset_name,
    dataset_path,
    seq_len=12,
    horizon=12,
    batch_size=128,
    max_timesteps=None,
    train_ratio=0.7,
    val_ratio=0.1,
    test_ratio=0.2,
    num_workers=4,
    persistent_workers=True,
):
    data, adj_mx = load_dataset_arrays(dataset_name, dataset_path)
    original_T = len(data)
    if max_timesteps is not None:
        data = data[:max_timesteps]

    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train/val/test doivent sommer à 1.")
    train_end = int(len(data) * train_ratio)
    val_end = int(len(data) * (train_ratio + val_ratio))
    train_data, val_data, test_data = (
        data[:train_end],
        data[train_end:val_end],
        data[val_end:],
    )

    mean, std = float(train_data.mean()), float(train_data.std())
    if std < 1e-8:
        raise ValueError("Écart-type du train nul ; impossible de normaliser.")
    stats = {"mean": mean, "std": std}
    train_data = ((train_data - mean) / std).astype(np.float32, copy=False)
    val_data = ((val_data - mean) / std).astype(np.float32, copy=False)
    test_data = ((test_data - mean) / std).astype(np.float32, copy=False)

    pin = torch.cuda.is_available()
    train_ds = TrafficDataset(train_data, seq_len, horizon)
    val_ds = TrafficDataset(val_data, seq_len, horizon)
    test_ds = TrafficDataset(test_data, seq_len, horizon)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    eval_kw = dict(batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)
    val_loader = DataLoader(val_ds, **eval_kw)
    test_loader = DataLoader(test_ds, **eval_kw)

    print(f"Dataset: {dataset_name} | série: {data.shape} | adj: {adj_mx.shape}")
    if max_timesteps is not None and len(data) < original_T:
        print(f"Sous-échantillon: {len(data)}/{original_T} pas (max_timesteps).")
    print(
        f"Fenêtres train/val/test: {len(train_ds)}/{len(val_ds)}/{len(test_ds)} | "
        f"seq={seq_len} horizon={horizon}"
    )
    return train_loader, val_loader, test_loader, adj_mx, stats


def make_batches(loader_name, graph_data, shuffle=True, **kwargs):
    name = str(loader_name).strip().lower()
    if name not in LOADER_NAMES:
        raise ValueError(
            f"loader doit etre un des {list(LOADER_NAMES)}, recu: {loader_name!r}"
        )

    edge_index = graph_data.edge_index
    num_nodes = graph_data.num_nodes

    if name in ("classic", "full"):
        return cycle([(torch.arange(num_nodes), edge_index, None)])

    if name == "neighbor":
        sampler = NeighborLoader(
            graph_data,
            num_neighbors=list(kwargs.get("num_neighbors") or [10, 25]),
            batch_size=kwargs.get("neighbor_batch_size") or max(1, num_nodes),
            input_nodes=torch.arange(num_nodes),
            shuffle=shuffle,
        )
        return _stream(sampler, with_seeds=True)

    if name == "cluster":
        from_data = Data(edge_index=edge_index, num_nodes=num_nodes)
        from_data["n_id"] = torch.arange(num_nodes)
        cluster = ClusterData(
            from_data,
            num_parts=kwargs.get("num_parts", 10),
            recursive=False,
        )
        sampler = ClusterLoader(
            cluster,
            batch_size=kwargs.get("cluster_batch_size", 1),
            shuffle=shuffle,
        )
        return _stream(sampler)

    saint = Data(edge_index=edge_index, num_nodes=num_nodes)
    saint["n_id"] = torch.arange(num_nodes)
    if name == "graphsaint":
        sampler = GraphSAINTRandomWalkSampler(
            saint,
            batch_size=(kwargs.get("batch_size", 128)),
            walk_length=kwargs.get("walk_length", 1),
            num_steps=kwargs.get("num_steps", 1),
        )
    else:
        sampler = GraphSAINTNodeSampler(
            saint,
            batch_size=kwargs.get("batch_size", 256),
            num_steps=kwargs.get("num_steps", 8),
        )
    return _stream(sampler)


def _stream(sampler, with_seeds=False):
    def make_iter():
        for sub in sampler:
            n_seed = sub.batch_size if with_seeds else None
            yield sub.n_id, sub.edge_index, n_seed

    while True:
        yield from make_iter()


class GraphBatchLoader:
    def __init__(self, loader, loader_name="classic", graph_data=None, **kwargs):
        self.dataset = loader.dataset
        self.batch_size = loader.batch_size
        self.num_nodes = graph_data.num_nodes if graph_data is not None else None
        self._loader = loader
        self._name = str(loader_name).strip().lower()
        self._graph_data = graph_data
        self._kwargs = dict(kwargs)

    def __len__(self):
        return len(self._loader)

    def __iter__(self):
        stream = make_batches(self._name, self._graph_data, **self._kwargs)
        for X, Y in self._loader:
            n_id, edge_index, n_seed = next(stream)
            data = Data(
                x=X[:, :, n_id, :],
                y=Y[:, :, n_id, :],
                edge_index=edge_index,
                n_id=n_id,
            )
            if n_seed is not None:
                data.seed_mask = torch.zeros(n_id.numel(), dtype=torch.bool)
                data.seed_mask[:n_seed] = True
            yield data


_BASE_DEFAULTS = dict(
    input_dim=1,
    output_dim=1,
    hidden_dim=64,
    seq_len=12,
    horizon=12,
    batch_size=128,
    node_dim=16,
    dropout=0.1,
    lr=1e-3,
    weight_decay=1e-4,
    epochs=50,
    patience=10,
    neighbor_sizes=None,
    neighbor_batch_size=256,
    loader="classic",
    loader_kwargs=None,
    eta_min=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
)


def attach_model_config(model, adj_mx, num_nodes, kw):
    cfg = dict(_BASE_DEFAULTS)
    cfg.update(kw)

    loader = str(cfg["loader"]).strip().lower()
    if loader not in LOADER_NAMES:
        raise ValueError(
            f"loader doit etre un des {list(LOADER_NAMES)}, recu: {loader!r}"
        )

    model.num_nodes = num_nodes
    model.input_dim = cfg["input_dim"]
    model.output_dim = cfg["output_dim"]
    model.hidden_dim = cfg["hidden_dim"]
    model.seq_len = cfg["seq_len"]
    model.horizon = cfg["horizon"]
    model.batch_size = cfg["batch_size"]
    model.lr = cfg["lr"]
    model.weight_decay = cfg["weight_decay"]
    model.epochs = cfg["epochs"]
    model.patience = cfg["patience"]
    model.eta_min = cfg["eta_min"]
    model.device = cfg["device"]
    model.loader = loader
    model.loader_kwargs = dict(cfg["loader_kwargs"] or {})
    model.loader_kwargs.setdefault(
        "num_neighbors",
        list(cfg["neighbor_sizes"]) if cfg["neighbor_sizes"] is not None else None,
    )
    model.loader_kwargs.setdefault("neighbor_batch_size", cfg["neighbor_batch_size"])

    model.register_buffer(
        "static_adj", torch.tensor(np.asarray(adj_mx, dtype=np.float32))
    )
    model.use_learned_adj = getattr(model, "use_learned_adj", True)
    if model.use_learned_adj:
        model.node_emb1 = nn.Parameter(torch.randn(num_nodes, cfg["node_dim"]))
        model.node_emb2 = nn.Parameter(torch.randn(cfg["node_dim"], num_nodes))
    model.dropout = nn.Dropout(cfg["dropout"])
    model.loss_fn = nn.MSELoss()

    model.optimizer = None
    model.scheduler = None
    model.requires_training = True
    model.to(model.device)
    return model


def flatten_nodes(x):
    B, T, N, C = x.shape
    return x.permute(0, 2, 1, 3).reshape(B * N, T * C)


def to_forecast(h, B, N, horizon, output_dim):
    return (
        h.reshape(B, N, horizon, output_dim)
        .permute(0, 2, 1, 3)
        .contiguous()
    )


def adaptive_adj(model):
    static = model.static_adj / (model.static_adj.sum(dim=1, keepdim=True) + 1e-8)
    if not model.use_learned_adj:
        return static
    learned = F.softmax(F.relu(model.node_emb1 @ model.node_emb2), dim=1)
    return 0.5 * static + 0.5 * learned


def to_edge_index(model):
    adj = adaptive_adj(model).detach()
    pairs = adj.nonzero(as_tuple=False).t().contiguous()
    return sort_edge_index(pairs, sort_by_row=False)


def batch_edge_index(model, edge_index, x):
    edge_index = edge_index.to(model.device, non_blocking=True)
    B, T, N = x.shape[0], x.shape[1], x.shape[2]
    blocks = B * T if getattr(model, "_graph_per_timestep", False) else B
    if blocks <= 1:
        return edge_index
    offsets = torch.arange(blocks, device=edge_index.device).repeat_interleave(
        edge_index.size(1)
    ) * N
    return edge_index.repeat(1, blocks) + offsets


def select_seeds(out, target, seed_mask):
    if seed_mask is None:
        return out, target
    mask = seed_mask.to(target.device)
    return out[:, :, mask], target[:, :, mask]


def fit(model, train_loader, val_loader=None, callback=None):
    if not getattr(model, "requires_training", True):
        model.train_losses, model.val_losses = [], []
        return model.train_losses, model.val_losses

    model.optimizer = torch.optim.AdamW(
        model.parameters(), lr=model.lr, weight_decay=model.weight_decay
    )
    if model.eta_min is not None:
        model.scheduler = CosineAnnealingLR(
            model.optimizer, T_max=model.epochs, eta_min=model.eta_min
        )
    best_val, patience_count, best_state = float("inf"), 0, None
    model.train_losses, model.val_losses = [], []

    for epoch in range(model.epochs):
        model.train()
        total_loss, total_steps = torch.tensor(0.0, device=model.device), 0
        for batch in train_loader:
            model.optimizer.zero_grad(set_to_none=True)
            pred, target = model.forward(batch)
            loss = model.loss_fn(pred, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            model.optimizer.step()
            total_loss += loss.detach()
            total_steps += 1

        train_loss = total_loss.item() / max(total_steps, 1)
        model.train_losses.append(train_loss)

        if val_loader is None:
            print(f"Epoch {epoch+1:03d}/{model.epochs} | Train: {train_loss:.4f}")
            continue

        model.eval()
        total_loss, total_steps = torch.tensor(0.0, device=model.device), 0
        rng_state = torch.random.get_rng_state()
        torch.manual_seed(0)
        with torch.inference_mode():
            for batch in val_loader:
                pred, target = model.forward(batch)
                total_loss += model.loss_fn(pred, target)
                total_steps += 1
        torch.random.set_rng_state(rng_state)
        val_loss = total_loss.item() / max(total_steps, 1)

        model.val_losses.append(val_loss)
        if callback is not None:
            callback(epoch, train_loss, val_loss)
        print(
            f"Epoch {epoch+1:03d}/{model.epochs} | "
            f"Train: {train_loss:.4f} | Val: {val_loss:.4f}"
        )

        if model.scheduler is not None:
            if isinstance(model.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                model.scheduler.step(val_loss)
            else:
                model.scheduler.step()

        if val_loss < best_val:
            best_val, patience_count = val_loss, 0
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
        else:
            patience_count += 1
            if patience_count >= model.patience:
                print(f"Early stopping a l'epoque {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model.train_losses, model.val_losses


def test(model, loader):
    model.eval()
    total_loss, total_steps = torch.tensor(0.0, device=model.device), 0
    rng_state = torch.random.get_rng_state()
    torch.manual_seed(0)
    xs, ys, preds = [], [], []
    with torch.inference_mode():
        for batch in loader:
            seed_mask = getattr(batch, "seed_mask", None)
            pred, target = model.forward(batch)
            total_loss += model.loss_fn(pred, target)
            total_steps += 1
            if seed_mask is not None:
                xs.append(batch.x[:, :, seed_mask, :].cpu())
            else:
                xs.append(batch.x.cpu())
            ys.append(target.cpu())
            preds.append(pred.cpu())
    torch.random.set_rng_state(rng_state)

    mse = total_loss.item() / max(total_steps, 1)
    return (
        mse,
        torch.cat(xs, dim=0),
        torch.cat(ys, dim=0),
        torch.cat(preds, dim=0),
    )


def prepare_model_dataloaders(
    model,
    dataset_name,
    dataset_path,
    seq_len=None,
    horizon=None,
    batch_size=None,
    max_timesteps=None,
    train_ratio=0.7,
    val_ratio=0.1,
    test_ratio=0.2,
    neighbor_sizes=None,
    neighbor_batch_size=None,
    num_workers=None,
    persistent_workers=None,
):
    seq_len = model.seq_len if seq_len is None else seq_len
    horizon = model.horizon if horizon is None else horizon
    batch_size = model.batch_size if batch_size is None else batch_size

    if neighbor_sizes is not None:
        model.loader_kwargs["num_neighbors"] = list(neighbor_sizes)
    elif model.loader == "neighbor" and model.loader_kwargs.get("num_neighbors") is None:
        degrees = model.static_adj.detach().cpu().numpy().sum(axis=1)
        mean_degree = float(np.mean(degrees))
        q90 = int(np.percentile(degrees, 90))
        base = max(1, int(mean_degree) - 1)
        n_layers = getattr(model, "num_layers", 2)
        sizes = [base] * max(n_layers, 1)
        sizes[-1] = max(1, q90)
        model.loader_kwargs["num_neighbors"] = sizes
    if neighbor_batch_size is not None:
        model.loader_kwargs["neighbor_batch_size"] = neighbor_batch_size

    nn_ = model.loader_kwargs.get("num_neighbors")
    num_layers_model = getattr(model, "num_layers", None)
    if nn_ is not None and num_layers_model is not None and len(nn_) < num_layers_model:
        model.loader_kwargs["num_neighbors"] = nn_ + [nn_[-1]] * (
            num_layers_model - len(nn_)
        )

    num_workers = 4 if num_workers is None else num_workers
    if num_workers == 0:
        persistent_workers = False

    train_loader, val_loader, test_loader, adj_mx, stats = prepare_dataloaders(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        seq_len=seq_len,
        horizon=horizon,
        batch_size=batch_size,
        max_timesteps=max_timesteps,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        num_workers=num_workers,
        persistent_workers=True if persistent_workers is None else persistent_workers,
    )

    graph_data = Data(
        edge_index=to_edge_index(model).cpu(), num_nodes=model.num_nodes
    )

    return (
        GraphBatchLoader(
            train_loader, model.loader, graph_data,
            batch_size=batch_size, shuffle=True, **model.loader_kwargs,
        ),
        GraphBatchLoader(
            val_loader, model.loader, graph_data,
            batch_size=batch_size, shuffle=True, **model.loader_kwargs,
        ),
        GraphBatchLoader(
            test_loader, "classic", graph_data,
            batch_size=batch_size, shuffle=False, **model.loader_kwargs,
        ),
        adj_mx,
        stats,
    )


def train_and_evaluate(
    model,
    dataset_name,
    dataset_path,
    name,
    seq_len=None,
    horizon=None,
    batch_size=128,
    max_timesteps=None,
    eval_split="test",
    compile=False,
    fit_callback=None,
    num_workers=None,
    persistent_workers=None,
):
    if eval_split not in ("val", "test"):
        raise ValueError(f"eval_split doit etre 'val' ou 'test', recu: {eval_split!r}")

    if compile:
        model = torch.compile(model)

    train_loader, val_loader, test_loader, _, stats = prepare_model_dataloaders(
        model,
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        seq_len=seq_len,
        horizon=horizon,
        batch_size=batch_size,
        max_timesteps=max_timesteps,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )
    loader = test_loader

    start = time.perf_counter()
    train_loss, val_loss = fit(model, train_loader, val_loader, callback=fit_callback)
    fit_time = time.perf_counter() - start

    mse, x, y_true, y_pred = test(model, loader)
    y_pred = y_pred.numpy() * stats["std"] + stats["mean"]
    y_true = y_true.numpy() * stats["std"] + stats["mean"]
    x = x.numpy() * stats["std"] + stats["mean"]

    valid = y_true > 0.5
    errors = y_pred - y_true
    err_valid = errors[valid]

    return {
        "name": name,
        "mse": float(mse),
        "mae": float(np.mean(np.abs(err_valid))) if err_valid.size else float("nan"),
        "mape": float(
            np.mean(np.abs(errors[valid]) / (np.abs(y_true[valid]) + 1e-8)) * 100
        )
        if err_valid.size
        else float("nan"),
        "rmse": float(np.sqrt(np.mean(err_valid**2))) if err_valid.size else float("nan"),
        "train time": fit_time,
        "epochs": len(train_loss),
        "errors": errors.reshape(-1),
        "errors_timed": np.mean(np.abs(errors), axis=(0, 2)).squeeze(-1),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "y_pred": y_pred,
        "y_true": y_true,
        "x": x,
    }