# Extended code (beyond the MTGNN example)

This file briefly describes the code added after the original one-experiment
script explained in `README.md`. The original MTGNN example is untouched; these
additions extend the project into a multi-model comparison framework.

## New files and modules

```text
for-mahin/
|-- degree_dist.py            # quick graph-degree exploration (boxplot)
|-- optimize.py               # Optuna hyperparameter search for all models
|-- models/
|   |-- ClusterGCN.py         # new
|   |-- GraphSAGE.py          # new
|   |-- GraphSAINT.py         # new
|   |-- RepeatLast.py         # new (baseline)
|-- results/                  # Optuna databases + best_params_*.json
`-- shared/loaders.py         # heavily extended (batching strategies, train/test helpers)
```

## `models/` — additional models

`models/__init__.py` exposes a `MODELS` registry (name -> class), so the
optimizer and any future script can iterate over:

- **`MTGNN`** — the original simplified model from `README.md`.
- **`GraphSAGE`** (`GraphSAGE`, `GraphSAGE-gru`, `GraphSAGE-lstm`) —
  a SAGE-style GNN operating on the traffic graph. The `-gru` / `-lstm`
  variants add a recurrent readout on the final layer. Trained with the
  **neighbor** sampling loader.
- **`GraphSAINT`** (`GraphSAINT`, `GraphSAINT-gru`, `GraphSAINT-lstm`) —
  subgraph-sampling GNN (random-walk sampling per batch), again with optional
  recurrent last layer. Uses the **graphsaint** loader.
- **`ClusterGCN`** — GCN trained with **mean-variant cluster loss** on
  graph partitions (k-means subgraphs). Uses the **cluster** loader.
- **`RepeatLast`** — trivial baseline that predicts the last observed value
  repeated over the horizon. Also useful to sanity-check real models.

## `shared/loaders.py` — extended data pipeline

Beyond the classic window loader, it now supports several batching strategies
via `make_batches` / `prepare_dataloaders`:

```text
classic        sliding windows over the time series (as before)
neighbor       torch_geometric NeighborLoader (GraphSAGE)
cluster        ClusterData + ClusterLoader (ClusterGCN)
graphsaint     GraphSAINTRandomWalkSampler (GraphSAINT)
```

It also provides shared training utilities reused by all models:

- `fit(model, train_loader, val_loader, callback)` — training loop with an
  optional callback for early stopping / pruning.
- `test(model, loader)` — evaluation, returns MAE / MSE.
- `train_and_evaluate(...)` — one-call helper: builds loaders, trains, tests
  and returns a dict containing metrics and arrays (`mse`, `mae`, outputs).
- `load_dataset_arrays(...)` — cached dataset loading (series + adjacency).

The matching loader name is passed per model by `optimize.py` via a `loader`
parameter, so each model gets an appropriate batching scheme.

## `optimize.py` — Optuna hyperparameter search

Searches hyperparameters for any subset of the models:

```bash
python3 optimize.py --dataset metr-la --n-trials 30 --epochs 15
# optional: --models MTGNN GraphSAGE-gru  --max-timesteps 2000
```

Key points:

- For each model, `suggest_params(trial, model_name)` defines a **model-specific
  search space** (e.g. `num_blocks`, `kernel_size`, `dilation_base` for MTGNN;
  `neighbor_sizes`, `cluster_batch_size`, `num_clusters` for the others).
- Each trial calls `train_and_evaluate` on the **validation split** and
  reports val MSE; MBM-style pruning via `trial.report(val_loss, epoch)` +
  `trial.should_prune()` kills bad trials early. CUDA OOM errors prune the
  trial instead of crashing the study.
- Results are stored in a persistent Optuna SQLite database
  (`results/optuna_<dataset>.db`), so a study can be resumed with
  `load_if_exists=True`.
- Best values/params per model are appended to
  `results/best_params_<dataset>.json` (with a header of run settings).

## `degree_dist.py` — quick graph exploration

```bash
python3 degree_dist.py --dataset metr-la
```

Loads the adjacency matrix and displays a single boxplot of the degree
distribution with the Q1/Q2/Q3 quartile values annotated, to help understand
the graph structure before training. Purely visual, no model involved.

## `results/` — outputs

Not committed code, but produced by `optimize.py`:

- `optuna_<dataset>.db` — persistent Optuna study storage.
- `best_params_<dataset>.json` — best validation MSE + best hyperparameters
  per model, ready to reuse in later training runs.

## Typical workflow

1. `python3 degree_dist.py` — look at the graph;
2. `python3 optimize.py --dataset metr-la` — find good hyperparameters;
3. reuse the values in `results/best_params_metr-la.json` in a training run of
   the chosen model (`main.py` still runs the original MTGNN example).
