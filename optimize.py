import argparse
import gc
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import optuna
import torch

from models import MODELS
from shared.loaders import load_dataset_arrays, train_and_evaluate

optuna.logging.set_verbosity(optuna.logging.WARNING)

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def suggest_params(trial, model_name, max_neighbors=None):
    if model_name == "RepeatLast":
        return {}
    params = {
        "hidden_dim": trial.suggest_categorical("hidden_dim", [8, 16, 32, 64]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "node_dim": trial.suggest_categorical("node_dim", [8, 16, 32]),
        "patience": trial.suggest_int("patience", 2, 8),
    }
    if model_name == "MTGNN":
        params.update(
            num_blocks=trial.suggest_int("num_blocks", 0, 3),
            kernel_size=trial.suggest_categorical("kernel_size", [2, 3]),
            dilation_base=trial.suggest_categorical("dilation_base", [2, 3]),
            gcn_depth=trial.suggest_int("gcn_depth", 1, 3),
            propalpha=trial.suggest_float("propalpha", 0.01, 0.1, log=True),
        )
        params["loader"] = "classic"
    elif model_name == "ClusterGCN":
        params.update(
            num_blocks=trial.suggest_int("num_blocks", 0, 3),
            num_clusters=trial.suggest_categorical("num_clusters", [5, 10, 15, 20]),
            cluster_batch_size=trial.suggest_categorical(
                "cluster_batch_size", [1, 2, 3]
            ),
            diag_lambda=trial.suggest_float("diag_lambda", 0.0, 2.0),
            eta_min=trial.suggest_float("eta_min", 1e-7, 1e-4, log=True),
        )
        params["loader"] = "cluster"
    elif model_name.startswith("GraphSAGE"):
        params.update(num_blocks=trial.suggest_int("num_blocks", 0, 3))
        if model_name == "GraphSAGE-gru":
            params["lastlayer"] = "gru"
        elif model_name == "GraphSAGE-lstm":
            params["lastlayer"] = "lstm"
        params["eta_min"] = trial.suggest_float("eta_min", 1e-7, 1e-4, log=True)
        params["loader"] = "neighbor"
        params["neighbor_batch_size"] = trial.suggest_categorical(
            "neighbor_batch_size", [64, 128, 256]
        )
        high = max_neighbors if max_neighbors else 32
        size_1 = trial.suggest_int("neighbor_size_1", 1, high-1)
        params["neighbor_sizes"] = [
            size_1,
            trial.suggest_int("neighbor_size_2", size_1, high-1),
        ]
    elif model_name.startswith("GraphSAINT"):
        params.update(
            num_blocks=trial.suggest_int("num_blocks", 0, 3),
            walk_length=trial.suggest_int("walk_length", 1, 4),
            num_steps=trial.suggest_int("num_steps", 1, 4),
        )
        if model_name == "GraphSAINT-gru":
            params["lastlayer"] = "gru"
        elif model_name == "GraphSAINT-lstm":
            params["lastlayer"] = "lstm"
        params["eta_min"] = trial.suggest_float("eta_min", 1e-7, 1e-4, log=True)
        params["loader"] = "graphsaint"
    return params


def make_objective(model_name, dataset_name, dataset_path, adj_mx, cfg):
    model_cls = MODELS[model_name]
    max_neighbors = max(1, int(np.asarray(adj_mx).sum(axis=1).mean()))

    def objective(trial):
        params = suggest_params(trial, model_name, max_neighbors)
        model = model_cls(
            adj_mx=adj_mx,
            num_nodes=adj_mx.shape[0],
            input_dim=1,
            output_dim=1,
            seq_len=cfg.seq_len,
            horizon=cfg.horizon,
            epochs=cfg.epochs,
            device=cfg.device,
            **params,
        )
        def prune_callback(epoch, train_loss, val_loss):
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        try:
            data = train_and_evaluate(
                model,
                dataset_name=dataset_name,
                dataset_path=dataset_path,
                name=model_name,
                seq_len=cfg.seq_len,
                horizon=cfg.horizon,
                batch_size=cfg.batch_size,
                max_timesteps=cfg.max_timesteps,
                eval_split="val",
                fit_callback=prune_callback,
                num_workers=cfg.workers,
                persistent_workers=False,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise optuna.TrialPruned()
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()
        return data["mse"]

    return objective


def main():
    parser = argparse.ArgumentParser(
        description="Optimisation Optuna des hyperparamètres de tous les modèles."
    )
    parser.add_argument("--dataset", default="metr-la", choices=["metr-la", "pems-bay"])
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=15, help="Époques max par essai")
    parser.add_argument(
        "--models", nargs="+", default=list(MODELS), choices=list(MODELS)
    )
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--max-timesteps",
        type=int,
        default=2000,
        help="Nombre max de pas de série par essai (None/0 = série complète). "
        "Une valeur réduite accélère l'optimisation.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Workers du DataLoader d'entraînement. Garder 0 avec CUDA: les "
        "workers forked depuis un processus CUDA peuvent crasher.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.max_timesteps == 0:
        args.max_timesteps = None
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositif: {args.device}")

    dataset_path = Path(__file__).resolve().parent / "data"
    _, adj_mx = load_dataset_arrays(args.dataset, dataset_path)

    out_path = args.out or RESULTS_DIR / f"best_params_{args.dataset}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    header = {
        "dataset": args.dataset,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_trials": args.n_trials,
        "epochs_per_trial": args.epochs,
        "max_timesteps": args.max_timesteps,
        "batch_size": args.batch_size,
        "workers": args.workers,
    }

    results = None
    if out_path.exists():
        try:
            with open(out_path) as f:
                loaded = json.load(f)
            if isinstance(loaded, dict) and isinstance(loaded.get("models"), dict):
                results = {**header, **loaded}
        except json.JSONDecodeError:
            pass
    if results is None:
        results = {**header, "models": {}}

    for model_name in args.models:
        print(f"\n=== Optimisation de {model_name} ===")
        start = time.perf_counter()

        study_name = f"optuna_{args.dataset}_{model_name}"
        storage = f"sqlite:///{RESULTS_DIR / f'optuna_{args.dataset}.db'}"
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=args.seed),
            load_if_exists=True,
        )
        objective = make_objective(model_name, args.dataset, dataset_path, adj_mx, args)
        study.optimize(objective, n_trials=args.n_trials)
        elapsed = time.perf_counter() - start

        n_completed = sum(
            t.state == optuna.trial.TrialState.COMPLETE for t in study.trials
        )
        n_pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)
        print(
            f"{model_name}: meilleure val MSE = {study.best_value:.6f} "
            f"({n_completed} essais terminés, {n_pruned} élagués, {elapsed:.0f}s)"
        )
        print(f"Meilleurs paramètres: {study.best_params}")

        best_params = dict(study.best_params)
        if model_name.startswith("GraphSAGE") and "neighbor_size_1" in best_params:
            best_params["neighbor_sizes"] = [
                best_params.pop("neighbor_size_1"),
                best_params.pop("neighbor_size_2"),
            ]

        results["models"][model_name] = {
            "best_value": study.best_value,
            "best_params": best_params,
            "n_completed": n_completed,
            "n_pruned": n_pruned,
        }

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRésultats sauvegardés dans {out_path}")


if __name__ == "__main__":
    main()