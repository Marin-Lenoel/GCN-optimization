import json
import os
from pathlib import Path
import numpy as np
import torch
import pandas as pd
from models import MODELS
from shared.loaders import load_dataset_arrays, train_and_evaluate
import matplotlib.pyplot as plt

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

DATASET_NAME = "pems-bay"
DATASET_PATH = Path(__file__).resolve().parent / "data"
SEQ_LEN = 12
HORIZON = 12
BATCH_SIZE = 64
EPOCHS = 5
MAX_TIMESTEPS = None
RESULTS_PATH = Path(__file__).resolve().parent / "results" / f"best_params_{DATASET_NAME}.json"
torch.set_float32_matmul_precision("high")

DEFAULT_PARAMS = {
    "MTGNN": dict(hidden_dim=8, num_blocks=1, patience=2),
    "GraphSAGE": dict(hidden_dim=32, num_blocks=3, patience=4,
                      loader="neighbor", neighbor_batch_size=64),
    "GraphSAGE-gru": dict(hidden_dim=16, num_blocks=2, patience=4, lastlayer="gru",
                          loader="neighbor", neighbor_batch_size=64),
    "GraphSAGE-lstm": dict(hidden_dim=16, num_blocks=2, patience=4, lastlayer="lstm",
                           loader="neighbor", neighbor_batch_size=64),
    "GraphSAINT": dict(hidden_dim=12, num_blocks=1, patience=8),
    "GraphSAINT-gru": dict(hidden_dim=12, num_blocks=1, patience=4, lastlayer="gru"),
    "GraphSAINT-lstm": dict(hidden_dim=12, num_blocks=1, patience=4, lastlayer="lstm"),
    "RepeatLast": dict(),
}


def main(dataset_name=DATASET_NAME, dataset_path=DATASET_PATH):
    np.random.seed(42)
    torch.manual_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {device}")

    _, adj_mx = load_dataset_arrays(dataset_name, dataset_path)

    best_params = {}
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            data = json.load(f)
        best_params = {name: info.get("best_params", {}) for name, info in data.get("models", {}).items()}

    common = dict(
        adj_mx=adj_mx,
        num_nodes=adj_mx.shape[0],
        input_dim=1,
        output_dim=1,
        seq_len=SEQ_LEN,
        horizon=HORIZON,
        epochs=EPOCHS,
        device=device,
    )
    models = []
    for name, cls in MODELS.items():
        params = {**DEFAULT_PARAMS[name], **best_params.get(name, {})}
        models.append((name, cls(**common, **params)))
        source = "Optuna" if name in best_params else "défaut"
        print(f"{name}: hyperparamètres {source} -> {params}")

    models_data = []
    for name, model in models:
        print("model : " + name)
        print("\nTreinando... ")
        data = train_and_evaluate(
            model,
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            seq_len=SEQ_LEN,
            horizon=HORIZON,
            name=name,
            batch_size=BATCH_SIZE,
            max_timesteps=MAX_TIMESTEPS,
            compile=False,
        )
        print("\nTestando...")
        models_data.append(data)
        print(f"y_true shape: {data['y_true'].shape}")
        print(f"y_pred shape: {data['y_pred'].shape}")
        print("\nExemplo — sensor 0, primeira janela de teste:")
        print("Verdadeiro:", np.round(data["y_true"][0, :, 0, 0], 2))
        print("Previsto:  ", np.round(data["y_pred"][0, :, 0, 0], 2))

    print(pd.DataFrame(models_data).drop(['errors', "train_loss", "val_loss", "errors_timed", "y_pred", "y_true", "epochs", "x"], axis=1))

    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(
        [data["errors"] for data in models_data],
        showfliers=False, patch_artist=True,
    )
    ax.set_xticklabels([data["name"] for data in models_data])
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(plt.rcParams["axes.prop_cycle"].by_key()["color"][i % len(bp["boxes"])])
    for element in ("medians", "whiskers", "caps"):
        for line in bp[element]:
            line.set_color("black")
    for i, data in enumerate(models_data):
        error = data["errors"]
        q1, q2, q3 = np.percentile(error, [25, 50, 75])
        ax.text(i + 1.30, q1, f'Q1={q1:.2f}', va='center', fontsize=8)
        ax.text(i + 1.30, q2, f'Q2={q2:.2f}', va='center', fontweight='bold', fontsize=8)
        ax.text(i + 1.30, q3, f'Q3={q3:.2f}', va='center', fontsize=8)
    ax.set_ylabel('Prédit - Vrai')
    ax.set_title('Distribution des erreurs par modèle')
    ax.grid(True, axis='y', alpha=0.3)
    plt.show()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, data in enumerate(models_data):
        c = colors[i % len(colors)]
        ax.plot(data["train_loss"], color=c, ls="-", label=f'{data["name"]} train')
        ax.plot(data["val_loss"], color=c, ls="--", label=f'{data["name"]} val')
    ax.set_yscale("log")
    ax.set_xlabel("Époque")
    ax.set_ylabel("Loss")
    ax.set_title("Courbes de perte par modèle")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()

    timesteps = np.arange(0, HORIZON)
    timesteps_label = ["t+" + str(i + 1) for i in timesteps]
    for i, data in enumerate(models_data):
        plt.plot(data["errors_timed"], label=data["name"])
    plt.legend()
    plt.xticks(timesteps, labels=timesteps_label)
    plt.show()
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    idx_ref = None
    for i, data in enumerate(models_data):
        yt = data["y_true"][:, :, 1, 0]
        valid = (yt > 0.5).all(axis=1)
        move = np.abs(np.diff(yt, axis=1)).mean(axis=1)
        cands = np.where(valid & (move > 1.0))[0]
        idx = int(cands[np.argmax(move[cands])]) if len(cands) else 0
        if idx_ref is None:
            idx_ref = idx
            xy = np.concatenate((data["x"][idx, :, 1, 0], data["y_true"][idx, :, 1, 0]))
            ax.plot(xy, "k-", lw=2, label="Réel")
            ax.axvline(SEQ_LEN - 0.5, color="gray", ls="--", lw=1)
        ax.plot(
            SEQ_LEN + np.arange(HORIZON),
            data["y_pred"][idx, :, 1, 0],
            marker="o", ms=3,
            color=colors[i % len(colors)],
            label=data["name"],
        )
    ax.set_title(f"Prédictions vs Réel — capteur 1, fenêtre {idx_ref}")
    ax.set_xlabel("Pas temporel")
    ax.set_ylabel("Valeur")
    ax.legend(loc="upper right")
    plt.show()
    return models_data[-1]["y_true"], models_data[-1]["y_pred"]


if __name__ == "__main__":
    main()