import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from shared.loaders import load_dataset_arrays

DATASET_PATH = Path(__file__).resolve().parent / "data"


def main():
    parser = argparse.ArgumentParser(description="Distribution des degrés du graphe.")
    parser.add_argument("--dataset", default="metr-la", choices=["metr-la", "pems-bay"])
    args = parser.parse_args()

    _, adj = load_dataset_arrays(args.dataset, DATASET_PATH)
    degrees = adj.sum(axis=1)
    q1, q2, q3 = np.percentile(degrees, [25, 50, 75])

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle(f"Degrés du graphe — {args.dataset} ({len(degrees)} nœuds)", fontsize=12)

    ax.boxplot(
        degrees,
        showfliers=True,
        patch_artist=True,
        widths=0.4,
        boxprops=dict(facecolor="seagreen", alpha=0.85),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        medianprops=dict(color="black"),
    )

    ymin, ymax = ax.get_ylim()
    pad = (ymax - ymin) * 0.04
    for q, c, label, bold in [
        (q1, "blue", f"Q1={q1:.1f}", False),
        (q2, "red", f"Q2={q2:.1f}", True),
        (q3, "green", f"Q3={q3:.1f}", False),
    ]:
        if abs(q - q2) < (ymax - ymin) * 0.06:
            continue
        ax.axhline(q, color=c, linestyle="--", linewidth=1.8)
        ax.annotate(
            label, xy=(1.25, q), xytext=(1.45, q + pad),
            color=c, fontsize=11, va="bottom",
            fontweight="bold" if bold else "normal",
        )
    ax.annotate(
        f"Q2={q2:.1f}",
        xy=(1.25, q2), xytext=(1.45, q2 + pad),
        color="red", fontsize=11, va="bottom", fontweight="bold",
    )

    ax.set_xlim(0.5, 2.2)
    ax.set_xticks([])
    ax.margins(y=0.1)
    ax.set_title("Distribution des degrés")
    ax.set_ylabel("Degré")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()