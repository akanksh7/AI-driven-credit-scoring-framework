import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
RESULTS_PATH = os.path.join(ROOT_DIR, "results.csv")


def plot_ci_auc(results: pd.DataFrame, output_path: str) -> None:
    datasets = sorted(results["dataset"].unique())
    fig, axes = plt.subplots(len(datasets), 1, figsize=(10, 4 * len(datasets)), sharex=False)
    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        subset = results[results["dataset"] == dataset].copy()
        agg = (
            subset.groupby("fairness_level", as_index=False)
            .agg(
                cv_mean_auc=("cv_mean_auc", "mean"),
                cv_ci_auc_low=("cv_ci_auc_low", "mean"),
                cv_ci_auc_high=("cv_ci_auc_high", "mean"),
            )
            .sort_values("fairness_level")
        )

        y = agg["cv_mean_auc"].to_numpy(dtype=float)
        yerr_low = y - agg["cv_ci_auc_low"].to_numpy(dtype=float)
        yerr_high = agg["cv_ci_auc_high"].to_numpy(dtype=float) - y

        ax.errorbar(
            agg["fairness_level"],
            y,
            yerr=[yerr_low, yerr_high],
            fmt="-o",
            capsize=4,
            linewidth=1.8,
        )
        ax.set_title(f"{dataset.upper()}: CV Mean AUC with 95% CI")
        ax.set_xlabel("Fairness level")
        ax.set_ylabel("CV mean AUC")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_heatmap_accuracy(results: pd.DataFrame, output_path: str) -> None:
    datasets = sorted(results["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.5 * len(datasets), 6), sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        subset = results[results["dataset"] == dataset].copy()
        pivot = subset.pivot_table(
            index="model",
            columns="fairness_level",
            values="test_accuracy",
            aggfunc="mean",
        )
        pivot = pivot.sort_index(axis=0)
        pivot = pivot.sort_index(axis=1)

        data = pivot.to_numpy(dtype=float)
        im = ax.imshow(data, aspect="auto", cmap="YlGnBu")

        ax.set_title(f"{dataset.upper()}: Test Accuracy Heatmap")
        ax.set_xlabel("Fairness level")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([f"{c:.2f}" for c in pivot.columns], rotation=45, ha="right")

        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                value = data[i, j]
                if not np.isnan(value):
                    ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=8, color="black")

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Test accuracy")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    results = pd.read_csv(RESULTS_PATH)

    ci_path = os.path.join(BASE_DIR, "figure_ci_auc_95.png")
    heatmap_path = os.path.join(BASE_DIR, "figure_accuracy_heatmap.png")

    plot_ci_auc(results, ci_path)
    plot_heatmap_accuracy(results, heatmap_path)

    print("Generated:")
    print(ci_path)
    print(heatmap_path)


if __name__ == "__main__":
    main()
