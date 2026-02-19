from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd


def plot_tradeoff_curves(results: pd.DataFrame, output_dir: str) -> None:
    for (dataset, model), group in results.groupby(["dataset", "model"]):
        group = group.sort_values(by="fairness_level")
        plt.figure(figsize=(6, 4))
        plt.plot(group["dp_diff"], group["test_accuracy"], marker="o")
        plt.xlabel("Demographic Parity Difference (lower is better)")
        plt.ylabel("Test Accuracy")
        plt.title(f"Trade-off Curve - {dataset} - {model}")
        plt.grid(True, alpha=0.3)
        path = f"{output_dir}/{dataset}_{model}_tradeoff.png"
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close()


def plot_pareto_frontier(results: pd.DataFrame, output_dir: str) -> None:
    for dataset, group in results.groupby("dataset"):
        pareto = _pareto_frontier(group)
        plt.figure(figsize=(6, 4))
        plt.scatter(group["dp_diff"], group["test_accuracy"], alpha=0.5)
        plt.plot(pareto["dp_diff"], pareto["test_accuracy"], color="red", marker="o")
        plt.xlabel("Demographic Parity Difference (lower is better)")
        plt.ylabel("Test Accuracy")
        plt.title(f"Pareto Frontier - {dataset}")
        plt.grid(True, alpha=0.3)
        path = f"{output_dir}/{dataset}_pareto_frontier.png"
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close()


def plot_combined_scatter(results: pd.DataFrame, output_dir: str) -> None:
    plt.figure(figsize=(7, 5))
    for model, group in results.groupby("model"):
        plt.scatter(group["dp_diff"], group["test_accuracy"], label=model, alpha=0.7)
    plt.xlabel("Demographic Parity Difference (lower is better)")
    plt.ylabel("Test Accuracy")
    plt.title("Fairness-Accuracy Trade-off (All Models)")
    plt.legend(fontsize=7)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/combined_scatter.png", dpi=300)
    plt.close()


def plot_baseline_vs_fairness(results: pd.DataFrame, output_dir: str) -> None:
    for dataset, group in results.groupby("dataset"):
        baseline = group[group["fairness_level"] == 0.0]
        mitigated = group[group["fairness_level"] > 0.0]
        if baseline.empty or mitigated.empty:
            continue

        best_mitigated = mitigated.sort_values(by=["dp_diff", "test_accuracy"], ascending=[True, False])
        best_mitigated = best_mitigated.groupby("model").first().reset_index()

        merged = baseline.merge(best_mitigated, on="model", suffixes=("_base", "_mit"))
        x = range(len(merged))

        plt.figure(figsize=(7, 4))
        plt.bar(x, merged["test_accuracy_base"], width=0.4, label="Baseline")
        plt.bar([i + 0.4 for i in x], merged["test_accuracy_mit"], width=0.4, label="Mitigated")
        plt.xticks([i + 0.2 for i in x], merged["model"], rotation=45, ha="right")
        plt.ylabel("Test Accuracy")
        plt.title(f"Baseline vs Mitigated Accuracy - {dataset}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{dataset}_baseline_vs_mitigated_accuracy.png", dpi=300)
        plt.close()

        plt.figure(figsize=(7, 4))
        plt.bar(x, merged["dp_diff_base"], width=0.4, label="Baseline")
        plt.bar([i + 0.4 for i in x], merged["dp_diff_mit"], width=0.4, label="Mitigated")
        plt.xticks([i + 0.2 for i in x], merged["model"], rotation=45, ha="right")
        plt.ylabel("Demographic Parity Difference")
        plt.title(f"Baseline vs Mitigated Fairness - {dataset}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{dataset}_baseline_vs_mitigated_fairness.png", dpi=300)
        plt.close()


def _pareto_frontier(group: pd.DataFrame) -> pd.DataFrame:
    sorted_group = group.sort_values(by=["dp_diff", "test_accuracy"], ascending=[True, False])
    frontier = []
    best_accuracy = -1.0
    for _, row in sorted_group.iterrows():
        if row["test_accuracy"] > best_accuracy:
            frontier.append(row)
            best_accuracy = row["test_accuracy"]
    return pd.DataFrame(frontier)
