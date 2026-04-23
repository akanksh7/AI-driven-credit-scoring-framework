import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# =========================
# CONFIG
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SUBGROUP_FILE = os.path.join(BASE_DIR, "subgroup_metrics.csv")
RESULTS_FILE = os.path.join(BASE_DIR, "results.csv")
PAIRED_FILE = os.path.join(BASE_DIR, "paired_tests.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "fairness_plots_clean")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# LOAD DATA
# =========================
subgroup_df = pd.read_csv(SUBGROUP_FILE)
results_df = pd.read_csv(RESULTS_FILE)
paired_df = pd.read_csv(PAIRED_FILE)

# Clean
subgroup_df = subgroup_df.dropna()
results_df = results_df.dropna()
paired_df = paired_df.dropna()

datasets = subgroup_df["dataset"].unique()

# =========================
# MODEL SELECTION (BEST/WORST)
# =========================
def get_best_worst_models(dataset):
    df = results_df[results_df["dataset"] == dataset]

    # Aggregate per model
    agg = df.groupby("model").agg({
        "auc": "mean",
        "dp_diff": "mean"
    }).reset_index()

    # Score: maximize AUC, minimize dp_diff
    agg["score"] = agg["auc"] - agg["dp_diff"]

    best_model = agg.sort_values("score", ascending=False).iloc[0]["model"]
    worst_model = agg.sort_values("score", ascending=True).iloc[0]["model"]

    return best_model, worst_model

# =========================
# HELPER
# =========================
def get_gap(model_df, metric):
    fairness_levels = sorted(model_df["fairness_level"].unique())
    gap = []

    for f in fairness_levels:
        temp = model_df[model_df["fairness_level"] == f]
        group_values = temp[metric].dropna().to_numpy(dtype=float)

        # Compute disparity between best and worst subgroup at each fairness level.
        if len(group_values) >= 2:
            gap.append(float(np.max(group_values) - np.min(group_values)))
        else:
            gap.append(np.nan)

    return fairness_levels, gap

# =========================
# 1. TPR GAP
# =========================
def plot_tpr_gap(dataset, models):
    plt.figure()

    for model in models:
        df = subgroup_df[
            (subgroup_df["dataset"] == dataset) &
            (subgroup_df["model"] == model)
        ]

        if df.empty:
            continue

        x, y = get_gap(df, "tpr")
        plt.plot(x, y, marker='o', label=model)

    plt.title(f"TPR Gap ({dataset})")
    plt.xlabel("Fairness Level")
    plt.ylabel("TPR Gap")
    plt.legend()
    plt.grid()

    plt.savefig(f"{OUTPUT_DIR}/{dataset}_tpr_gap.png")
    plt.close()

# =========================
# 2. SELECTION RATE GAP
# =========================
def plot_selection_gap(dataset, models):
    plt.figure()

    for model in models:
        df = subgroup_df[
            (subgroup_df["dataset"] == dataset) &
            (subgroup_df["model"] == model)
        ]

        if df.empty:
            continue

        x, y = get_gap(df, "selection_rate")
        plt.plot(x, y, marker='o', label=model)

    plt.title(f"Selection Rate Gap ({dataset})")
    plt.xlabel("Fairness Level")
    plt.ylabel("Gap")
    plt.legend()
    plt.grid()

    plt.savefig(f"{OUTPUT_DIR}/{dataset}_selection_gap.png")
    plt.close()

# =========================
# 3. AUC vs FAIRNESS
# =========================
def plot_auc(dataset, models):
    plt.figure()

    for model in models:
        df = results_df[
            (results_df["dataset"] == dataset) &
            (results_df["model"] == model)
        ]

        if df.empty:
            continue

        plt.plot(df["fairness_level"], df["auc"], marker='o', label=model)

    plt.title(f"AUC vs Fairness ({dataset})")
    plt.xlabel("Fairness Level")
    plt.ylabel("AUC")
    plt.legend()
    plt.grid()

    plt.savefig(f"{OUTPUT_DIR}/{dataset}_auc.png")
    plt.close()

# =========================
# 4. FAIRNESS METRICS
# =========================
def plot_metric(dataset, models, metric):
    plt.figure()

    for model in models:
        df = results_df[
            (results_df["dataset"] == dataset) &
            (results_df["model"] == model)
        ]

        if df.empty or metric not in df.columns:
            continue

        plt.plot(df["fairness_level"], df[metric], marker='o', label=model)

    plt.title(f"{metric} ({dataset})")
    plt.xlabel("Fairness Level")
    plt.ylabel(metric)
    plt.legend()
    plt.grid()

    plt.savefig(f"{OUTPUT_DIR}/{dataset}_{metric}.png")
    plt.close()

# =========================
# 5. P-VALUES
# =========================
def plot_pvalues(dataset, models):
    plt.figure()

    for model in models:
        df = paired_df[
            (paired_df["dataset"] == dataset) &
            (paired_df["model"] == model)
        ]

        if df.empty or "p_value" not in df.columns:
            continue

        plt.plot(df["fairness_level"], df["p_value"], marker='o', label=model)

    plt.axhline(y=0.05, linestyle='--')

    plt.title(f"P-Values ({dataset})")
    plt.xlabel("Fairness Level")
    plt.ylabel("p-value")
    plt.legend()
    plt.grid()

    plt.savefig(f"{OUTPUT_DIR}/{dataset}_pvalues.png")
    plt.close()

# =========================
# 6. TRADEOFF
# =========================
def plot_tradeoff(dataset, models):
    plt.figure()

    for model in models:
        df = results_df[
            (results_df["dataset"] == dataset) &
            (results_df["model"] == model)
        ]

        if df.empty:
            continue

        df = df.sort_values(by="dp_diff")
        plt.plot(df["dp_diff"], df["auc"], marker='o', label=model)

    plt.title(f"Fairness vs Performance ({dataset})")
    plt.xlabel("dp_diff")
    plt.ylabel("AUC")
    plt.legend()
    plt.grid()

    plt.savefig(f"{OUTPUT_DIR}/{dataset}_tradeoff.png")
    plt.close()

# =========================
# RUN
# =========================
for dataset in datasets:
    print(f"\nProcessing: {dataset}")

    best, worst = get_best_worst_models(dataset)
    print(f"Best: {best}, Worst: {worst}")

    models = [best, worst]

    plot_tpr_gap(dataset, models)
    plot_selection_gap(dataset, models)
    plot_auc(dataset, models)
    plot_metric(dataset, models, "dp_diff")
    plot_metric(dataset, models, "eo_diff")
    plot_pvalues(dataset, models)
    plot_tradeoff(dataset, models)

print("\n✅ Clean plots generated!")