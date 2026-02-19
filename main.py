import os

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from config import (
    CV_FOLDS,
    FAIRNESS_LEVELS,
    GLOBAL_RANDOM_SEED,
    MAX_ALLOWED_ACCURACY_DROP,
    OUTPUT_DIR,
    PAIRED_TESTS_PATH,
    PLOTS_DIR,
    RESULTS_PATH,
    SUBGROUP_METRICS_PATH,
    TARGET_FAIRNESS_DP,
    TEST_SIZE,
)
from data_loader import load_datasets
from decision import recommend_configuration, stakeholder_summary
from evaluation import compute_metrics, compute_cv_metrics, summarize_cv_metrics
from fairness import build_estimator, compute_fairness_metrics, fit_estimator, predict_scores, subgroup_metrics
from models import get_models
from preprocess import build_preprocessor, split_features_target
from visualize import (
    plot_baseline_vs_fairness,
    plot_combined_scatter,
    plot_pareto_frontier,
    plot_tradeoff_curves,
)


def run_experiment() -> None:
    np.random.seed(GLOBAL_RANDOM_SEED)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    datasets = load_datasets()
    if not datasets:
        print("No datasets found in data/. Place files and rerun.")
        return

    results = []
    subgroup_rows = []
    paired_tests = []

    for dataset in datasets:
        X, y, sensitive = split_features_target(dataset.data, dataset.target, dataset.protected)
        X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
            X,
            y,
            sensitive,
            test_size=TEST_SIZE,
            stratify=y,
            random_state=GLOBAL_RANDOM_SEED,
        )

        preprocessor = build_preprocessor(X_train)
        models = get_models()
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=GLOBAL_RANDOM_SEED)

        for model_name, model in models.items():
            baseline_fold_metrics = None

            for fairness_level in FAIRNESS_LEVELS:
                pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
                estimator = build_estimator(pipeline, fairness_level)

                fold_metrics = compute_cv_metrics(estimator, X_train, y_train, s_train, fairness_level, cv)
                summary = summarize_cv_metrics(fold_metrics)

                if fairness_level == 0.0:
                    baseline_fold_metrics = fold_metrics

                fitted = fit_estimator(estimator, X_train, y_train, s_train, fairness_level)
                y_pred = fitted.predict(X_test)
                y_score = predict_scores(fitted, X_test)

                metrics = compute_metrics(y_test, y_pred, y_score)
                fairness = compute_fairness_metrics(y_test.to_numpy(), y_pred, s_test)
                subgroup_df, _ = subgroup_metrics(y_test.to_numpy(), y_pred, s_test)

                for _, row in subgroup_df.iterrows():
                    subgroup_rows.append(
                        {
                            "dataset": dataset.name,
                            "model": model_name,
                            "fairness_level": fairness_level,
                            "group": row[dataset.protected],
                            "selection_rate": row["selection_rate"],
                            "tpr": row["tpr"],
                            "fpr": row["fpr"],
                        }
                    )

                results.append(
                    {
                        "dataset": dataset.name,
                        "model": model_name,
                        "fairness_level": fairness_level,
                        "cv_mean_accuracy": summary["accuracy"][0],
                        "cv_std_accuracy": summary["accuracy"][1],
                        "cv_ci_accuracy_low": summary["accuracy"][2],
                        "cv_ci_accuracy_high": summary["accuracy"][3],
                        "cv_mean_auc": summary["roc_auc"][0],
                        "cv_std_auc": summary["roc_auc"][1],
                        "cv_ci_auc_low": summary["roc_auc"][2],
                        "cv_ci_auc_high": summary["roc_auc"][3],
                        "cv_mean_f1": summary["f1"][0],
                        "cv_std_f1": summary["f1"][1],
                        "cv_ci_f1_low": summary["f1"][2],
                        "cv_ci_f1_high": summary["f1"][3],
                        "test_accuracy": metrics["accuracy"],
                        "auc": metrics["roc_auc"],
                        "f1": metrics["f1"],
                        "dp_diff": fairness["dp_diff"],
                        "eo_diff": fairness["eo_diff"],
                        "disparate_impact": fairness["disparate_impact"],
                    }
                )

                if fairness_level > 0.0 and baseline_fold_metrics is not None:
                    t_stat, p_val = ttest_rel(
                        baseline_fold_metrics["accuracy"],
                        fold_metrics["accuracy"],
                        nan_policy="omit",
                    )
                    paired_tests.append(
                        {
                            "dataset": dataset.name,
                            "model": model_name,
                            "fairness_level": fairness_level,
                            "t_stat": t_stat,
                            "p_value": p_val,
                        }
                    )

    results_df = pd.DataFrame(results)
    subgroup_df = pd.DataFrame(subgroup_rows)
    paired_df = pd.DataFrame(paired_tests)

    results_df.to_csv(RESULTS_PATH, index=False)
    subgroup_df.to_csv(SUBGROUP_METRICS_PATH, index=False)
    paired_df.to_csv(PAIRED_TESTS_PATH, index=False)

    plot_tradeoff_curves(results_df, PLOTS_DIR)
    plot_pareto_frontier(results_df, PLOTS_DIR)
    plot_combined_scatter(results_df, PLOTS_DIR)
    plot_baseline_vs_fairness(results_df, PLOTS_DIR)

    for dataset_name, group in results_df.groupby("dataset"):
        best, explanation = recommend_configuration(
            group,
            max_allowed_accuracy_drop=MAX_ALLOWED_ACCURACY_DROP,
            target_fairness_dp=TARGET_FAIRNESS_DP,
        )
        baseline = group[group["fairness_level"] == 0.0].sort_values(by="test_accuracy", ascending=False).iloc[0]
        print(f"\nDataset: {dataset_name}")
        print(stakeholder_summary(best, baseline))
        if explanation:
            print(f"Reasoning: {explanation['strategy']} ({explanation['threshold']})")


if __name__ == "__main__":
    run_experiment()
