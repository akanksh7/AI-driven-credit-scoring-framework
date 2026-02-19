from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from fairness import fit_estimator, predict_scores


def compute_metrics(y_true, y_pred, y_score) -> Dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics


def compute_cv_metrics(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    sensitive: pd.Series,
    fairness_level: float,
    cv: StratifiedKFold,
) -> Dict[str, List[float]]:
    fold_metrics: Dict[str, List[float]] = {"accuracy": [], "f1": [], "roc_auc": []}

    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        s_train, s_val = sensitive.iloc[train_idx], sensitive.iloc[val_idx]

        fitted = fit_estimator(estimator, X_train, y_train, s_train, fairness_level)
        y_pred = fitted.predict(X_val)
        y_score = predict_scores(fitted, X_val)
        metrics = compute_metrics(y_val, y_pred, y_score)

        for key in fold_metrics:
            fold_metrics[key].append(metrics[key])

    return fold_metrics


def summarize_cv_metrics(fold_metrics: Dict[str, List[float]]) -> Dict[str, Tuple[float, float, float, float]]:
    summary = {}
    for name, values in fold_metrics.items():
        values_array = np.array(values, dtype=float)
        mean = float(np.nanmean(values_array))
        std = float(np.nanstd(values_array, ddof=1))
        ci_low, ci_high = confidence_interval(values_array)
        summary[name] = (mean, std, ci_low, ci_high)
    return summary


def confidence_interval(values: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    z = 1.96
    half_width = z * std / np.sqrt(len(values))
    return mean - half_width, mean + half_width
