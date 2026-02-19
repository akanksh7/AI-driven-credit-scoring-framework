from typing import Dict, Tuple

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
    selection_rate,
    true_positive_rate,
    false_positive_rate,
)
from fairlearn.reductions import DemographicParity, ExponentiatedGradient
from sklearn.base import clone
from sklearn.pipeline import Pipeline


def fairness_eps_from_level(level: float) -> float:
    return max(0.01, 1.0 - level)


def build_estimator(base_pipeline: Pipeline, fairness_level: float):
    if fairness_level <= 0.0:
        return base_pipeline

    eps = fairness_eps_from_level(fairness_level)
    return ExponentiatedGradient(
        base_pipeline,
        constraints=DemographicParity(),
        eps=eps,
        sample_weight_name="model__sample_weight",
    )


def fit_estimator(estimator, X, y, sensitive, fairness_level: float):
    estimator = clone(estimator)
    if fairness_level <= 0.0:
        estimator.fit(X, y)
    else:
        estimator.fit(X, y, sensitive_features=sensitive)
    return estimator


def predict_scores(estimator, X) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    if hasattr(estimator, "decision_function"):
        return estimator.decision_function(X)
    return estimator.predict(X)


def compute_fairness_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: pd.Series,
) -> Dict[str, float]:
    dp_diff = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive)
    eo_diff = equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive)
    disparate_impact = _disparate_impact_ratio(y_pred, sensitive)
    return {
        "dp_diff": dp_diff,
        "eo_diff": eo_diff,
        "disparate_impact": disparate_impact,
    }


def subgroup_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: pd.Series,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    metrics = {
        "selection_rate": selection_rate,
        "tpr": true_positive_rate,
        "fpr": false_positive_rate,
    }
    frame = MetricFrame(metrics=metrics, y_true=y_true, y_pred=y_pred, sensitive_features=sensitive)
    subgroup_df = frame.by_group.reset_index().rename(columns={"index": "group"})
    overall = {name: frame.overall[name] for name in metrics}
    return subgroup_df, overall


def _disparate_impact_ratio(y_pred: np.ndarray, sensitive: pd.Series) -> float:
    rates = (
        pd.Series(y_pred)
        .groupby(sensitive)
        .apply(lambda x: (x == 1).mean())
        .dropna()
        .values
    )
    if len(rates) < 2:
        return 1.0
    min_rate = float(np.min(rates))
    max_rate = float(np.max(rates))
    if max_rate == 0:
        return 1.0
    return min_rate / max_rate
