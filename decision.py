from typing import Dict, Tuple

import pandas as pd


def recommend_configuration(
    results: pd.DataFrame,
    max_allowed_accuracy_drop: float = None,
    target_fairness_dp: float = None,
) -> Tuple[pd.Series, Dict[str, str]]:
    baseline = results[results["fairness_level"] == 0.0]
    best_baseline_acc = float(baseline["test_accuracy"].max()) if not baseline.empty else 0.0

    explanation: Dict[str, str] = {}

    if max_allowed_accuracy_drop is not None:
        threshold = best_baseline_acc * (1.0 - max_allowed_accuracy_drop)
        eligible = results[results["test_accuracy"] >= threshold]
        if not eligible.empty:
            best = eligible.sort_values(by=["dp_diff", "test_accuracy"], ascending=[True, False]).iloc[0]
            explanation["strategy"] = "Accuracy tolerance"
            explanation["threshold"] = f"Maintained accuracy >= {threshold:.3f}"
            return best, explanation

    if target_fairness_dp is not None:
        eligible = results[results["dp_diff"] <= target_fairness_dp]
        if not eligible.empty:
            best = eligible.sort_values(by=["test_accuracy"], ascending=[False]).iloc[0]
            explanation["strategy"] = "Fairness target"
            explanation["threshold"] = f"Demographic parity difference <= {target_fairness_dp:.3f}"
            return best, explanation

    best = results.sort_values(by=["dp_diff", "test_accuracy"], ascending=[True, False]).iloc[0]
    explanation["strategy"] = "Closest trade-off"
    explanation["threshold"] = "No configuration met requested constraints"
    return best, explanation


def stakeholder_summary(best: pd.Series, baseline: pd.Series) -> str:
    acc_change = best["test_accuracy"] - baseline["test_accuracy"]
    dp_improvement = baseline["dp_diff"] - best["dp_diff"]

    lines = [
        f"Recommended model: {best['model']} with fairness level {best['fairness_level']}",
        f"Accuracy change: {acc_change:+.3f} (from {baseline['test_accuracy']:.3f} to {best['test_accuracy']:.3f})",
        f"Fairness improvement (DP difference): {dp_improvement:+.3f} (from {baseline['dp_diff']:.3f} to {best['dp_diff']:.3f})",
        "This option balances predictive performance with more equitable outcomes across protected groups.",
    ]
    return "\n".join(lines)
