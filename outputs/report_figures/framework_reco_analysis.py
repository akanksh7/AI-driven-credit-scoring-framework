import pandas as pd

results = pd.read_csv(r"c:/Users/AKANKSH/Desktop/Project-2/outputs/results.csv")
paired = pd.read_csv(r"c:/Users/AKANKSH/Desktop/Project-2/outputs/paired_tests.csv")

for dataset in sorted(results["dataset"].unique()):
    d = results[results["dataset"] == dataset].copy()
    print(f"\n=== {dataset} ===")

    # Low risk: maximize test accuracy
    low = d.sort_values("test_accuracy", ascending=False).iloc[0]
    print("low_risk_best", low["model"], "fair", low["fairness_level"], "acc", round(float(low["test_accuracy"]), 4), "dp", round(float(low["dp_diff"]), 4))

    # Medium risk: maintain >=95% of best baseline accuracy, then minimize dp_diff
    base = d[d["fairness_level"] == 0.0]
    base_best_acc = float(base["test_accuracy"].max())
    threshold = base_best_acc * 0.95
    eligible = d[d["test_accuracy"] >= threshold]
    med = eligible.sort_values(["dp_diff", "test_accuracy"], ascending=[True, False]).iloc[0]
    print("medium_risk_best", med["model"], "fair", med["fairness_level"], "acc", round(float(med["test_accuracy"]), 4), "dp", round(float(med["dp_diff"]), 4))

    # High risk: fairness first, then accuracy
    high = d.sort_values(["dp_diff", "test_accuracy"], ascending=[True, False]).iloc[0]
    print("high_risk_best", high["model"], "fair", high["fairness_level"], "acc", round(float(high["test_accuracy"]), 4), "dp", round(float(high["dp_diff"]), 4))

    p = paired[paired["dataset"] == dataset]
    if not p.empty:
        sig = (p["p_value"] < 0.05).mean()
        print("p_lt_0_05_share", round(float(sig), 3))
