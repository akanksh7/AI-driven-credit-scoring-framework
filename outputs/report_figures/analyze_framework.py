import os
import sys

import pandas as pd

sys.path.append(r"c:/Users/AKANKSH/Desktop/Project-2/src")
from data_loader import load_datasets  # noqa: E402


def main() -> None:
    datasets = load_datasets()
    print("loaded", [d.name for d in datasets])

    for d in datasets:
        df = d.data
        counts = df[d.protected].value_counts(dropna=False)
        total = len(df)
        shares = (counts / total).to_dict()
        rates = df.groupby(d.protected)[d.target].mean().to_dict()
        rate_gap = max(rates.values()) - min(rates.values()) if rates else 0.0

        print(f"--- {d.name}")
        print("n", total)
        print("groups", counts.to_dict())
        print("group_share", {k: round(v, 4) for k, v in shares.items()})
        print("default_rate_by_group", {k: round(v, 4) for k, v in rates.items()})
        print("default_rate_gap", round(rate_gap, 4))

    results = pd.read_csv(r"c:/Users/AKANKSH/Desktop/Project-2/outputs/results.csv")

    print("\nModel stability view from baseline->fairness progression (using dp_diff):")
    for dataset in sorted(results["dataset"].unique()):
        print(f"\n[{dataset}]")
        for model in sorted(results["model"].unique()):
            g = results[(results["dataset"] == dataset) & (results["model"] == model)].sort_values("fairness_level")
            if g.empty:
                continue
            y = g["dp_diff"].to_numpy()
            diffs = y[1:] - y[:-1]
            if len(diffs) == 0:
                trend = "n/a"
                instability = 0.0
            else:
                decreases = (diffs < 0).sum()
                trend = "mostly_improves" if decreases >= 3 else "mixed"
                instability = float(abs(diffs).std())
            print(model, "trend", trend, "instability", round(instability, 5), "start", round(float(y[0]), 4), "end", round(float(y[-1]), 4))


if __name__ == "__main__":
    main()
