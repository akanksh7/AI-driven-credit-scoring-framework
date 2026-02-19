# Fairness–Accuracy Trade-off in Credit Risk Prediction

## What this repository is about

This project runs a **research-grade, reproducible experiment pipeline** to study the trade-off between:

- predictive performance (accuracy, ROC-AUC, F1), and
- fairness across protected groups (demographic parity, equalized odds, disparate impact).

The goal is to support scientifically sound model selection for credit-risk decisions where fairness constraints matter.

## Research goal

For each credit dataset, we compare multiple model families and fairness mitigation strengths to answer:

1. How much fairness improvement is achievable?
2. What performance is lost (or preserved) as fairness constraints increase?
3. Which configuration is the best practical compromise for decision-makers?

## What the pipeline does

Running `python main.py` executes the full workflow:

1. Load available datasets from `data/` (currently: `australian.dat`, `german.data`, `gmsc.csv`)
2. Apply a consistent preprocessing pipeline (imputation, encoding, scaling)
3. Train model families in increasing complexity:
	 - Logistic Regression
	 - Decision Tree
	 - Random Forest
	 - Gradient Boosting
	 - Explainable Boosting Machine (EBM)
	 - XGBoost (CPU)
4. Apply fairness mitigation at levels: `[0.0, 0.25, 0.5, 0.75, 1.0]`
5. Evaluate performance + fairness metrics
6. Compute cross-validation mean/std and 95% confidence intervals
7. Run paired baseline-vs-mitigated statistical comparison (paired t-test)
8. Generate publication-ready plots and decision recommendation summary

## Reproducibility setup

- Global random seed: `42`
- Stratified train/test split: `80/20`
- Cross-validation: `5-fold stratified`
- Fold-wise metrics stored and summarized with confidence intervals

## Repository structure

- `config.py`: experiment constants and paths
- `data_loader.py`: dataset-specific loading/parsing
- `preprocess.py`: unified feature preprocessing
- `models.py`: model definitions
- `fairness.py`: mitigation and fairness metric utilities
- `evaluation.py`: CV/test metric computation and CI logic
- `decision.py`: decision-aware recommender + stakeholder summary
- `visualize.py`: trade-off and comparison plots
- `main.py`: end-to-end experiment runner

## Outputs and what they mean

- `outputs/results.csv`
	- One row per dataset × model × fairness level
	- Contains CV stats, test performance, and fairness metrics
- `outputs/subgroup_metrics.csv`
	- Group-level selection rate, TPR, and FPR per configuration
- `outputs/paired_tests.csv`
	- Paired t-test results comparing baseline vs mitigated accuracy
- `outputs/plots/*.png`
	- Trade-off curves per model
	- Pareto frontier per dataset
	- Combined fairness–accuracy scatter
	- Baseline vs mitigated bar comparisons

## What is achieved so far

The full experiment has been run and artifacts are already generated for:

- all 3 datasets (`australian`, `german`, `gmsc`)
- all 6 model families
- all 5 fairness levels

Generated plots include per-dataset trade-off/pareto figures and global comparison visuals in `outputs/plots/`.

## Current recommendation snapshot (latest run)

These are the stakeholder-facing recommendations produced by the current run configuration:

- **Dataset: australian**
	- Recommended model: `GradientBoosting` at fairness level `0.0`
	- Accuracy change: `-0.043` (from `0.870` to `0.826`)
	- Reasoning: Accuracy tolerance (maintained accuracy `>= 0.722`)

- **Dataset: gmsc**
	- Recommended model: `LogisticRegression` at fairness level `0.25`
	- Accuracy change: `-0.004` (from `0.938` to `0.934`)
	- Fairness improvement (DP difference): `+0.016` (from `0.024` to `0.009`)
	- Reasoning: Accuracy tolerance (maintained accuracy `>= 0.891`)

Notes:

- The recommendation logic prioritizes fairness under an allowed accuracy-drop threshold.
- Values can change if data preprocessing, fairness levels, or model settings are modified and rerun.

## Run

Install dependencies:

`pip install -r requirements.txt`

Execute:

`python main.py`
