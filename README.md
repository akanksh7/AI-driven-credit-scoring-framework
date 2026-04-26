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

Running `python src/main.py` executes the full workflow:

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

`python src/main.py`

## TECHNICAL SPECIFICATION

### 3.1 REQUIREMENTS

#### 3.1.1 FUNCTIONAL REQUIREMENTS

FR1. The system is designed to load and preprocess credit datasets directly from the `data/` folder. Right now, it includes working loaders for Australian Credit, German Credit, and GMSC, and it automatically runs on whichever of these files are available.

FR2. The system trains a diverse set of machine learning models for credit-risk prediction, including:
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- Explainable Boosting Machine (EBM)
- XGBoost (CPU mode)

FR3. The system evaluates trained models using clear, standard performance metrics, including:
- Accuracy
- ROC-AUC
- F1-score

FR4. The system measures fairness across protected groups using established fairness indicators, including:
- Demographic Parity Difference
- Equalized Odds Difference
- Disparate Impact Ratio

FR5. The system applies fairness mitigation through constrained optimization (Fairlearn Exponentiated Gradient with Demographic Parity) at multiple severity levels, then records how those choices affect model performance.

FR6. The system runs stratified cross-validation and reports mean, standard deviation, and confidence intervals so model comparisons are statistically meaningful, not just anecdotal.

FR7. The system performs paired statistical testing (paired t-test) between baseline and fairness-mitigated runs to check whether observed accuracy changes are likely real or just random variation.

FR8. The system produces decision-friendly visual outputs, including:
- Fairness-accuracy trade-off curves
- Pareto frontier plots
- Combined fairness-accuracy scatter plots across models
- Baseline vs mitigated comparison charts (accuracy and fairness)

FR9. The system supports reproducible experiments through centralized configuration, fixed random seeds, and a consistent preprocessing/training workflow.

#### 3.1.2 NON-FUNCTIONAL REQUIREMENTS

NFR1. Performance:
Experiments are expected to run on CPU-only hardware and finish within practical research timelines for small-to-medium tabular datasets.

NFR2. Scalability:
The current implementation handles typical academic-scale tabular datasets well; if the data grows very large (for example, around 1 million rows), extra sampling or optimization steps may be needed.

NFR3. Reliability:
Results are reproducible across runs thanks to deterministic seeds, fixed train/test splitting, and a uniform preprocessing pipeline.

NFR4. Usability:
The full pipeline can be executed with a single command (`python src/main.py`) and does not require specialized hardware.

NFR5. Maintainability:
The codebase is organized into modular, task-focused files for loading, preprocessing, modeling, fairness logic, evaluation, visualization, and decision recommendation, making updates easier over time.

NFR6. Portability:
The system runs on standard Windows, Linux, and macOS environments as long as Python and the required dependencies are installed.

### 3.2 FEASIBILITY STUDY

#### 3.2.1 Technical Feasibility
This project is technically feasible with the open-source Python stack already integrated into the codebase: `scikit-learn`, `fairlearn`, `interpret` (EBM), `xgboost`, `pandas`, `numpy`, `matplotlib`, and `scipy`. The entire workflow is CPU-based, so it can run comfortably on a regular student laptop.

#### 3.2.2 Economic Feasibility
The project has essentially no direct software cost because it relies on open-source tools, publicly available datasets, and existing personal hardware.

#### 3.2.3 Social Feasibility
The project is socially relevant because it focuses on fairness, transparency, and accountability in automated credit decisions, helping stakeholders make more informed and responsible model choices.

### 3.3 SYSTEM SPECIFICATION

#### 10.1 Hardware Specification
- Processor: Intel i5 / AMD Ryzen 5 (or equivalent)
- RAM: Minimum 8 GB (16 GB recommended for smoother multi-model runs)
- Storage: 5-20 GB free disk space (data, outputs, and plots)
- GPU: Not required

#### 10.2 Software Specification
- Operating System: Windows / Linux / macOS
- Programming Language: Python 3.8+
- Libraries:
	- scikit-learn
	- fairlearn
	- interpret
	- xgboost
	- pandas
	- numpy
	- matplotlib
	- scipy
