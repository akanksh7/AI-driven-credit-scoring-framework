GLOBAL_RANDOM_SEED = 42

DATASETS = {
    "australian": {
        "path": "data/australian.dat",
        "target": "approved",
        "protected": "A1",
    },
    "german": {
        "path": "data/german.data",
        "target": "credit_risk",
        "protected": "sex",
    },
    "gmsc": {
        "path": "data/gmsc.csv",
        "target": "SeriousDlqin2yrs",
        "protected": "age_group",
    },
}

FAIRNESS_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]

TEST_SIZE = 0.2
CV_FOLDS = 5

MAX_ALLOWED_ACCURACY_DROP = 0.05
TARGET_FAIRNESS_DP = None

OUTPUT_DIR = "outputs"
PLOTS_DIR = "outputs/plots"
RESULTS_PATH = "outputs/results.csv"
SUBGROUP_METRICS_PATH = "outputs/subgroup_metrics.csv"
PAIRED_TESTS_PATH = "outputs/paired_tests.csv"
