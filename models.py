from typing import Dict

from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from config import GLOBAL_RANDOM_SEED


def get_models() -> Dict[str, object]:
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=GLOBAL_RANDOM_SEED),
        "DecisionTree": DecisionTreeClassifier(random_state=GLOBAL_RANDOM_SEED),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            random_state=GLOBAL_RANDOM_SEED,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(random_state=GLOBAL_RANDOM_SEED),
        "EBM": ExplainableBoostingClassifier(random_state=GLOBAL_RANDOM_SEED),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=GLOBAL_RANDOM_SEED,
            n_jobs=-1,
        ),
    }
