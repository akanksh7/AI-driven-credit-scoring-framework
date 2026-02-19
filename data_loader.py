from dataclasses import dataclass
from typing import List

import pandas as pd

from config import DATASETS


@dataclass
class DatasetSpec:
    name: str
    target: str
    protected: str
    data: pd.DataFrame


def _load_german(path: str) -> pd.DataFrame:
    columns = [
        "status",
        "duration",
        "credit_history",
        "purpose",
        "credit_amount",
        "savings",
        "employment",
        "installment_rate",
        "personal_status",
        "other_debtors",
        "residence_since",
        "property",
        "age",
        "other_installment_plans",
        "housing",
        "existing_credits",
        "job",
        "num_liable",
        "telephone",
        "foreign_worker",
        "credit_risk",
    ]
    df = pd.read_csv(path, sep=r"\s+", header=None, names=columns)
    df["credit_risk"] = (df["credit_risk"] == 1).astype(int)
    df["sex"] = df["personal_status"].map(
        {
            "A91": "male",
            "A93": "male",
            "A94": "male",
            "A92": "female",
            "A95": "female",
        }
    )
    return df


def _load_australian(path: str) -> pd.DataFrame:
    columns = [f"A{i}" for i in range(1, 15)] + ["approved"]
    df = pd.read_csv(path, sep=r"\s+", header=None, names=columns)
    df["approved"] = df["approved"].astype(int)
    return df


def _load_gmsc(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.columns[0] == "Unnamed: 0" or df.columns[0] == "":
        df = df.drop(columns=[df.columns[0]])
    df["age_group"] = (df["age"] >= 45).map({True: "older", False: "younger"})
    return df


def load_datasets() -> List[DatasetSpec]:
    datasets: List[DatasetSpec] = []
    for name, spec in DATASETS.items():
        path = spec["path"]
        try:
            if name == "australian":
                df = _load_australian(path)
            elif name == "german":
                df = _load_german(path)
            elif name == "gmsc":
                df = _load_gmsc(path)
            else:
                continue
        except FileNotFoundError:
            continue

        datasets.append(
            DatasetSpec(
                name=name,
                target=spec["target"],
                protected=spec["protected"],
                data=df,
            )
        )

    return datasets
