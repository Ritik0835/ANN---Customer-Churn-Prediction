"""Compare the committed ANN with simple baselines on the same held-out split.

This script trains a majority-class dummy model and a regularized logistic
regression model on the 4,500-row fitting fold used by the ANN, evaluates both
on the 1,407-row test set, and prints the same metric names as
``scripts/evaluate.py``.  It does not train or modify the neural network.

Usage
-----
    python scripts/baseline.py
    TELCO_DATA_PATH=/path/to/Telco-Customer-Churn.csv python scripts/baseline.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (  # noqa: E402
    load_clean_dataset,
    split_train_test,
    split_train_validation,
)

DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"


def _make_preprocessor(features) -> ColumnTransformer:
    numerical = features.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical = features.select_dtypes(exclude=["number"]).columns.tolist()
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )


def _metrics(y_true, probabilities, threshold: float = 0.5) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(os.environ.get("TELCO_DATA_PATH", DEFAULT_DATA_PATH)),
        help="Path to the raw Telco CSV file",
    )
    args = parser.parse_args()

    features, target = load_clean_dataset(args.data)
    x_train, x_test, y_train, y_test = split_train_test(features, target)
    x_fit, _, y_fit, _ = split_train_validation(x_train, y_train)

    models = {
        "majority_class": DummyClassifier(strategy="prior"),
        "logistic_regression": Pipeline(
            [
                ("preprocess", _make_preprocessor(x_fit)),
                ("model", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        ),
    }

    results: dict[str, dict[str, float]] = {}
    for name, model in models.items():
        model.fit(x_fit, y_fit)
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(x_test)[:, 1]
        else:  # pragma: no cover - both chosen models expose predict_proba
            probabilities = model.predict(x_test).astype(float)
        results[name] = _metrics(y_test, probabilities)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
