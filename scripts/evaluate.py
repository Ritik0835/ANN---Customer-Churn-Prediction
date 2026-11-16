"""Reproduce the held-out test metrics for the saved churn model.

This script does not train anything.  It rebuilds the same stratified
train/test split used in the notebook, loads the saved artifacts with the
NumPy-only predictor, and reports the final test metrics.

Usage
-----
    python scripts/evaluate.py
    TELCO_DATA_PATH=/path/to/Telco-Customer-Churn.csv python scripts/evaluate.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import ChurnPredictor, DEFAULT_MODEL_DIR  # noqa: E402

DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
RANDOM_STATE = 42


def load_clean_dataset(data_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Apply the same cleaning steps as the training notebook."""

    frame = pd.read_csv(data_path)
    frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")
    frame = frame.dropna().drop(columns=["customerID"]).reset_index(drop=True)

    features = frame.drop(columns=["Churn"])
    target = frame["Churn"].map({"No": 0, "Yes": 1})
    return features, target


def evaluate(data_path: Path, model_dir: Path) -> dict:
    features, target = load_clean_dataset(data_path)

    # This is the exact split from the notebook.  The test set is never used
    # for training or threshold selection.
    _, x_test, _, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    predictor = ChurnPredictor(model_dir)
    probabilities = predictor.predict_proba(x_test)
    predictions = (probabilities >= predictor.threshold).astype(int)

    metrics = {
        "threshold": float(predictor.threshold),
        "test_samples": int(len(y_test)),
        "positive_test_samples": int(y_test.sum()),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision": float(
            average_precision_score(y_test, probabilities)
        ),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(os.environ.get("TELCO_DATA_PATH", DEFAULT_DATA_PATH)),
        help="Path to the raw Telco CSV file",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Directory containing the saved artifacts",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path for writing the metrics as JSON",
    )
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(
            f"Dataset not found: {args.data}. "
            "Download the IBM Telco Customer Churn CSV or set TELCO_DATA_PATH."
        )

    metrics = evaluate(args.data, args.model_dir)

    print(json.dumps(metrics, indent=2))
    print("\nConfusion matrix:")
    print(np.asarray(metrics["confusion_matrix"]))

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=2)
        print(f"\nWrote metrics to {args.json_out}")


if __name__ == "__main__":
    main()
