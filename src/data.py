"""Shared dataset loading and splitting for the Telco churn project.

Keeping these steps in one place avoids the drift that comes from copying the
cleaning logic into the notebook, evaluation scripts, and tests.  The cleaning
behaviour matches the original notebook: blank ``TotalCharges`` values are
treated as missing and those rows are removed, and ``customerID`` is dropped.

Unlike the original copies, this loader validates that no *unexpected* missing
values or target labels are present instead of silently dropping rows.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"
RANDOM_STATE = 42
VALID_LABELS = {"No": 0, "Yes": 1}

__all__ = [
    "ID_COLUMN",
    "RANDOM_STATE",
    "TARGET_COLUMN",
    "VALID_LABELS",
    "load_clean_dataset",
    "split_train_test",
    "split_train_validation",
]


def load_clean_dataset(data_path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load the raw Telco CSV and return ``(features, target)``.

    Parameters
    ----------
    data_path:
        Path to ``WA_Fn-UseC_-Telco-Customer-Churn.csv`` (or any renamed copy).

    Raises
    ------
    ValueError
        If required columns are absent, unexpected missing values remain after
        cleaning, or the target contains labels other than ``No`` / ``Yes``.
    """

    path = Path(data_path)
    frame = pd.read_csv(path)

    if "TotalCharges" not in frame.columns:
        raise ValueError(f"{path} has no 'TotalCharges' column")
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"{path} has no '{TARGET_COLUMN}' column")

    frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")

    blank_total = frame["TotalCharges"].isna()
    if blank_total.any():
        frame = frame.loc[~blank_total].copy()

    if ID_COLUMN in frame.columns:
        frame = frame.drop(columns=[ID_COLUMN])

    missing = frame.isna().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        raise ValueError(
            "Unexpected missing values after cleaning TotalCharges: "
            f"{missing.to_dict()}"
        )

    labels = set(frame[TARGET_COLUMN].unique())
    unexpected = labels - set(VALID_LABELS)
    if unexpected:
        raise ValueError(
            f"Unexpected {TARGET_COLUMN!r} labels: {sorted(map(str, unexpected))}"
        )

    target = frame[TARGET_COLUMN].map(VALID_LABELS).astype(int).reset_index(drop=True)
    features = frame.drop(columns=[TARGET_COLUMN]).reset_index(drop=True)
    return features, target


def split_train_test(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return the stratified 80/20 split used by the notebook and scripts."""

    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )


def split_train_validation(
    features: pd.DataFrame,
    target: pd.Series,
    validation_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split a training frame into the fitting fold and validation fold."""

    return train_test_split(
        features,
        target,
        test_size=validation_size,
        random_state=random_state,
        stratify=target,
    )
