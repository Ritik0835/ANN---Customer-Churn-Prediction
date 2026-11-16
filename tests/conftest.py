"""Shared pytest fixtures for the churn prediction test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import ChurnPredictor  # noqa: E402

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"


@pytest.fixture(scope="session")
def predictor() -> ChurnPredictor:
    return ChurnPredictor()


@pytest.fixture
def sample_customer() -> dict:
    """A valid, generic customer used for input/range tests.

    This is deliberately an arbitrary customer, not the first dataset row, so
    tests do not depend on a hard-coded expected churn prediction.
    """

    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "One year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Mailed check",
        "MonthlyCharges": 50.0,
        "TotalCharges": 600.0,
    }


@pytest.fixture(scope="session")
def raw_dataset() -> pd.DataFrame | None:
    if not DATA_PATH.exists():
        return None
    return pd.read_csv(DATA_PATH)
