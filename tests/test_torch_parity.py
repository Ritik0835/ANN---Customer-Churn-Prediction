"""Optional PyTorch parity tests.

These tests are skipped automatically when PyTorch is not installed, because
PyTorch is not an inference dependency.  Install ``requirements-train.txt`` to
run them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import ChurnPredictor, WEIGHT_KEYS  # noqa: E402

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"


class CustomerChurnANN(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, inputs):
        return self.network(inputs)


@pytest.fixture(scope="session")
def torch_model(predictor: ChurnPredictor):
    model = CustomerChurnANN(predictor.input_size)
    state_dict = torch.load(
        PROJECT_ROOT / "models" / "ann_customer_churn.pth",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model, state_dict


def _torch_probabilities(model, matrix: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        logits = model(torch.tensor(matrix, dtype=torch.float32))
        return torch.sigmoid(logits).cpu().numpy().ravel()


def test_exported_weights_match_pytorch(
    predictor: ChurnPredictor, torch_model
) -> None:
    _, state_dict = torch_model
    for key in WEIGHT_KEYS:
        torch_weight = state_dict[key].detach().cpu().numpy()
        np.testing.assert_array_equal(predictor.weights[key], torch_weight)


def test_numpy_and_pytorch_match_on_generic_customer(
    predictor: ChurnPredictor, torch_model, sample_customer: dict
) -> None:
    model, _ = torch_model
    matrix = predictor.preprocess(sample_customer)
    numpy_probabilities = predictor.predict_proba(sample_customer)
    torch_probabilities = _torch_probabilities(model, matrix)
    np.testing.assert_allclose(
        numpy_probabilities, torch_probabilities, rtol=1e-6, atol=1e-6
    )


def test_numpy_and_pytorch_match_on_test_set(
    predictor: ChurnPredictor, torch_model
) -> None:
    if not DATA_PATH.exists():
        pytest.skip("Raw dataset is not present under data/raw/.")

    frame = pd.read_csv(DATA_PATH)
    frame["TotalCharges"] = pd.to_numeric(
        frame["TotalCharges"], errors="coerce"
    )
    frame = frame.dropna().drop(columns=["customerID"]).reset_index(drop=True)
    features = frame.drop(columns=["Churn"])
    target = frame["Churn"].map({"No": 0, "Yes": 1})

    _, x_test, _, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    model, _ = torch_model
    matrix = predictor.preprocess(x_test)
    numpy_probabilities = predictor.predict_proba(x_test)
    torch_probabilities = _torch_probabilities(model, matrix)

    maximum_difference = float(
        np.max(np.abs(numpy_probabilities - torch_probabilities))
    )
    assert maximum_difference < 1e-6

    numpy_predictions = (numpy_probabilities >= predictor.threshold).astype(int)
    torch_predictions = (torch_probabilities >= predictor.threshold).astype(int)
    assert np.array_equal(numpy_predictions, torch_predictions)
