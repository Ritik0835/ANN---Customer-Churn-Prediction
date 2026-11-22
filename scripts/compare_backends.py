"""Compare the NumPy inference path with the original PyTorch model.

The comparison is run on the held-out test set and on the first five dataset
rows.  It verifies both the exported weights and the final probabilities.

This script is optional: the project's inference requirements do not include
PyTorch.  Install ``requirements-train.txt`` to run it.

Usage
-----
    python scripts/compare_backends.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_clean_dataset, split_train_test  # noqa: E402
from src.predict import ChurnPredictor, DEFAULT_MODEL_DIR, WEIGHT_KEYS  # noqa: E402

DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"


def build_torch_model(input_size: int):
    import torch.nn as nn

    class CustomerChurnANN(nn.Module):
        def __init__(self, features: int):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(features, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            )

        def forward(self, inputs):
            return self.network(inputs)

    return CustomerChurnANN(input_size)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(os.environ.get("TELCO_DATA_PATH", DEFAULT_DATA_PATH)),
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    args = parser.parse_args()

    try:
        import torch
    except ImportError as error:  # pragma: no cover - optional dependency
        raise SystemExit(
            "PyTorch is needed for this comparison. "
            "Install the training dependencies first."
        ) from error

    predictor = ChurnPredictor(args.model_dir)

    state_dict = torch.load(
        args.model_dir / "ann_customer_churn.pth",
        map_location="cpu",
        weights_only=True,
    )

    print("Weight comparison")
    print("-" * 60)
    for key in WEIGHT_KEYS:
        numpy_weight = predictor.weights[key]
        torch_weight = state_dict[key].detach().cpu().numpy()
        if not np.array_equal(numpy_weight, torch_weight):
            maximum = float(np.max(np.abs(numpy_weight - torch_weight)))
            raise SystemExit(f"Mismatch in {key}: max abs diff = {maximum}")
        print(f"  {key:20s} exact match, shape={numpy_weight.shape}")

    model = build_torch_model(predictor.input_size)
    model.load_state_dict(state_dict)
    model.eval()

    features, target = load_clean_dataset(args.data)
    _, x_test, _, _ = split_train_test(features, target)

    matrices = {
        "customer_1": predictor.preprocess(features.iloc[[0]]),
        "first_five": predictor.preprocess(features.iloc[:5]),
        "test_set": predictor.preprocess(x_test),
    }

    results = {}
    for name, matrix in matrices.items():
        with torch.no_grad():
            logits = model(torch.tensor(matrix, dtype=torch.float32))
            torch_probabilities = torch.sigmoid(logits).cpu().numpy().ravel()
        numpy_probabilities = predictor.predict_proba(
            features.iloc[[0]]
            if name == "customer_1"
            else features.iloc[:5]
            if name == "first_five"
            else x_test
        )

        absolute_difference = np.abs(torch_probabilities - numpy_probabilities)
        results[name] = {
            "rows": int(len(numpy_probabilities)),
            "max_abs_diff": float(absolute_difference.max()),
            "mean_abs_diff": float(absolute_difference.mean()),
        }
        print(f"\n{name}")
        print(f"  rows: {len(numpy_probabilities)}")
        print(f"  max |PyTorch - NumPy|: {absolute_difference.max():.3e}")
        print(f"  mean |PyTorch - NumPy|: {absolute_difference.mean():.3e}")
        print(f"  first-row NumPy probability: {numpy_probabilities[0]:.6f}")
        print(f"  first-row PyTorch probability: {torch_probabilities[0]:.6f}")

    # Class decisions must agree at the saved threshold.
    with torch.no_grad():
        logits = model(
            torch.tensor(matrices["test_set"], dtype=torch.float32)
        )
        torch_test_probabilities = torch.sigmoid(logits).cpu().numpy().ravel()
    numpy_test_probabilities = predictor.predict_proba(x_test)
    torch_predictions = (torch_test_probabilities >= predictor.threshold).astype(int)
    numpy_predictions = (numpy_test_probabilities >= predictor.threshold).astype(int)
    disagreements = int(np.sum(torch_predictions != numpy_predictions))

    print("\nClassification agreement on the test set")
    print(f"  threshold: {predictor.threshold}")
    print(f"  disagreements: {disagreements} / {len(numpy_predictions)}")

    summary = {
        "weight_export_exact": True,
        "threshold": predictor.threshold,
        "comparisons": results,
        "test_prediction_disagreements": disagreements,
    }
    print("\nSummary")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
