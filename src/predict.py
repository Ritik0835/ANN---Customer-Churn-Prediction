"""NumPy-only inference for the Telco Customer Churn ANN.

This module deliberately does **not** import PyTorch.  It loads the NumPy
weights exported from the trained PyTorch model, the fitted scikit-learn
preprocessor, and the training metadata, then reproduces the model's forward
pass with NumPy.

The trained architecture is::

    45 inputs -> Linear(45, 32) -> ReLU -> Linear(32, 16) -> ReLU
              -> Linear(16, 1) -> Sigmoid

The output is a churn probability.  The classification threshold is read from
``metadata.json`` (``best_threshold``) and was selected on validation data
during training, not on the test set.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"

# Names used by PyTorch's state_dict.  Keeping the same names makes it easy to
# verify that ann_weights.npz is an exact export of ann_customer_churn.pth.
WEIGHT_KEYS: tuple[str, ...] = (
    "network.0.weight",
    "network.0.bias",
    "network.2.weight",
    "network.2.bias",
    "network.4.weight",
    "network.4.bias",
)


def _install_sklearn_pickle_compat() -> None:
    """Allow loading preprocessors pickled by scikit-learn 1.6 on newer versions.

    ``preprocessor.pkl`` was created with scikit-learn 1.6.1.  A private helper
    class used by ``ColumnTransformer`` in that release was removed in later
    releases.  The class only stored a list of remainder columns, so a minimal
    list subclass is enough to unpickle the estimator.  Newer scikit-learn
    versions already define the class, in which case this function does nothing.
    """

    try:
        from sklearn.compose import _column_transformer
    except ImportError:  # pragma: no cover - scikit-learn is a hard dependency
        return

    if not hasattr(_column_transformer, "_RemainderColsList"):

        class _RemainderColsList(list):
            def __init__(self, iterable=(), warn_on_update: bool = False):
                super().__init__(iterable)
                self._warn_on_update = warn_on_update

        _column_transformer._RemainderColsList = _RemainderColsList


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Compute a numerically stable sigmoid for a NumPy array."""

    values = np.asarray(values, dtype=np.float32)
    result = np.empty_like(values)

    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))

    negative = ~positive
    exp_values = np.exp(values[negative])
    result[negative] = exp_values / (1.0 + exp_values)

    return result


class ChurnPredictor:
    """Load the saved artifacts and make NumPy-only churn predictions.

    Parameters
    ----------
    model_dir:
        Directory containing ``ann_weights.npz``, ``preprocessor.pkl`` and
        ``metadata.json``.  Defaults to the repository's ``models/`` directory.
    """

    def __init__(self, model_dir: str | Path = DEFAULT_MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.metadata = self._load_metadata()
        self.preprocessor = self._load_preprocessor()
        self.weights = self._load_weights()

        self.input_size = int(self.metadata["input_size"])
        self.architecture = list(
            self.metadata.get("architecture", [self.input_size, 32, 16, 1])
        )
        self.threshold = float(self.metadata["best_threshold"])
        self.numerical_columns = list(self.metadata["numerical_columns"])
        self.categorical_columns = list(self.metadata["categorical_columns"])

        self._validate_configuration()

    # ------------------------------------------------------------------
    # Artifact loading
    # ------------------------------------------------------------------
    def _load_metadata(self) -> dict[str, Any]:
        path = self.model_dir / "metadata.json"
        if not path.exists():
            raise FileNotFoundError(f"Metadata file not found: {path}")

        with path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)

        required = {
            "input_size",
            "best_threshold",
            "numerical_columns",
            "categorical_columns",
        }
        missing = required - set(metadata)
        if missing:
            raise ValueError(
                f"metadata.json is missing required keys: {sorted(missing)}"
            )

        return metadata

    def _load_preprocessor(self):
        path = self.model_dir / "preprocessor.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Preprocessor file not found: {path}")

        _install_sklearn_pickle_compat()

        try:
            from sklearn.exceptions import InconsistentVersionWarning

            warning_filter: Any = InconsistentVersionWarning
        except ImportError:  # pragma: no cover - very old scikit-learn
            warning_filter = Warning

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", warning_filter)
            preprocessor = joblib.load(path)

        if not hasattr(preprocessor, "transform"):
            raise TypeError(
                "preprocessor.pkl does not contain a fitted transformer "
                "with a transform() method."
            )

        return preprocessor

    def _load_weights(self) -> dict[str, np.ndarray]:
        path = self.model_dir / "ann_weights.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"Weights file not found: {path}. "
                "Run scripts/export_weights.py to create it."
            )

        with np.load(path, allow_pickle=False) as archive:
            missing = [key for key in WEIGHT_KEYS if key not in archive]
            if missing:
                raise ValueError(
                    f"ann_weights.npz is missing arrays: {missing}"
                )
            weights = {
                key: np.asarray(archive[key], dtype=np.float32)
                for key in WEIGHT_KEYS
            }

        return weights

    def _validate_configuration(self) -> None:
        hidden_activation = str(
            self.metadata.get("hidden_activation", "ReLU")
        ).lower()
        if hidden_activation != "relu":
            raise ValueError(
                "This inference implementation supports the saved ReLU model; "
                f"metadata.json declares {hidden_activation!r}."
            )

        if not 0.0 < self.threshold < 1.0:
            raise ValueError(
                "best_threshold must be strictly between 0 and 1; "
                f"got {self.threshold}."
            )

        if len(self.architecture) != 4 or self.architecture[0] != self.input_size:
            raise ValueError(
                "architecture in metadata.json must be [input_size, hidden_1, "
                f"hidden_2, output_size]; got {self.architecture}."
            )

        _, hidden_1, hidden_2, output_size = self.architecture
        expected = {
            "network.0.weight": (hidden_1, self.input_size),
            "network.0.bias": (hidden_1,),
            "network.2.weight": (hidden_2, hidden_1),
            "network.2.bias": (hidden_2,),
            "network.4.weight": (output_size, hidden_2),
            "network.4.bias": (output_size,),
        }
        for name, shape in expected.items():
            if self.weights[name].shape != shape:
                raise ValueError(
                    f"Unexpected shape for {name}: "
                    f"{self.weights[name].shape}, expected {shape}. "
                    "Check input_size in metadata.json and the exported weights."
                )

    # ------------------------------------------------------------------
    # Input validation and preprocessing
    # ------------------------------------------------------------------
    def _as_dataframe(self, customer_data: Any) -> pd.DataFrame:
        if isinstance(customer_data, Mapping):
            frame = pd.DataFrame([customer_data])
        elif isinstance(customer_data, pd.DataFrame):
            frame = customer_data.copy()
        elif isinstance(customer_data, Sequence) and not isinstance(
            customer_data, (str, bytes)
        ):
            frame = pd.DataFrame(list(customer_data))
        else:
            raise TypeError(
                "customer_data must be a mapping, a pandas DataFrame, "
                "or a sequence of mappings."
            )

        if frame.empty:
            raise ValueError("customer_data must contain at least one row.")

        return frame

    def preprocess(self, customer_data: Any) -> np.ndarray:
        """Validate raw customer data and return the 45-column model matrix."""

        frame = self._as_dataframe(customer_data)
        required_columns = self.numerical_columns + self.categorical_columns

        missing = [column for column in required_columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        frame = frame.loc[:, required_columns].copy()

        # Numeric columns must be finite numbers.  Blank TotalCharges values
        # are a common problem in the Telco dataset.
        for column in self.numerical_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        numeric_missing = frame[self.numerical_columns].isna()
        if numeric_missing.any().any():
            bad_columns = sorted(
                column for column in self.numerical_columns
                if numeric_missing[column].any()
            )
            raise ValueError(
                "Non-numeric or missing values in numerical columns: "
                f"{bad_columns}. Clean the input before predicting."
            )

        categorical_missing = frame[self.categorical_columns].isna()
        if categorical_missing.any().any():
            bad_columns = sorted(
                column for column in self.categorical_columns
                if categorical_missing[column].any()
            )
            raise ValueError(
                "Missing values in categorical columns: "
                f"{bad_columns}. Clean the input before predicting."
            )

        transformed = self.preprocessor.transform(frame)

        # ColumnTransformer may return a sparse matrix depending on its
        # configuration; this saved pipeline returns a dense array, but the
        # conversion keeps the code robust.
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()

        matrix = np.asarray(transformed, dtype=np.float32)

        if matrix.ndim != 2 or matrix.shape[1] != self.input_size:
            raise ValueError(
                "Preprocessing produced an unexpected shape: "
                f"{matrix.shape}; expected (*, {self.input_size})."
            )

        if not np.isfinite(matrix).all():
            raise ValueError(
                "Preprocessing produced non-finite values. "
                "Check for invalid numeric input."
            )

        return matrix

    # ------------------------------------------------------------------
    # Forward pass and predictions
    # ------------------------------------------------------------------
    def predict_proba(self, customer_data: Any) -> np.ndarray:
        """Return churn probabilities in the range [0, 1]."""

        matrix = self.preprocess(customer_data)

        w1 = self.weights["network.0.weight"]
        b1 = self.weights["network.0.bias"]
        w2 = self.weights["network.2.weight"]
        b2 = self.weights["network.2.bias"]
        w3 = self.weights["network.4.weight"]
        b3 = self.weights["network.4.bias"]

        hidden_1 = np.maximum(matrix @ w1.T + b1, 0.0)
        hidden_2 = np.maximum(hidden_1 @ w2.T + b2, 0.0)
        logits = (hidden_2 @ w3.T + b3).ravel()

        return _sigmoid(logits)

    def predict(self, customer_data: Any) -> np.ndarray:
        """Return binary predictions (1 = churn, 0 = no churn)."""

        probabilities = self.predict_proba(customer_data)
        return (probabilities >= self.threshold).astype(int)

    def predict_churn(self, customer_data: Any) -> list[dict[str, Any]]:
        """Return one result dictionary per input row.

        Each dictionary contains the raw churn probability, the predicted
        class, and the threshold that was used.
        """

        probabilities = self.predict_proba(customer_data)
        predicted_classes = (probabilities >= self.threshold).astype(int)

        return [
            {
                "churn_probability": float(probability),
                "prediction": "Churn" if int(predicted) == 1 else "No Churn",
                "predicted_class": int(predicted),
                "threshold": self.threshold,
            }
            for probability, predicted in zip(probabilities, predicted_classes)
        ]


_PREDICTOR_CACHE: dict[Path, ChurnPredictor] = {}


def get_predictor(model_dir: str | Path = DEFAULT_MODEL_DIR) -> ChurnPredictor:
    """Return a cached predictor for ``model_dir``."""

    path = Path(model_dir).resolve()
    if path not in _PREDICTOR_CACHE:
        _PREDICTOR_CACHE[path] = ChurnPredictor(path)
    return _PREDICTOR_CACHE[path]


def predict_churn(
    customer_data: Any,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> list[dict[str, Any]]:
    """Convenience wrapper around :class:`ChurnPredictor`.

    Parameters
    ----------
    customer_data:
        A dictionary, a pandas DataFrame, or a sequence of dictionaries.
    model_dir:
        Directory containing the three saved artifacts.
    """

    return get_predictor(model_dir).predict_churn(customer_data)


if __name__ == "__main__":  # pragma: no cover - simple manual example
    import argparse

    parser = argparse.ArgumentParser(
        description="Predict churn for rows in a CSV file using NumPy only."
    )
    parser.add_argument("csv_path", type=Path, help="CSV file with customer rows")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Directory containing the saved model artifacts",
    )
    args = parser.parse_args()

    raw_customers = pd.read_csv(args.csv_path)
    results = predict_churn(raw_customers, args.model_dir)

    for row_number, result in enumerate(results, start=1):
        print(
            f"{row_number}: "
            f"probability={result['churn_probability']:.4f} "
            f"prediction={result['prediction']} "
            f"threshold={result['threshold']:.2f}"
        )
