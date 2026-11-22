"""Tests for the NumPy-only inference pipeline."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import (  # noqa: E402
    DEFAULT_MODEL_DIR,
    WEIGHT_KEYS,
    ChurnPredictor,
    get_predictor,
)


def test_artifacts_load(predictor: ChurnPredictor) -> None:
    assert predictor.input_size == 45
    assert predictor.architecture == [45, 32, 16, 1]
    assert 0.0 < predictor.threshold < 1.0
    assert set(WEIGHT_KEYS).issubset(predictor.weights)
    assert hasattr(predictor.preprocessor, "transform")


def test_weight_shapes(predictor: ChurnPredictor) -> None:
    expected = {
        "network.0.weight": (32, 45),
        "network.0.bias": (32,),
        "network.2.weight": (16, 32),
        "network.2.bias": (16,),
        "network.4.weight": (1, 16),
        "network.4.bias": (1,),
    }
    for name, shape in expected.items():
        assert predictor.weights[name].shape == shape
        assert predictor.weights[name].dtype == np.float32


def test_preprocessing_shape(predictor: ChurnPredictor, sample_customer: dict) -> None:
    matrix = predictor.preprocess(sample_customer)
    assert matrix.shape == (1, 45)
    assert matrix.dtype == np.float32
    assert np.isfinite(matrix).all()


def test_probability_range(predictor: ChurnPredictor, sample_customer: dict) -> None:
    probabilities = predictor.predict_proba(sample_customer)
    assert probabilities.shape == (1,)
    assert 0.0 <= float(probabilities[0]) <= 1.0


def test_probability_range_on_dataset_rows(
    predictor: ChurnPredictor, raw_dataset: pd.DataFrame | None
) -> None:
    if raw_dataset is None:
        pytest.skip("Raw dataset is not present under data/raw/.")

    probabilities = predictor.predict_proba(raw_dataset.iloc[:10])
    assert probabilities.shape == (10,)
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)


def test_prediction_uses_saved_threshold(
    predictor: ChurnPredictor, sample_customer: dict
) -> None:
    result = predictor.predict_churn(sample_customer)[0]
    expected_class = int(result["churn_probability"] >= predictor.threshold)
    assert result["threshold"] == predictor.threshold
    assert result["predicted_class"] == expected_class
    assert result["prediction"] == ("Churn" if expected_class else "No Churn")


def test_custom_threshold_changes_decision(
    predictor: ChurnPredictor, monkeypatch: pytest.MonkeyPatch, sample_customer: dict
) -> None:
    monkeypatch.setattr(predictor, "threshold", 0.0)
    assert predictor.predict_churn(sample_customer)[0]["prediction"] == "Churn"

    monkeypatch.setattr(predictor, "threshold", 1.0 - 1e-12)
    assert predictor.predict_churn(sample_customer)[0]["prediction"] == "No Churn"

    monkeypatch.setattr(predictor, "threshold", predictor.metadata["best_threshold"])


def test_dict_and_dataframe_inputs_match(
    predictor: ChurnPredictor, sample_customer: dict
) -> None:
    from_dict = predictor.predict_proba(sample_customer)
    from_frame = predictor.predict_proba(pd.DataFrame([sample_customer]))
    np.testing.assert_allclose(from_dict, from_frame, rtol=0, atol=0)


def test_batch_prediction(predictor: ChurnPredictor, sample_customer: dict) -> None:
    batch = [sample_customer, sample_customer]
    results = predictor.predict_churn(batch)
    assert len(results) == 2
    assert results[0] == results[1]


def test_missing_column_raises(
    predictor: ChurnPredictor, sample_customer: dict
) -> None:
    incomplete = dict(sample_customer)
    incomplete.pop("tenure")

    with pytest.raises(ValueError, match="Missing required columns"):
        predictor.predict_proba(incomplete)


def test_blank_total_charges_raises(
    predictor: ChurnPredictor, sample_customer: dict
) -> None:
    invalid = dict(sample_customer)
    invalid["TotalCharges"] = ""

    with pytest.raises(ValueError, match="Non-numeric or missing values"):
        predictor.predict_proba(invalid)


def test_non_numeric_value_raises(
    predictor: ChurnPredictor, sample_customer: dict
) -> None:
    invalid = dict(sample_customer)
    invalid["MonthlyCharges"] = "not-a-number"

    with pytest.raises(ValueError, match="Non-numeric or missing values"):
        predictor.predict_proba(invalid)


def test_missing_categorical_value_raises(
    predictor: ChurnPredictor, sample_customer: dict
) -> None:
    invalid = dict(sample_customer)
    invalid["Contract"] = None

    with pytest.raises(ValueError, match="Missing values in categorical columns"):
        predictor.predict_proba(invalid)


def test_empty_input_raises(predictor: ChurnPredictor) -> None:
    with pytest.raises(ValueError, match="at least one row"):
        predictor.predict_proba([])


def test_invalid_input_type_raises(predictor: ChurnPredictor) -> None:
    with pytest.raises(TypeError, match="customer_data must be"):
        predictor.predict_proba(42)


def test_preprocessor_resolves_for_customer_1(
    predictor: ChurnPredictor, raw_dataset: pd.DataFrame | None
) -> None:
    if raw_dataset is None:
        pytest.skip("Raw dataset is not present under data/raw/.")

    # The first CSV row is customer 7590-VHVEG.  We only assert internal
    # consistency here; the expected probability is not hard-coded.
    result = predictor.predict_churn(raw_dataset.iloc[[0]])[0]
    probability = result["churn_probability"]
    assert 0.0 <= probability <= 1.0
    assert result["predicted_class"] == int(probability >= predictor.threshold)


def test_inference_module_does_not_import_torch() -> None:
    code = (
        "import sys; "
        "import src.predict; "
        "assert 'torch' not in sys.modules, 'torch was imported'; "
        "print('torch-free import ok')"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "torch-free import ok" in completed.stdout


def test_unseen_category_warns_and_strict_mode_raises(
    predictor: ChurnPredictor, sample_customer: dict
) -> None:
    invalid = dict(sample_customer)
    invalid["Contract"] = "Unheard-of contract"

    with pytest.warns(UserWarning, match="Unseen categorical values"):
        predictor.preprocess(invalid)

    strict = ChurnPredictor(predictor.model_dir, strict_categories=True)
    with pytest.raises(ValueError, match="Unseen categorical values"):
        strict.preprocess(invalid)


def test_predictor_cache_reuses_unchanged_artifacts(tmp_path: Path) -> None:
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    for name in ("metadata.json", "preprocessor.pkl", "ann_weights.npz"):
        shutil.copy2(DEFAULT_MODEL_DIR / name, model_dir / name)

    first = get_predictor(model_dir)
    assert get_predictor(model_dir) is first


def test_predictor_cache_invalidates_when_artifacts_change(tmp_path: Path) -> None:
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    for name in ("metadata.json", "preprocessor.pkl", "ann_weights.npz"):
        shutil.copy2(DEFAULT_MODEL_DIR / name, model_dir / name)

    first = get_predictor(model_dir)
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    metadata["best_threshold"] = 0.42
    metadata["cache_bust_note"] = "artifacts replaced by a test"
    (model_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    second = get_predictor(model_dir)
    assert second is not first
    assert second.threshold == 0.42
