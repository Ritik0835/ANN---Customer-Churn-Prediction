"""Tests for the centralized dataset loading and splitting helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (  # noqa: E402
    load_clean_dataset,
    split_train_test,
    split_train_validation,
)

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
pytestmark = pytest.mark.skipif(
    not DATA_PATH.exists(), reason="Raw dataset is not present under data/raw/."
)


def test_load_clean_dataset_shape_and_labels() -> None:
    features, target = load_clean_dataset(DATA_PATH)

    assert features.shape == (7032, 19)
    assert "customerID" not in features.columns
    assert "Churn" not in features.columns
    assert target.shape == (7032,)
    assert set(target.unique()) == {0, 1}
    assert target.mean() == pytest.approx(1869 / 7032, abs=1e-12)


def test_load_clean_dataset_rejects_unexpected_missing_values(
    tmp_path: Path,
) -> None:
    frame = pd.read_csv(DATA_PATH)
    frame.loc[0, "gender"] = None
    corrupted = tmp_path / "corrupted.csv"
    frame.to_csv(corrupted, index=False)

    with pytest.raises(ValueError, match="Unexpected missing values"):
        load_clean_dataset(corrupted)


def test_splits_are_reproducible_and_stratified() -> None:
    features, target = load_clean_dataset(DATA_PATH)

    x_train_1, x_test_1, y_train_1, y_test_1 = split_train_test(features, target)
    x_train_2, x_test_2, y_train_2, y_test_2 = split_train_test(features, target)

    pd.testing.assert_frame_equal(x_train_1, x_train_2)
    pd.testing.assert_frame_equal(x_test_1, x_test_2)
    pd.testing.assert_series_equal(y_train_1, y_train_2)
    pd.testing.assert_series_equal(y_test_1, y_test_2)
    assert len(x_test_1) == 1407
    assert y_test_1.mean() == pytest.approx(y_train_1.mean(), abs=0.01)

    x_fit, x_val, y_fit, y_val = split_train_validation(x_train_1, y_train_1)
    assert len(x_fit) == 4500
    assert len(x_val) == 1125
    assert y_fit.mean() == pytest.approx(y_val.mean(), abs=0.01)


def test_load_clean_dataset_rejects_unknown_target_label(tmp_path: Path) -> None:
    frame = pd.read_csv(DATA_PATH)
    frame.loc[0, "Churn"] = "Maybe"
    corrupted = tmp_path / "bad_labels.csv"
    frame.to_csv(corrupted, index=False)

    with pytest.raises(ValueError, match="Unexpected .* labels"):
        load_clean_dataset(corrupted)
