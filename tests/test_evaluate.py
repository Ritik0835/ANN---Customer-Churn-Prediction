"""Integration test: evaluation script must reproduce the documented metrics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import evaluate as evaluate_script  # noqa: E402
from src.predict import DEFAULT_MODEL_DIR  # noqa: E402

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
METRICS_PATH = PROJECT_ROOT / "docs" / "metrics.json"
pytestmark = pytest.mark.skipif(
    not DATA_PATH.exists(), reason="Raw dataset is not present under data/raw/."
)


def test_evaluate_reproduces_documented_metrics() -> None:
    actual = evaluate_script.evaluate(DATA_PATH, DEFAULT_MODEL_DIR)
    expected = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    assert actual == expected
