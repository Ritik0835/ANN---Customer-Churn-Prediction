# Customer Churn Prediction with an Artificial Neural Network

An end-to-end customer churn prediction project built with PyTorch, with a
**PyTorch-free NumPy inference path** for lightweight deployment.

The model estimates the probability that a telecom customer will churn. It was
trained on the IBM Telco Customer Churn dataset and exported as a small,
self-contained set of inference artifacts.

> **Educational project.** The dataset contains historical customer records.
> This model is intended for learning and portfolio demonstration, not for
> production retention decisions.

## Results

Metrics were computed on a held-out test set of **1,407 customers** (374
churned) using the artifacts committed in `models/`. The decision threshold was
selected on a separate validation split, never on the test set.

| Metric | Value |
| --- | ---: |
| Decision threshold | 0.30 |
| Accuracy | 0.7619 |
| Precision | 0.5377 |
| Recall | 0.7433 |
| F1 score | 0.6240 |
| ROC-AUC | 0.8343 |
| Average precision | 0.6245 |

Confusion matrix (rows = actual, columns = predicted):

|  | No Churn | Churn |
| --- | ---: | ---: |
| **No Churn** | 794 | 239 |
| **Churn** | 96 | 278 |

ROC-AUC and average precision are threshold-independent and measure the ranking
quality of the predicted probabilities. The other metrics use the selected
threshold of 0.30.

The same numbers are stored in [`docs/metrics.json`](docs/metrics.json) and can
be reproduced with `python scripts/evaluate.py`.

### Baseline comparison

`python scripts/baseline.py` trains two simple models on the same 4,500-row
fitting fold and evaluates the same 1,407-row test set. The baselines use their
default 0.50 decision threshold; the ANN uses the validation-selected 0.30.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Majority class | 0.7342 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |
| Logistic regression | 0.7989 | 0.6350 | 0.5722 | 0.6020 | 0.8352 |
| ANN (this project) | 0.7619 | 0.5377 | 0.7433 | 0.6240 | 0.8343 |

The linear model and the ANN have nearly identical ranking quality (ROC-AUC),
but the ANN's lower threshold trades precision for substantially higher recall.
This comparison is a sanity check, not a claim that the ANN is universally
better.

## Highlights

- **PyTorch-free inference.** `src/predict.py` loads the exported NumPy weights
  and performs the forward pass with NumPy only. Importing it does not import
  PyTorch, which is enforced by a test.
- **Validation-only threshold selection.** The threshold search covers
  `0.20` to `0.70` in steps of `0.05` and maximizes validation F1. The test set
  is untouched until the final evaluation.
- **Exact weight export.** `models/ann_weights.npz` uses the same tensor names
  as the PyTorch `state_dict`. The exported weights match the `.pth` checkpoint
  exactly, and NumPy/PyTorch probabilities agree to within `1.192e-07` on the
  test set.
- **Reproducible workflow.** A cleaned notebook, evaluation script, weight
  export utility, backend comparison script, and pytest suite are included.
- **Honest artifact handling.** The committed model artifacts are from a later
  training run than the original notebook checkpoint. See
  [Historical output note](#historical-output-note).

## Dataset

**IBM Telco Customer Churn** (Kaggle "Telco Customer Churn"):

- Source: <https://www.kaggle.com/datasets/blastchar/telco-customer-churn>
- Expected local path: `data/raw/Telco-Customer-Churn.csv`
- Original size: 7,043 rows and 21 columns
- Cleaning: 11 rows with blank `TotalCharges` are removed, and `customerID` is
  dropped, leaving 7,032 rows
- Target: `Churn` (`Yes` = 1, `No` = 0)
- Split: 5,625 training rows / 1,407 test rows, stratified
- Validation: the training portion is split again into 4,500 training rows and
  1,125 validation rows

The CSV is **not committed** to this repository. Download it from Kaggle and
place it at the path above, or set `TELCO_DATA_PATH` when running scripts. See
[`data/README.md`](data/README.md).

## Model

```
45 inputs
  -> Linear(45, 32) -> ReLU
  -> Linear(32, 16) -> ReLU
  -> Linear(16, 1)
  -> sigmoid -> churn probability
```

- **Loss:** `BCEWithLogitsLoss`
- **Optimizer:** Adam, learning rate `0.001`
- **Batch size:** 64
- **Epochs:** up to 50 with early stopping (patience 5)
- **Seed:** `42` for Python, NumPy, and PyTorch
- **Decision rule:** `churn_probability >= 0.30`

Preprocessing uses a scikit-learn `ColumnTransformer`:

- `StandardScaler` for `SeniorCitizen`, `tenure`, `MonthlyCharges`, and
  `TotalCharges`
- `OneHotEncoder(handle_unknown="ignore")` for the 15 categorical columns

The fitted preprocessor is saved as `models/preprocessor.pkl`.

## Repository structure

```text
.
├── data/
│   ├── README.md
│   ├── raw/                 # dataset (gitignored)
│   └── processed/           # generated data (gitignored)
├── docs/
│   └── metrics.json         # verified held-out test metrics
├── models/
│   ├── ann_customer_churn.pth   # original PyTorch state_dict
│   ├── ann_weights.npz          # NumPy export used for inference
│   ├── metadata.json            # architecture, features, threshold
│   └── preprocessor.pkl         # fitted scikit-learn pipeline
├── notebooks/
│   └── ann-customer-churn-prediction.ipynb
├── scripts/
│   ├── baseline.py          # majority-class and logistic-regression baselines
│   ├── compare_backends.py  # optional NumPy vs. PyTorch parity check
│   ├── evaluate.py          # reproduces the held-out test metrics
│   └── export_weights.py    # .pth -> .npz maintenance utility
├── src/
│   ├── __init__.py
│   ├── data.py              # shared dataset loading and splitting
│   └── predict.py           # NumPy-only inference
├── tests/
│   ├── conftest.py
│   ├── test_predict.py
│   └── test_torch_parity.py
├── requirements.txt         # inference dependencies (no PyTorch)
├── requirements-train.txt   # notebook/training dependencies
├── requirements-dev.txt     # test dependencies
└── .github/workflows/ci.yml # pytest on push and pull request
```

## Installation

Clone the repository and create a virtual environment from the repository root:

```bash
git clone https://github.com/Ritik0835/ANN---Customer-Churn-Prediction.git
cd ANN---Customer-Churn-Prediction

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### Inference only (no PyTorch)

```bash
pip install -r requirements.txt
```

This installs NumPy, pandas, scikit-learn, and joblib. It deliberately does
**not** install PyTorch. Version upper bounds are intentionally conservative
because `preprocessor.pkl` depends on scikit-learn internals; the suite was
verified here with NumPy 2.5.5, pandas 3.1.1, scikit-learn 1.9.1, and joblib
1.6.0.

### Notebook and training

```bash
pip install -r requirements-train.txt
```

This adds PyTorch, matplotlib, seaborn, and Jupyter.

### Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

The PyTorch parity tests are skipped automatically when PyTorch is not
installed.

## Usage

### Python API

```python
from src.predict import predict_churn

customer = {
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

result = predict_churn(customer)[0]
print(result)
# {'churn_probability': ..., 'prediction': '...', 'predicted_class': ..., 'threshold': 0.3}
```

`predict_churn` accepts a dictionary, a pandas `DataFrame`, or a sequence of
dictionaries. Use `ChurnPredictor.predict_proba(...)` when you only need
probabilities. By default, categorical values unseen during preprocessing
training produce a warning and are encoded as all zeros; pass
`strict_categories=True` to `ChurnPredictor` or `get_predictor` to raise a
`ValueError` instead. The predictor cache is keyed by artifact modification
times and sizes, so replacing model files in a long-running process does not
serve stale predictions.

### Command line

```bash
python src/predict.py data/raw/Telco-Customer-Churn.csv
```

Optional arguments:

```bash
python src/predict.py path/to/customers.csv --model-dir models
```

The CLI prints one line per row with the probability, prediction, and threshold.

## Reproducing the results

Run these commands from the repository root:

```bash
# Rebuild the held-out split and evaluate the committed artifacts
python scripts/evaluate.py

# Optional: compare NumPy inference with the original PyTorch checkpoint
python scripts/compare_backends.py

# Optional: compare the ANN with simple baselines
python scripts/baseline.py

# Run the test suite
pytest -q
```

Expected evaluation output:

```text
Threshold: 0.30
Test samples: 1407
Accuracy: 0.7619
Precision: 0.5377
Recall: 0.7433
F1: 0.6240
ROC-AUC: 0.8343
Average precision: 0.6245
Confusion matrix:
[[794 239]
 [ 96 278]]
```

The notebook also contains a verification section that evaluates the committed
artifacts through the NumPy predictor. Running the notebook's training cells
retrains the model and rewrites the files in `models/`; the metrics above
describe the currently committed artifacts.

The notebook now splits the training portion into fitting and validation folds
**before** fitting the preprocessor, so validation rows no longer influence
scaler statistics or one-hot categories. The committed artifacts were produced
before this correction and are intentionally left unchanged; a fresh notebook
run will produce a new, internally consistent artifact set.

## NumPy vs. PyTorch parity

`scripts/compare_backends.py` checks the exported weights and the forward pass
against the original PyTorch model:

```text
Weight keys exported: 6
Exported weights match PyTorch exactly.
Customer 1 probability (PyTorch): 0.5180
Customer 1 probability (NumPy):   0.5180
Max |PyTorch - NumPy| on test set: 1.192e-07
Prediction disagreements at threshold 0.30: 0 / 1407
```

The small probability difference comes from floating-point accumulation order;
all 1,407 test-set classifications agree at the saved threshold.

## Artifacts

| File | Description |
| --- | --- |
| `models/ann_customer_churn.pth` | Original PyTorch `state_dict` |
| `models/ann_weights.npz` | NumPy export with matching tensor names |
| `models/preprocessor.pkl` | Fitted `ColumnTransformer` |
| `models/metadata.json` | Architecture, feature columns, threshold, best epoch |

`preprocessor.pkl` was created with scikit-learn 1.6.1. `src/predict.py`
includes a small compatibility shim so the pickle can also be loaded with newer
scikit-learn releases. The shim only restores a private list-like helper used by
`ColumnTransformer`; it does not change the preprocessing behavior.

## Historical output note

Older notebook output or screenshots from a different training run may show
different probabilities. The committed artifacts (`models/ann_customer_churn.pth`
and `models/ann_weights.npz`) are the source of truth: the first customer
currently scores `0.5180` (`Churn`), as shown by the notebook verification cell
and by `scripts/compare_backends.py`.

The weights were not retrained or modified to reproduce any older value. If you
need results for a different run, retrain the notebook and update the artifacts
together.

## Testing

The current test suite covers:

- artifact loading and configuration validation
- weight shapes and dtypes
- preprocessing output shape and finite values
- probability ranges and threshold behavior
- dictionary, DataFrame, and batch inputs
- missing, blank, non-numeric, and unseen-category input handling
- a check that importing `src.predict` does not import PyTorch
- optional PyTorch parity tests
- centralized data cleaning/splitting and strict row/label validation
- predictor-cache reuse and invalidation when artifacts change
- an integration check that `scripts/evaluate.py` reproduces `docs/metrics.json`

Verified results in this workspace:

- With PyTorch installed: **28 passed**
- Without PyTorch: **25 passed, 1 skipped** (the parity module is skipped)

CI runs the suite on every push and pull request via
[`.github/workflows/ci.yml`](.github/workflows/ci.yml).

## Limitations

- The dataset is small and comes from a single historical snapshot.
- Metrics come from one stratified split; there is still no cross-validation or
  confidence interval. `scripts/baseline.py` provides simple model comparisons,
  but not variance estimates.
- The network is small and uncalibrated. Probabilities should not be interpreted
  as precise real-world risk.
- The threshold maximizes F1, which may not match the business cost of false
  positives versus false negatives.
- The model uses historical customer attributes and is not suitable for
  production decisions without further validation, fairness review, and
  monitoring.

## License

No license file is currently included in this repository. Add one before
redistributing or reusing the project.
