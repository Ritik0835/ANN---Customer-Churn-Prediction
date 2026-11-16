# Data

The project uses the **IBM Telco Customer Churn** dataset (also known as the
Kaggle "Telco Customer Churn" dataset).

- Source: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- File: `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- Expected local path: `data/raw/Telco-Customer-Churn.csv`
- Rows: 7,043
- Columns: 21
- 11 rows with blank `TotalCharges` are removed by the notebook, leaving 7,032
  rows.

The CSV is **not committed** to this repository. Download it from Kaggle and
place it at the path above, or set `TELCO_DATA_PATH` to its location when
running scripts.

The dataset contains historical customer records. It is used here for
educational purposes, not for production decisions.
