# Datasets

The four benchmark datasets are publicly available and are **not redistributed**
here. Download them from their original sources and place them as follows
(original file names, one folder per dataset):

| Dataset | Source | Expected path | Target |
|---|---|---|---|
| Bank Customer Churn | Kaggle: *Bank Customer Churn Prediction* | `data/Bank Customer Churn Prediction/Bank Customer Churn Prediction.csv` | `churn` (1 = churned) |
| Bank Marketing | UCI Machine Learning Repository: *Bank Marketing* (`bank.csv`, the 10% subset) | `data/bank-marketing-uci/bank.csv` | `y` ("yes") |
| HR Promotion | Kaggle / Analytics Vidhya: *HR Analytics: Employee Promotion* (`train.csv`) | `data/HR Analytics Employee Promotion Data/train.csv` | `is_promoted` (1) |
| Loan Approval | Analytics Vidhya: *Loan Prediction Problem* (`train_u6lujuX_CVtuZ9i.csv`) | `data/Loan Prediction Problem Dataset/train_u6lujuX_CVtuZ9i.csv` | `Loan_Status` ("Y") |

Notes

* Bank Marketing is the semicolon-separated `bank.csv` (4,521 records), not
  `bank-full.csv` or `bank-additional-full.csv`. The `duration` column is
  removed before modelling because it is known only after the call ends.
* Identifier columns (`customer_id`, `employee_id`, `Loan_ID`) are dropped.
* To keep the data elsewhere, change `DATA_ROOT` in the second code cell of
  each notebook.
