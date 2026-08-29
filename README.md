# Bank Marketing Classification

A machine learning project that predicts whether a bank customer is likely to subscribe to a term deposit.

## Problem

The goal is to identify customers who are more likely to subscribe to a bank term deposit using customer and campaign information available before a marketing call.

## Dataset

The project uses the Bank Marketing dataset containing 45,211 customer records and 17 columns.

Target variable:

- `y` — whether the customer subscribed to a term deposit (`yes` / `no`)

## Important Data Leakage Finding

The feature `duration` was found to contain strong predictive information.

However, `duration` represents the length of the contact and is only known after the interaction has taken place. Using it for pre-call customer targeting would therefore create data leakage.

Two models were evaluated:

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| With `duration` | ~0.930 | ~0.617 |
| Without `duration` | ~0.801 | ~0.453 |

The final deployment model excludes `duration`.

## Final Model

The deployment model uses a tuned Random Forest classifier with preprocessing integrated into the saved pipeline.

The selected decision threshold is:

`0.55`

The model returns:

- Prediction: `yes` or `no`
- Subscription probability
- Decision threshold

## Important Features

The deployment model's most important features were:

1. `balance`
2. `month`
3. `age`
4. `day`
5. `job`
6. `poutcome`
7. `contact`
8. `campaign`
9. `housing`
10. `pdays`

Feature importance indicates which variables the model relies on for prediction. It does not establish causal relationships.

## Project Structure

```text
Bank_Marketing_Classification/
│
├── data/
│   └── raw/
│       └── bank-full.csv
│
├── models/
│   └── bank_marketing_model.joblib
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_preprocessing_baseline.ipynb
│   └── 03_modeling_tuning.ipynb
│
├── reports/
│   └── baseline_model_results.csv
│
├── src/
│   ├── api.py
│   ├── model.py
│   └── __init__.py
│
├── tests/
│   ├── sample_customer.json
│   └── test_api.py
│
├── .gitignore
├── requirements.txt
└── README.md