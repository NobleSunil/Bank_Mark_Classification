# Bank Marketing Classification

An end-to-end machine learning project that predicts whether a bank customer is likely to subscribe to a term deposit.

The project covers the complete ML workflow:

- Data understanding and quality analysis
- Exploratory Data Analysis (EDA)
- Preprocessing
- Baseline modeling
- Model comparison
- Cross-validation
- Hyperparameter tuning
- Threshold optimization
- Feature importance analysis
- FastAPI deployment
- Automated API testing
- Docker containerization
- Git/GitHub version control

---

## Problem

The goal is to identify customers who are more likely to subscribe to a bank term deposit using customer and marketing campaign information.

This can help a bank prioritize customers for marketing campaigns instead of treating every customer equally.

The problem is formulated as a binary classification task:

- `yes` → customer subscribed
- `no` → customer did not subscribe

---

## Dataset

The project uses the **Bank Marketing dataset** containing:

- **45,211 records**
- **17 columns**
- **16 input features**
- **1 target variable**

Target variable:

```text
y
```

Target distribution:

| Class | Count | Percentage |
|---|---:|---:|
| no | 39,922 | 88.30% |
| yes | 5,289 | 11.70% |

The dataset is therefore highly imbalanced, which makes accuracy alone an insufficient evaluation metric.

---

## Data Understanding & EDA

The dataset contains numerical and categorical variables describing:

- Customer demographics
- Financial information
- Housing/loan status
- Contact information
- Previous campaign outcomes
- Current campaign information

Important observations from EDA included:

- `poutcome` contains a large number of `unknown` values.
- `contact` also contains many `unknown` values.
- Numerical variables such as `balance`, `duration`, `campaign`, and `previous` contain highly skewed distributions.
- Customers who subscribed showed substantially higher average contact duration than customers who did not.
- Subscription rates varied across categories such as `job` and `education`.

`unknown` values were treated as meaningful categorical values rather than automatically converting them to missing values.

---

## Data Leakage Investigation

One of the most important findings in the project was **target leakage caused by `duration`**.

`duration` represents the length of the marketing contact.

Although it is highly predictive of the final outcome, it is only known **after the customer interaction has taken place**.

Therefore, using it for pre-call customer targeting would not be valid.

Two versions of the model were evaluated:

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| With `duration` | ~0.930 | ~0.617 |
| Without `duration` | ~0.801 | ~0.453 |

The large performance difference demonstrates why understanding the real-world meaning of features is critical in machine learning.

### Final deployment decision

`duration` was excluded from the deployment model.

This makes the model suitable for predicting customer propensity **before the marketing interaction**.

---

## Preprocessing

The preprocessing pipeline handles both numerical and categorical features.

Categorical features are encoded before being passed to the model, while numerical features are handled appropriately through the preprocessing pipeline.

The preprocessing steps are integrated with the model so that the same transformations are automatically applied during inference.

This reduces the risk of training-serving inconsistencies.

---

## Model Development

Several stages were used to develop the final model:

1. Establish a baseline.
2. Compare model performance.
3. Evaluate using cross-validation.
4. Tune the selected model.
5. Optimize the classification threshold.
6. Analyze feature importance.
7. Save the final model pipeline.

The final model is a tuned:

**Random Forest Classifier**

Best hyperparameters:

```text
n_estimators = 500
min_samples_split = 5
min_samples_leaf = 2
max_features = sqrt
max_depth = None
```

Best cross-validation PR-AUC:

```text
0.6117
```

---

## Model Evaluation

Because the target is imbalanced, the project focuses on metrics beyond accuracy.

The final model with `duration` excluded achieved approximately:

| Metric | Score |
|---|---:|
| Accuracy | 0.861 |
| Precision | 0.426 |
| Recall | 0.542 |
| F1 Score | 0.477 |
| ROC-AUC | 0.801 |
| PR-AUC | 0.453 |

The lower PR-AUC compared with the model containing `duration` reflects the difficulty of making a prediction using only information available before the interaction.

---

## Threshold Optimization

The default classification threshold of `0.50` was not automatically assumed to be optimal.

Different probability thresholds were evaluated to understand the trade-off between:

- Precision
- Recall
- F1 Score

For the model containing `duration`, a threshold of:

```text
0.55
```

provided the best F1 score in the evaluated threshold range:

```text
Precision: 0.5257
Recall:    0.7731
F1:        0.6258
```

The deployed model therefore uses a configurable decision threshold rather than blindly relying on `0.50`.

---

## Feature Importance

Feature importance was analyzed to understand which variables the Random Forest relied on most.

The most important features in the analyzed model included:

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

Feature importance indicates which variables contribute strongly to model predictions.

It does **not** establish a causal relationship between a feature and customer subscription.

---

# API Deployment

The trained model is exposed through a REST API built with **FastAPI**.

### Available endpoints

#### Health check

```http
GET /
```

Returns:

```json
{
  "message": "Bank Marketing Prediction API is running"
}
```

#### API health

```http
GET /health
```

Returns:

```json
{
  "status": "healthy"
}
```

#### Prediction

```http
POST /predict
```

Example request:

```json
{
  "age": 58,
  "job": "management",
  "marital": "married",
  "education": "tertiary",
  "default": "no",
  "balance": 2143,
  "housing": "yes",
  "loan": "no",
  "contact": "unknown",
  "day": 5,
  "month": "may",
  "campaign": 1,
  "pdays": -1,
  "previous": 0,
  "poutcome": "unknown"
}
```

Example response:

```json
{
  "prediction": "no",
  "probability": 0.1345,
  "threshold": 0.55
}
```

---

# Testing

The API is tested using `pytest` and FastAPI's test client.

Current test coverage includes:

- Root endpoint
- Health endpoint
- Prediction endpoint

Test result:

```text
3 passed
```

Run tests with:

```bash
pytest -v
```

---

# Docker

The API is containerized using Docker.

The Docker image installs only the dependencies required to run the API rather than the complete Jupyter/data-science development environment.

Build the image:

```bash
docker build -t bank-marketing-api .
```

Run the container:

```bash
docker run -p 8000:8000 bank-marketing-api
```

The API will then be available at:

```text
http://localhost:8000
```

---

# Local Development

## 1. Clone the repository

```bash
git clone https://github.com/NobleSunil/Bank_Mark_Classification.git
cd Bank_Mark_Classification
```

## 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Run the API

```bash
uvicorn src.api:app --reload
```

The API will run at:

```text
http://localhost:8000
```

FastAPI documentation is available at:

```text
http://localhost:8000/docs
```

---

# Project Structure

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
│   ├── __init__.py
│   ├── api.py
│   └── model.py
│
├── tests/
│   ├── sample_customer.json
│   └── test_api.py
│
├── .dockerignore
├── .gitignore
├── Dockerfile
├── requirements.txt
└── README.md
```

> The trained `.joblib` model is excluded from the Git repository because GitHub has a 100 MB per-file limit. The deployment model artifact is required locally to run the API.

---

# Key Engineering Decisions

### 1. Accuracy was not treated as the primary metric

The target variable is imbalanced, with only about 11.7% positive examples.

Therefore, Precision, Recall, F1, ROC-AUC and especially PR-AUC were considered alongside accuracy.

### 2. Data leakage was explicitly investigated

`duration` produced strong predictive performance but is unavailable before the customer interaction.

It was therefore excluded from the deployment scenario.

### 3. Threshold was optimized

Instead of assuming:

```text
probability >= 0.50 → yes
```

multiple thresholds were evaluated to find a better balance between precision and recall.

### 4. Preprocessing and model were packaged together

The saved model artifact contains the preprocessing pipeline and classifier, allowing inference to use the same transformations as training.

### 5. The model was tested before deployment

The API was tested using automated tests before containerization.

---

# Technologies Used

### Machine Learning

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib

### API

- FastAPI
- Pydantic
- Uvicorn

### Testing

- Pytest

### Deployment

- Docker

### Development

- Jupyter Notebook
- Git
- GitHub

---

# What This Project Demonstrates

This project demonstrates an end-to-end machine learning workflow rather than only model training.

The workflow goes from:

```text
Raw Dataset
     ↓
Data Understanding
     ↓
EDA
     ↓
Data Leakage Investigation
     ↓
Preprocessing
     ↓
Baseline Model
     ↓
Model Comparison
     ↓
Cross Validation
     ↓
Hyperparameter Tuning
     ↓
Evaluation
     ↓
Threshold Optimization
     ↓
Feature Importance
     ↓
Model Serialization
     ↓
FastAPI
     ↓
Automated Testing
     ↓
Docker
     ↓
GitHub
```

The main objective was to build a model that is not only accurate on a dataset, but also follows a realistic machine learning deployment workflow.
