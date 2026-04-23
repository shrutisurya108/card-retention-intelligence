# ChurnShield 🛡️
### ML-Powered Customer Retention Intelligence Platform

> Predicts banking customer churn with XGBoost, explains decisions with SHAP,
> and delivers actionable retention recommendations via an interactive dashboard.

![Python](https://img.shields.io/badge/Python-3.11.9-blue)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Overview

ChurnShield is an end-to-end machine learning pipeline that identifies at-risk
bank customers, quantifies business impact using ROI framing, and surfaces
SHAP-based explanations to help retention teams act on predictions — not just trust them.

**Dataset:** [Credit Card Customers — Kaggle](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers)
- 10,127 bank customers · 21 features · ~16% churn rate
- Features: credit limit, transaction count, utilisation ratio, contact frequency, and more

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data Ingestion | SQLite · SQLAlchemy · pandas |
| EDA & Features | pandas · seaborn · matplotlib |
| Modelling | XGBoost · Logistic Regression · scikit-learn |
| Explainability | SHAP |
| Dashboard | Streamlit · Plotly |
| Deployment | Streamlit Community Cloud |

---

## Project Structure

```
card-retention-intelligence/
├── data/
│   ├── raw/                    # Original BankChurners.csv (not tracked)
│   └── processed/              # Cleaned + feature-engineered data
├── database/                   # SQLite database (not tracked)
├── notebooks/                  # Exploratory scratch work
├── src/
│   ├── logger.py               # Shared loguru logger
│   ├── ingestion.py            # Phase 1: CSV → SQLite
│   ├── eda.py                  # Phase 2: EDA — 10 saved plots
│   ├── features.py             # Phase 2: Feature engineering
│   ├── train.py                # Phase 3: Model training
│   ├── evaluate.py             # Phase 3: Metrics + comparison
│   └── explain.py              # Phase 4: SHAP values
├── models/                     # Saved model artifacts + scaler
├── dashboard/
│   └── app.py                  # Phase 5: Streamlit app
├── tests/                      # pytest test suite
├── logs/                       # Runtime logs (not tracked)
├── outputs/figures/            # Saved EDA plots (10 PNGs)
├── Makefile                    # Convenience commands
├── run_pipeline.py             # Master orchestrator
└── requirements.txt
```

---

## Quickstart

```bash
git clone https://github.com/YOUR_USERNAME/card-retention-intelligence.git
cd card-retention-intelligence
pip install -r requirements.txt

# Place BankChurners.csv in data/raw/ first (download from Kaggle link above)
python run_pipeline.py

# Launch dashboard (Phase 5)
python -m streamlit run dashboard/app.py
```

---

## Pipeline Phases

- [x] Phase 0 — Project Setup & Structure
- [x] Phase 1 — Data Ingestion (CSV → SQLite)
- [x] Phase 2 — EDA & Feature Engineering
- [ ] Phase 3 — Model Training & Evaluation (XGBoost + Logistic Regression)
- [ ] Phase 4 — SHAP Explainability
- [ ] Phase 5 — Streamlit Business Dashboard
- [ ] Phase 6 — Deployment (Streamlit Community Cloud)

---

## EDA Outputs

Ten plots saved to `outputs/figures/`:

| Plot | File |
|---|---|
| Churn class distribution | `churn_distribution.png` |
| Age distribution by churn | `age_distribution.png` |
| Churn rate by age group | `churn_by_age.png` |
| Credit limit (raw vs log) | `credit_limit_dist.png` |
| Transaction count vs churn | `trans_count_vs_churn.png` |
| Feature correlation heatmap | `correlation_heatmap.png` |
| Churn rate by segment | `churn_by_category.png` |
| All numeric distributions | `numeric_distributions.png` |
| Utilisation ratio vs churn | `utilisation_vs_churn.png` |
| Contact frequency vs churn | `contacts_vs_churn.png` |

---

## Feature Engineering Decisions

| Transformation | Reason |
|---|---|
| Drop `CLIENTNUM` | ID column — no predictive value |
| Binary encode `Gender` | Simple M/F mapping |
| Ordinal encode `Education_Level`, `Income_Category` | Natural order exists |
| One-hot encode `Marital_Status`, `Card_Category` | No natural order |
| Log-transform `Credit_Limit`, `Avg_Open_To_Buy`, `Total_Trans_Amt` | Right-skewed (confirmed in EDA) |
| Engineer `transaction_velocity` | Avg spend per transaction — compound signal |
| Engineer `inactivity_risk` | Inactive × contacts — frustration signal |
| Engineer `credit_usage_gap` | Actual credit used — cleaner than raw utilisation |
| StandardScaler on all numeric | Required for logistic regression |

---

## Dataset Notes

The raw `BankChurners.csv` contains two Naive Bayes classifier output columns
injected by the original dataset author. These are dropped during ingestion to
prevent **data leakage**.

---
