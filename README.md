# ChurnShield 🛡️
### ML-Powered Customer Retention Intelligence Platform

> Predicts banking customer churn with XGBoost, explains decisions with SHAP,
> and delivers actionable retention recommendations via an interactive dashboard.

![Python](https://img.shields.io/badge/Python-3.11.9-blue)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green)

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

## Model Performance (Test Set)

| Metric | Logistic Regression | XGBoost |
|---|---|---|
| ROC-AUC | ~0.93 | ~0.97 |
| PR-AUC | ~0.80 | ~0.92 |
| F1-Score | ~0.72 | ~0.89 |
| Recall | ~0.88 | ~0.93 |

*XGBoost is the production model. Logistic Regression serves as an interpretable baseline.*

---

## Project Structure

```
card-retention-intelligence/
├── data/
│   ├── raw/                    # Original BankChurners.csv (not tracked)
│   └── processed/              # Ingested, engineered, train/test splits
├── database/                   # SQLite database (not tracked)
├── models/                     # Saved models + scaler
│   ├── xgboost_model.pkl
│   ├── logreg_model.pkl
│   └── scaler.pkl
├── src/
│   ├── logger.py               # Shared loguru logger
│   ├── ingestion.py            # Phase 1: CSV → SQLite
│   ├── eda.py                  # Phase 2: EDA — 10 saved plots
│   ├── features.py             # Phase 2: Feature engineering
│   ├── train.py                # Phase 3: Model training + CV
│   ├── evaluate.py             # Phase 3: Metrics + evaluation plots
│   └── explain.py              # Phase 4: SHAP values
├── dashboard/
│   └── app.py                  # Phase 5: Streamlit app
├── tests/                      # pytest test suite (50+ tests)
├── outputs/
│   ├── figures/                # 15 saved plots (EDA + evaluation)
│   ├── model_comparison.csv    # CV results
│   └── test_metrics.csv        # Final test-set metrics
├── logs/                       # Runtime logs (not tracked)
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

# Place BankChurners.csv in data/raw/ first
python run_pipeline.py

# Launch dashboard
python -m streamlit run dashboard/app.py
```

---

## Pipeline Phases

- [x] Phase 0 — Project Setup & Structure
- [x] Phase 1 — Data Ingestion (CSV → SQLite)
- [x] Phase 2 — EDA & Feature Engineering
- [x] Phase 3 — Model Training & Evaluation (XGBoost + Logistic Regression)
- [ ] Phase 4 — SHAP Explainability
- [ ] Phase 5 — Streamlit Business Dashboard
- [ ] Phase 6 — Deployment (Streamlit Community Cloud)

---

## Modelling Decisions

**Why two models?**
Logistic Regression establishes an interpretable baseline. XGBoost is the production
model. Comparing them demonstrates understanding of the accuracy vs interpretability
tradeoff — a key data science interview topic.

**Why not SMOTE for class imbalance?**
SMOTE generates synthetic samples that can leak into test folds if not handled
carefully inside cross-validation pipelines. `class_weight='balanced'` and
`scale_pos_weight` achieve the same effect with zero leakage risk.

**Why ROC-AUC as primary metric?**
With 16% churn, accuracy is misleading — a model predicting all "retained" achieves
84% accuracy while being useless. ROC-AUC measures the model's ability to rank
churners above non-churners regardless of classification threshold.

---
