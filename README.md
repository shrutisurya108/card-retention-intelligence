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
| Explainability | SHAP (TreeExplainer) |
| Dashboard | Streamlit · Plotly |
| Deployment | Streamlit Community Cloud |

---

## Model Performance (Test Set · 2,026 held-out customers)

| Metric | Logistic Regression | XGBoost |
|---|---|---|
| ROC-AUC | 0.9370 | **0.9930** |
| PR-AUC | 0.7770 | **0.9680** |
| F1-Score | 0.6562 | **0.9159** |
| Recall | 0.8338 | **0.9046** |
| Accuracy | 0.8598 | **0.9733** |

**Business impact:** XGBoost correctly identifies 293 of 325 actual churners
in the test set → **$87,900 retention value per cycle** at $300 avg acquisition cost.

---

## Project Structure

```
card-retention-intelligence/
├── data/
│   ├── raw/                    # BankChurners.csv (not tracked)
│   └── processed/              # Ingested, engineered, train/test splits
├── database/                   # SQLite database (not tracked)
├── models/
│   ├── xgboost_model.pkl       # Production model
│   ├── logreg_model.pkl        # Baseline model
│   └── scaler.pkl              # Fitted StandardScaler
├── src/
│   ├── logger.py               # Shared loguru logger
│   ├── ingestion.py            # Phase 1: CSV → SQLite
│   ├── eda.py                  # Phase 2: EDA — 10 saved plots
│   ├── features.py             # Phase 2: Feature engineering
│   ├── train.py                # Phase 3: Model training + CV
│   ├── evaluate.py             # Phase 3: Metrics + evaluation plots
│   └── explain.py              # Phase 4: SHAP values + 4 plots
├── dashboard/
│   └── app.py                  # Phase 5: Streamlit app
├── tests/                      # 80+ pytest tests across 4 phases
├── outputs/
│   ├── figures/                # 19 saved plots (EDA + eval + SHAP)
│   ├── shap_values.npy         # Precomputed SHAP values (dashboard)
│   ├── shap_expected_value.npy # SHAP baseline (dashboard)
│   ├── shap_feature_names.json # Feature reference (dashboard)
│   ├── model_comparison.csv    # CV results
│   └── test_metrics.csv        # Final test-set metrics
├── logs/                       # Runtime logs (not tracked)
├── Makefile
├── run_pipeline.py
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
- [x] Phase 4 — SHAP Explainability
- [ ] Phase 5 — Streamlit Business Dashboard
- [ ] Phase 6 — Deployment (Streamlit Community Cloud)

---

## SHAP Explainability

SHAP (SHapley Additive exPlanations) answers the question retention teams
actually care about: *why* was this customer flagged as high risk?

Four plots covering three storytelling levels:

| Level | Plot | What it answers |
|---|---|---|
| Global | SHAP Summary (beeswarm) | Which features matter most, and in which direction? |
| Global | SHAP Importance Bar | Clean feature ranking by mean absolute SHAP |
| Local | SHAP Waterfall | Why was this specific customer flagged? |
| Interaction | SHAP Dependence | How does transaction count drive churn risk? |

**Key insight from SHAP:** Customers with fewer than ~40 annual transactions
show a sharp step-change in churn SHAP values — a threshold that directly
informs a business intervention rule.

---

## Modelling Decisions

**Why two models?** Logistic Regression establishes an interpretable baseline.
XGBoost is the production model. The gap (AUC 0.937 → 0.993) demonstrates that
non-linear patterns exist and are worth capturing.

**Why not SMOTE?** `class_weight='balanced'` and `scale_pos_weight` handle
imbalance without leakage risk from synthetic sample generation.

**Why ROC-AUC as primary metric?** With 16% churn, accuracy is misleading —
an all-"retained" model achieves 84% accuracy while being useless.

**Why TreeExplainer for SHAP?** Exact SHAP values in milliseconds vs
KernelExplainer which takes minutes and only approximates.

---

## License

MIT