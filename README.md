# Card-Retention-Intelligence 🛡️
### ML-Powered Customer Retention Intelligence Platform

> Predicts banking customer churn with XGBoost, explains decisions with SHAP,
> and delivers actionable retention recommendations via an interactive dashboard.

![Python](https://img.shields.io/badge/Python-3.11.9-blue)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Overview

Card-Retention-Intelligence is an end-to-end machine learning pipeline that identifies at-risk
bank customers, quantifies business impact using ROI framing, and surfaces
SHAP-based explanations to help retention teams act on predictions — not just trust them.

**Dataset:** [Credit Card Customers — Kaggle](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers)
- 10,127 bank customers · 21 features · ~16% churn rate

---

## Live Dashboard

> 🚀 **[Launch Card-Retention-Intelligence Dashboard](https://card-retention-intelligence-pipeline.streamlit.app)**

5 interactive pages:
| Page | What it does |
|---|---|
| 📊 Executive Summary | KPI cards, ROI calculator, model comparison chart |
| 🎯 Customer Risk Scorer | Live churn prediction + SHAP waterfall for any customer |
| 📈 Model Performance | All evaluation plots, CV results, metrics table |
| 🔍 SHAP Explainability | Global + local + interaction SHAP plots |
| 🔬 EDA Explorer | Interactive histograms, all EDA plots, raw data preview |

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
→ **$87,900 retention value per cycle** at $300 avg acquisition cost.

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
│   └── app.py                  # Phase 5: 5-page Streamlit app
├── tests/                      # 110+ pytest tests across 5 phases
├── outputs/
│   ├── figures/                # 19 PNG plots (EDA + eval + SHAP)
│   ├── shap_values.npy         # Precomputed SHAP (dashboard speed)
│   ├── shap_expected_value.npy # SHAP baseline
│   ├── shap_feature_names.json # Feature reference
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

# 1. Place BankChurners.csv in data/raw/
# 2. Run full pipeline
python run_pipeline.py

# 3. Launch dashboard
python -m streamlit run dashboard/app.py
```

---

## Pipeline Phases

- [x] Phase 0 — Project Setup & Structure
- [x] Phase 1 — Data Ingestion (CSV → SQLite)
- [x] Phase 2 — EDA & Feature Engineering
- [x] Phase 3 — Model Training & Evaluation (XGBoost + Logistic Regression)
- [x] Phase 4 — SHAP Explainability
- [x] Phase 5 — Streamlit Business Dashboard
- [x] Phase 6 — Deployment (Streamlit Community Cloud)

---

## Key Design Decisions

**Why two models?** XGBoost is production. Logistic Regression is the interpretable
baseline. The AUC gap (0.937 → 0.993) proves non-linear patterns exist and matter.

**Why not SMOTE?** `class_weight` and `scale_pos_weight` handle imbalance without
leakage risk from synthetic sample generation inside CV folds.

**Why TreeExplainer for SHAP?** Exact values in milliseconds. KernelExplainer
approximates and takes minutes.

**Why precompute SHAP values?** Dashboard responsiveness. Loading `.npy` takes
milliseconds; recomputing for 2,026 samples takes 30 seconds.

**Why Stratified K-Fold?** With 16% churn, regular KFold risks putting all churners
in one fold. Stratified guarantees each fold reflects the true class distribution.

---

## Collaboration and Acknowledgements
This project was built and developed in collaboration with [Harshith Bhattaram](https://github.com/maniharshith68).
 
 
## 👤 Authors
- [Harshith Bhattaram](https://github.com/maniharshith68)
- [Shruti Kumari](https://github.com/shrutisurya108)
 
---

