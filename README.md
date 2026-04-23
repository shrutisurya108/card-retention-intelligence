# ChurnShield 🛡️
### ML-Powered Customer Retention Intelligence Platform

> Predicts banking customer churn with XGBoost, explains decisions with SHAP,
> and delivers actionable retention recommendations via an interactive dashboard.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Overview

ChurnShield is an end-to-end machine learning pipeline that identifies at-risk
bank customers, quantifies business impact using ROI framing, and surfaces
SHAP-based explanations to help retention teams act on predictions — not just trust them.

---

## Dataset

**Source:** [Credit Card Customers — Kaggle (Sakshi Goyal)](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers)

| Property | Value |
|---|---|
| Domain | Banking / Credit Card |
| Rows | 10,127 customers |
| Features | 21 (after cleaning) |
| Target | Churn (16.07% attrition rate) |
| Class Imbalance | ~84% retained / ~16% churned |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data Ingestion | SQLite · SQLAlchemy · pandas |
| EDA & Features | pandas · seaborn · matplotlib |
| Modeling | XGBoost · Logistic Regression · scikit-learn |
| Explainability | SHAP |
| Dashboard | Streamlit · Plotly |
| Deployment | Streamlit Community Cloud |

---

## Project Structure

```
card-retention-intelligence/
│
├── .github/workflows/ci.yml       # GitHub Actions CI
├── data/
│   ├── raw/                       # BankChurners.csv (not committed)
│   └── processed/                 # Cleaned CSV after ingestion
├── database/                      # SQLite churn.db (not committed)
├── notebooks/                     # Exploratory scratch work
├── src/
│   ├── logger.py                  # Shared loguru logger
│   ├── ingestion.py               # Phase 1: CSV → SQLite
│   ├── eda.py                     # Phase 2: EDA plots
│   ├── features.py                # Phase 2: Feature engineering
│   ├── train.py                   # Phase 3: Model training
│   ├── evaluate.py                # Phase 3: Metrics & comparison
│   └── explain.py                 # Phase 4: SHAP explainability
├── models/                        # Saved model artifacts (.pkl)
├── dashboard/app.py               # Phase 5: Streamlit dashboard
├── tests/                         # pytest test suites per phase
├── logs/                          # Runtime logs (not committed)
├── outputs/figures/               # Saved plots
├── requirements.txt
├── run_pipeline.py                # Master pipeline runner
└── README.md
```

---

## Quickstart

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/card-retention-intelligence.git
cd card-retention-intelligence

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place BankChurners.csv in data/raw/
#    Download from: https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers

# 4. Run the full pipeline
python run_pipeline.py

# 5. Launch the dashboard
streamlit run dashboard/app.py
```

---

## Pipeline Phases

- [x] **Phase 0** — Project structure, dependencies, Git setup
- [x] **Phase 1** — Data ingestion: CSV → SQLite via SQLAlchemy
- [ ] **Phase 2** — EDA & feature engineering
- [ ] **Phase 3** — Model training & evaluation (XGBoost vs Logistic Regression)
- [ ] **Phase 4** — SHAP explainability
- [ ] **Phase 5** — Streamlit business dashboard
- [ ] **Phase 6** — Deployment to Streamlit Community Cloud

---

## Running Tests

```bash
pytest tests/ -v
```

---

## License

MIT
