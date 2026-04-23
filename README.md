# Card retention Intelligence 🛡️
### ML-Powered Customer Retention Intelligence Platform

> Predicts banking customer churn with XGBoost, explains decisions with SHAP, 
> and delivers actionable retention recommendations via an interactive dashboard.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Overview
Card retention Intelligence is an end-to-end machine learning pipeline that identifies at-risk.
bank customers, quantifies business impact using ROI framing, and surfaces.
SHAP-based explanations to help retention teams act on predictionsm.

## Tech Stack
- **Data Ingestion:** SQL (SQLite via SQLAlchemy)
- **EDA & Features:** pandas, seaborn, matplotlib
- **Modeling:** XGBoost, Logistic Regression, scikit-learn
- **Explainability:** SHAP
- **Dashboard:** Streamlit + Plotly
- **Deployment:** Streamlit Community Cloud

## Project Structure
[See directory tree above]

## Quickstart
\```bash
git clone https://github.com/shrutisurya108/card-retention-intelligence.git
cd card-retention-intelligence
pip install -r requirements.txt
python run_pipeline.py
streamlit run dashboard/app.py
\```

## Phases
- [x] Phase 0: Project Setup
- [ ] Phase 1: Data Ingestion
- [ ] Phase 2: EDA & Feature Engineering
- [ ] Phase 3: Model Training & Evaluation
- [ ] Phase 4: SHAP Explainability
- [ ] Phase 5: Streamlit Dashboard
- [ ] Phase 6: Deployment