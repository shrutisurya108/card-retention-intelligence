"""
app.py
------
Phase 5 — Card-Retention-Intelligence Streamlit Dashboard

5-page interactive business intelligence application:
  Page 1 — Executive Summary      : KPIs, ROI calculator, overview charts
  Page 2 — Customer Risk Scorer   : Live churn prediction + SHAP waterfall
  Page 3 — Model Performance      : All evaluation plots + metrics table
  Page 4 — SHAP Explainability    : All SHAP plots + feature importance table
  Page 5 — EDA Explorer           : All EDA plots + dataset statistics

Run with:
    python -m streamlit run dashboard/app.py

Fixes applied:
  - st.image() uses use_column_width=True (compatible with Streamlit 1.35.0)
  - All CSS uses explicit dark-mode-safe colours — no Streamlit theme variables
  - Plotly charts use transparent backgrounds with explicit font colours
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import shap
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).resolve().parent.parent
MODELS_DIR  = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = ROOT_DIR / "outputs" / "figures"
DATA_DIR    = ROOT_DIR / "data" / "processed"

# ── Colour palette ────────────────────────────────────────────────────────────
CLR_CHURN  = "#E74C3C"
CLR_RETAIN = "#2ECC71"
CLR_XGB    = "#2C3E50"
CLR_LR     = "#E74C3C"
CLR_ACCENT = "#3498DB"


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  — must be first Streamlit call
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Card-Retention-Intelligence — Customer Retention Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════════════════════════
# CACHED LOADERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_models():
    xgb    = joblib.load(MODELS_DIR / "xgboost_model.pkl")
    logreg = joblib.load(MODELS_DIR / "logreg_model.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    return xgb, logreg, scaler


@st.cache_resource
def load_shap_explainer():
    xgb, _, _ = load_models()
    return shap.TreeExplainer(xgb)


@st.cache_data
def load_shap_artifacts():
    shap_values    = np.load(OUTPUTS_DIR / "shap_values.npy")
    expected_value = float(np.load(OUTPUTS_DIR / "shap_expected_value.npy").flat[0])
    with open(OUTPUTS_DIR / "shap_feature_names.json") as f:
        feature_names = json.load(f)
    return shap_values, expected_value, feature_names


@st.cache_data
def load_metrics():
    return pd.read_csv(OUTPUTS_DIR / "test_metrics.csv")


@st.cache_data
def load_cv_comparison():
    return pd.read_csv(OUTPUTS_DIR / "model_comparison.csv")


@st.cache_data
def load_ingested_data():
    return pd.read_csv(DATA_DIR / "customers_ingested.csv")


def load_figure(filename: str):
    path = FIGURES_DIR / filename
    if path.exists():
        return Image.open(path)
    return None


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS — dark-mode-safe explicit colours
# ══════════════════════════════════════════════════════════════════════════════

def inject_css():
    st.markdown("""
    <style>

    /* KPI cards */
    .kpi-card {
        background: #1E2A38;
        border-radius: 12px;
        padding: 18px 22px;
        border-left: 5px solid #3498DB;
        margin-bottom: 14px;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: #FFFFFF;
        margin: 0;
        line-height: 1.2;
    }
    .kpi-label {
        font-size: 0.78rem;
        color: #A0AEC0;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin: 0 0 4px 0;
    }
    .kpi-delta {
        font-size: 0.82rem;
        color: #68D391;
        margin-top: 5px;
    }

    /* Risk badges */
    .risk-high {
        background: #2D1B1B;
        color: #FC8181;
        border: 2px solid #E74C3C;
        border-radius: 10px;
        padding: 14px 20px;
        font-size: 1.6rem;
        font-weight: 700;
        text-align: center;
        line-height: 1.5;
    }
    .risk-medium {
        background: #2D2516;
        color: #F6AD55;
        border: 2px solid #F39C12;
        border-radius: 10px;
        padding: 14px 20px;
        font-size: 1.6rem;
        font-weight: 700;
        text-align: center;
        line-height: 1.5;
    }
    .risk-low {
        background: #1A2D1E;
        color: #68D391;
        border: 2px solid #2ECC71;
        border-radius: 10px;
        padding: 14px 20px;
        font-size: 1.6rem;
        font-weight: 700;
        text-align: center;
        line-height: 1.5;
    }

    /* Section headers */
    .section-header {
        font-size: 1.05rem;
        font-weight: 600;
        color: #90CDF4;
        border-bottom: 2px solid #3498DB;
        padding-bottom: 5px;
        margin: 18px 0 12px 0;
    }

    /* Insight box */
    .insight-box {
        background: #1A2744;
        border-left: 4px solid #3498DB;
        border-radius: 6px;
        padding: 12px 16px;
        margin: 10px 0;
        font-size: 0.9rem;
        color: #BEE3F8;
        line-height: 1.6;
    }

    /* Recommendation card */
    .rec-card {
        background: #1B2A1B;
        border-radius: 10px;
        padding: 12px 16px;
        margin: 8px 0;
        border-left: 4px solid #F39C12;
        color: #E2E8F0;
        font-size: 0.9rem;
        line-height: 1.5;
    }
    .rec-card b { color: #F6AD55; }

    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }

    </style>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPER — Streamlit 1.35.0 compatible image display
# ══════════════════════════════════════════════════════════════════════════════

def show_image(img):
    """
    use_column_width=True works in Streamlit 1.35.0.
    use_container_width was only added in Streamlit 1.36+.
    """
    if img is not None:
        st.image(img, use_column_width=True)
    else:
        st.warning("Plot not found. Run `python run_pipeline.py` first.")


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.markdown("# 🛡️ Card-Retention-Intelligence")
        st.markdown("*Customer Retention Intelligence*")
        st.markdown("---")

        page = st.radio(
            "Navigation",
            options=[
                "📊 Executive Summary",
                "🎯 Customer Risk Scorer",
                "📈 Model Performance",
                "🔍 SHAP Explainability",
                "🔬 EDA Explorer",
            ],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("**Dataset**")
        st.markdown("10,127 bank customers")
        st.markdown("~16% churn rate")
        st.markdown("")
        st.markdown("**Production Model**")
        st.markdown("XGBoost · AUC 0.9930")
        st.markdown("---")
        st.markdown(
            "<small style='color:#718096'>Built with Python · XGBoost · SHAP · Streamlit</small>",
            unsafe_allow_html=True
        )

    return page


# ══════════════════════════════════════════════════════════════════════════════
# PLOTLY LAYOUT DEFAULTS — dark-mode-safe transparent background
# ══════════════════════════════════════════════════════════════════════════════

PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#E2E8F0"),
)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def page_executive_summary():
    st.title("📊 Executive Summary")
    st.markdown("*Card-Retention-Intelligence — ML-Powered Customer Retention Intelligence Platform*")

    metrics_df  = load_metrics()
    df_raw      = load_ingested_data()
    xgb_metrics = metrics_df[metrics_df["model"] == "XGBoost"].iloc[0]
    lr_metrics  = metrics_df[metrics_df["model"] == "LogisticRegression"].iloc[0]

    total_customers = len(df_raw)
    churned         = int(df_raw["churn"].sum())
    churn_rate      = df_raw["churn"].mean()
    xgb_caught      = int(xgb_metrics["recall"] * churned * 0.20)
    retention_value = xgb_caught * 300

    # ── KPI Cards ─────────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">Key Performance Indicators</p>',
                unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color:#2ECC71">
          <p class="kpi-label">XGBoost ROC-AUC</p>
          <p class="kpi-value">{xgb_metrics['roc_auc']:.4f}</p>
          <p class="kpi-delta">↑ +{xgb_metrics['roc_auc']-lr_metrics['roc_auc']:.4f} vs baseline</p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color:#E74C3C">
          <p class="kpi-label">Customers at Risk</p>
          <p class="kpi-value">{churned:,}</p>
          <p class="kpi-delta">{churn_rate*100:.1f}% of portfolio</p>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color:#F39C12">
          <p class="kpi-label">Est. Retention Value</p>
          <p class="kpi-value">${retention_value:,}</p>
          <p class="kpi-delta">Per cycle · $300 avg CAC</p>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color:#3498DB">
          <p class="kpi-label">Model Recall</p>
          <p class="kpi-value">{xgb_metrics['recall']*100:.1f}%</p>
          <p class="kpi-delta">Churners correctly identified</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Charts ────────────────────────────────────────────────────────────────
    col_left, col_right = st.columns([1, 1.6])

    with col_left:
        st.markdown('<p class="section-header">Churn Distribution</p>',
                    unsafe_allow_html=True)
        fig_donut = go.Figure(go.Pie(
            labels=["Retained", "Churned"],
            values=[total_customers - churned, churned],
            hole=0.55,
            marker_colors=[CLR_RETAIN, CLR_CHURN],
            textinfo="label+percent",
            textfont=dict(size=13, color="#E2E8F0"),
            hovertemplate="%{label}: %{value:,}<br>%{percent}<extra></extra>",
        ))
        fig_donut.update_layout(
            **PLOTLY_BASE,
            showlegend=False, height=280,
            margin=dict(t=20, b=20, l=20, r=20),
            annotations=[dict(
                text=f"<b>{total_customers:,}</b><br>customers",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="#E2E8F0")
            )],
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_right:
        st.markdown('<p class="section-header">Model Comparison — Key Metrics</p>',
                    unsafe_allow_html=True)
        m_keys   = ["roc_auc", "pr_auc", "f1", "recall", "precision"]
        m_labels = ["ROC-AUC", "PR-AUC", "F1", "Recall", "Precision"]
        fig_bar  = go.Figure()
        fig_bar.add_trace(go.Bar(
            name="Logistic Regression", x=m_labels,
            y=[lr_metrics[m] for m in m_keys],
            marker_color=CLR_LR, opacity=0.82,
            text=[f"{lr_metrics[m]:.3f}" for m in m_keys],
            textposition="outside", textfont=dict(size=10, color="#E2E8F0"),
        ))
        fig_bar.add_trace(go.Bar(
            name="XGBoost", x=m_labels,
            y=[xgb_metrics[m] for m in m_keys],
            marker_color=CLR_ACCENT, opacity=0.85,
            text=[f"{xgb_metrics[m]:.3f}" for m in m_keys],
            textposition="outside", textfont=dict(size=10, color="#E2E8F0"),
        ))
        fig_bar.update_layout(
            **PLOTLY_BASE,
            barmode="group", height=280,
            yaxis=dict(range=[0, 1.15], title="Score",
                       gridcolor="#2D3748", color="#A0AEC0"),
            xaxis=dict(color="#A0AEC0"),
            margin=dict(t=10, b=10, l=10, r=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        font=dict(color="#E2E8F0")),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # ── ROI Calculator ────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">💰 Business ROI Calculator</p>',
                unsafe_allow_html=True)

    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        portfolio_size = st.slider("Total customers in portfolio",
                                   1_000, 500_000, 10_000, step=1_000)
    with rc2:
        cac = st.slider("Avg customer acquisition cost ($)",
                        50, 1_000, 300, step=50)
    with rc3:
        intervention_rate = st.slider("Intervention success rate (%)",
                                      10, 60, 30, step=5)

    est_churners = int(portfolio_size * churn_rate)
    xgb_recall   = float(xgb_metrics["recall"])
    caught       = int(est_churners * xgb_recall)
    saved        = int(caught * (intervention_rate / 100))
    roi          = saved * cac

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Estimated churners",    f"{est_churners:,}")
    r2.metric("Flagged by XGBoost",    f"{caught:,}", f"{xgb_recall*100:.1f}% recall")
    r3.metric("Interventions succeed", f"{saved:,}",  f"{intervention_rate}% success")
    r4.metric("💰 Retention value",    f"${roi:,}")

    st.markdown(
        f'<div class="insight-box">💡 With XGBoost\'s {xgb_recall*100:.1f}% recall, '
        f'Card-Retention-Intelligence identifies <b>{caught:,}</b> at-risk customers from a portfolio of '
        f'<b>{portfolio_size:,}</b>. At {intervention_rate}% intervention success, '
        f'that saves <b>${roi:,}</b> per retention cycle.</div>',
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — CUSTOMER RISK SCORER
# ══════════════════════════════════════════════════════════════════════════════

def page_customer_risk_scorer():
    st.title("🎯 Customer Risk Scorer")
    st.markdown("*Enter customer details to get a live churn probability and SHAP explanation.*")

    xgb, _, scaler   = load_models()
    explainer        = load_shap_explainer()
    _, _, feat_names = load_shap_artifacts()

    st.markdown('<p class="section-header">Customer Profile</p>',
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Demographics**")
        age          = st.slider("Customer Age", 18, 75, 45)
        gender       = st.selectbox("Gender", ["Female", "Male"])
        dependent_ct = st.slider("Number of Dependents", 0, 5, 2)
        education    = st.selectbox("Education Level",
                                    ["Uneducated", "High School", "College",
                                     "Graduate", "Post-Graduate", "Doctorate"])
        marital      = st.selectbox("Marital Status", ["Married", "Single", "Unknown"])

    with col2:
        st.markdown("**Financial Profile**")
        income_cat   = st.selectbox("Income Category",
                                    ["Less than $40K", "$40K - $60K",
                                     "$60K - $80K", "$80K - $120K", "$120K +"])
        card_cat     = st.selectbox("Card Category", ["Blue", "Silver", "Gold", "Platinum"])
        credit_limit = st.slider("Credit Limit ($)", 1_000, 35_000, 8_000, step=500)
        revolving_bal= st.slider("Total Revolving Balance ($)", 0, 3_000, 800, step=100)
        utilisation  = st.slider("Avg Utilisation Ratio", 0.0, 1.0, 0.25, step=0.01)

    with col3:
        st.markdown("**Behavioural Signals**")
        months_book  = st.slider("Months on Book", 12, 56, 36)
        rel_count    = st.slider("Total Relationship Count", 1, 6, 4)
        months_inact = st.slider("Months Inactive (12mo)", 0, 6, 2)
        contacts_ct  = st.slider("Contacts Count (12mo)", 0, 6, 2)
        trans_amt    = st.slider("Total Transaction Amount ($)", 500, 20_000, 4_500, step=100)
        trans_ct     = st.slider("Total Transaction Count", 10, 140, 55)
        amt_chng     = st.slider("Amt Change Q4/Q1", 0.0, 3.5, 0.8, step=0.05)
        ct_chng      = st.slider("Count Change Q4/Q1", 0.0, 3.5, 0.7, step=0.05)

    # ── Feature engineering (mirrors features.py exactly) ─────────────────────
    education_map = {"Uneducated": 0, "High School": 1, "College": 2,
                     "Graduate": 3, "Post-Graduate": 4, "Doctorate": 5, "Unknown": -1}
    income_map    = {"Less than $40K": 0, "$40K - $60K": 1, "$60K - $80K": 2,
                     "$80K - $120K": 3, "$120K +": 4, "Unknown": -1}

    credit_log     = np.log1p(credit_limit)
    open_buy_log   = np.log1p(max(0, credit_limit - revolving_bal))
    trans_amt_log  = np.log1p(trans_amt)
    trans_velocity = trans_amt_log / (trans_ct + 1)
    inact_risk     = months_inact * contacts_ct
    credit_gap     = credit_log - open_buy_log

    raw_features = {
        "Customer_Age"             : age,
        "Gender"                   : 1 if gender == "Male" else 0,
        "Dependent_count"          : dependent_ct,
        "Education_Level"          : education_map[education],
        "Income_Category"          : income_map[income_cat],
        "Months_on_book"           : months_book,
        "Total_Relationship_Count" : rel_count,
        "Months_Inactive_12_mon"   : months_inact,
        "Contacts_Count_12_mon"    : contacts_ct,
        "Total_Revolving_Bal"      : revolving_bal,
        "Total_Amt_Chng_Q4_Q1"     : amt_chng,
        "Total_Trans_Ct"           : trans_ct,
        "Total_Ct_Chng_Q4_Q1"      : ct_chng,
        "Avg_Utilization_Ratio"    : utilisation,
        "Credit_Limit_log"         : credit_log,
        "Avg_Open_To_Buy_log"      : open_buy_log,
        "Total_Trans_Amt_log"      : trans_amt_log,
        "transaction_velocity"     : trans_velocity,
        "inactivity_risk"          : inact_risk,
        "credit_usage_gap"         : credit_gap,
        "Marital_Status_Married"   : 1 if marital == "Married"  else 0,
        "Marital_Status_Single"    : 1 if marital == "Single"   else 0,
        "Marital_Status_Unknown"   : 1 if marital == "Unknown"  else 0,
        "Card_Category_Gold"       : 1 if card_cat == "Gold"    else 0,
        "Card_Category_Platinum"   : 1 if card_cat == "Platinum" else 0,
        "Card_Category_Silver"     : 1 if card_cat == "Silver"  else 0,
    }

    input_df        = pd.DataFrame([raw_features])[feat_names]
    input_scaled    = scaler.transform(input_df)
    input_scaled_df = pd.DataFrame(input_scaled, columns=feat_names)

    churn_prob = float(xgb.predict_proba(input_scaled_df)[0][1])
    churn_pred = int(xgb.predict(input_scaled_df)[0])

    st.markdown("---")
    st.markdown('<p class="section-header">Churn Risk Assessment</p>',
                unsafe_allow_html=True)

    score_col, gauge_col = st.columns([1, 1.5])

    with score_col:
        if churn_prob >= 0.60:
            risk_label, risk_class = "🔴 HIGH RISK",   "risk-high"
            risk_msg = "Immediate intervention recommended."
        elif churn_prob >= 0.30:
            risk_label, risk_class = "🟡 MEDIUM RISK", "risk-medium"
            risk_msg = "Monitor closely and consider proactive outreach."
        else:
            risk_label, risk_class = "🟢 LOW RISK",    "risk-low"
            risk_msg = "Customer appears engaged. Continue standard service."

        st.markdown(
            f'<div class="{risk_class}">{risk_label}<br>'
            f'<span style="font-size:1.1rem">{churn_prob*100:.1f}% churn probability</span>'
            f'</div>', unsafe_allow_html=True)
        st.markdown(f"<br><span style='color:#A0AEC0'>{risk_msg}</span>",
                    unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        m1.metric("Churn Probability", f"{churn_prob*100:.1f}%")
        m2.metric("Prediction", "Will Churn" if churn_pred else "Will Stay")

    with gauge_col:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=churn_prob * 100,
            delta={"reference": 16.07, "suffix": "%", "valueformat": ".1f",
                   "increasing": {"color": CLR_CHURN},
                   "decreasing": {"color": CLR_RETAIN}},
            number={"suffix": "%", "valueformat": ".1f",
                    "font": {"size": 40, "color": "#E2E8F0"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1,
                         "tickcolor": "#718096", "tickfont": {"color": "#A0AEC0"}},
                "bar" : {"color": CLR_CHURN if churn_prob > 0.5 else CLR_RETAIN,
                         "thickness": 0.25},
                "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
                "steps": [
                    {"range": [0,  30],  "color": "rgba(46,204,113,0.12)"},
                    {"range": [30, 60],  "color": "rgba(243,156,18,0.12)"},
                    {"range": [60, 100], "color": "rgba(231,76,60,0.12)"},
                ],
                "threshold": {"line": {"color": "#90CDF4", "width": 3},
                              "thickness": 0.8, "value": 16.07},
            },
            title={"text": "Churn Probability<br>"
                           "<span style='font-size:0.75em;color:#718096'>"
                           "Blue line = dataset avg (16.07%)</span>",
                   "font": {"size": 13, "color": "#A0AEC0"}},
        ))
        fig_gauge.update_layout(
            **PLOTLY_BASE,
            height=270, margin=dict(t=50, b=10, l=30, r=30),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    # ── Live SHAP Waterfall ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-header">🔍 Why This Prediction? (SHAP Explanation)</p>',
                unsafe_allow_html=True)

    with st.spinner("Computing SHAP explanation..."):
        shap_exp = explainer(input_scaled_df)

    with plt.style.context("dark_background"):
        fig_wf, _ = plt.subplots(figsize=(11, 6))
        shap.plots.waterfall(shap_exp[0], max_display=12, show=False)
        plt.title(f"SHAP Waterfall — Predicted: {churn_prob*100:.1f}% churn probability",
                  fontsize=11, pad=10, color="white")
        plt.tight_layout()
        st.pyplot(fig_wf, use_container_width=True)
        plt.close("all")

    st.markdown(
        '<div class="insight-box">📖 <b>How to read this:</b> '
        'Each bar shows how much a feature pushes the prediction '
        '<b style="color:#FC8181">toward churn (right)</b> or '
        '<b style="color:#68D391">away from churn (left)</b>. '
        'Bottom = model baseline. Top = final predicted probability.</div>',
        unsafe_allow_html=True
    )

    # ── Retention Recommendations ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-header">💡 Retention Recommendations</p>',
                unsafe_allow_html=True)

    recs = []
    if trans_ct < 40:
        recs.append(("🛒 Transaction Engagement",
                     "Customer transacts fewer than 40 times per year. "
                     "Offer bonus rewards or cashback on next 10 transactions."))
    if months_inact >= 3:
        recs.append(("😴 Re-engagement Campaign",
                     f"Inactive for {months_inact} months. "
                     "Send a personalised re-engagement offer."))
    if contacts_ct >= 4:
        recs.append(("📞 Service Quality Review",
                     f"Contacted support {contacts_ct} times — dissatisfaction signal. "
                     "Escalate to relationship manager."))
    if utilisation < 0.10:
        recs.append(("💳 Card Usage Incentive",
                     f"Utilisation is {utilisation:.2f} — near zero. "
                     "Offer a targeted spend challenge with rewards."))
    if churn_prob < 0.30:
        recs.append(("✅ Maintain Relationship",
                     "Low risk. Consider loyalty tier upgrade to deepen engagement."))
    if not recs:
        recs.append(("📋 Standard Monitoring",
                     "Schedule a quarterly relationship review."))

    for title, body in recs:
        st.markdown(
            f'<div class="rec-card"><b>{title}</b><br><small>{body}</small></div>',
            unsafe_allow_html=True
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════

def page_model_performance():
    st.title("📈 Model Performance")
    st.markdown("*Test-set evaluation on 2,026 held-out customers never seen during training.*")

    metrics_df = load_metrics()
    cv_df      = load_cv_comparison()

    st.markdown('<p class="section-header">Test-Set Metrics</p>',
                unsafe_allow_html=True)
    display_df = metrics_df.copy()
    display_df.columns = [c.replace("_", " ").title() for c in display_df.columns]
    st.dataframe(
        display_df.style.format(
            {c: "{:.4f}" for c in display_df.columns if c != "Model"}
        ).highlight_max(
            subset=[c for c in display_df.columns if c != "Model"],
            color="#1A3A1A"
        ),
        use_container_width=True, hide_index=True
    )

    st.markdown('<p class="section-header">5-Fold Cross-Validation Results</p>',
                unsafe_allow_html=True)
    cv_display = cv_df.copy()
    cv_display.columns = ["Metric", "LogReg Mean", "LogReg Std",
                           "XGBoost Mean", "XGBoost Std"]
    st.dataframe(
        cv_display.style.format(
            {c: "{:.4f}" for c in cv_display.columns if c != "Metric"}
        ),
        use_container_width=True, hide_index=True
    )

    st.markdown(
        '<div class="insight-box">📖 Cross-validation runs on the <b>training set only</b>. '
        'The test set is held out completely — metrics are unbiased. '
        'XGBoost achieves ROC-AUC <b>0.9930 ± 0.0016</b> across all 5 folds.</div>',
        unsafe_allow_html=True
    )
    st.markdown("---")

    plot_groups = [
        ("ROC Curves",                  "roc_curves.png",
         "AUC comparison. XGBoost (0.9930) dominates the random baseline."),
        ("Precision-Recall Curves",     "pr_curves.png",
         "More informative than ROC for imbalanced classes. XGBoost PR-AUC 0.9680."),
        ("Confusion Matrices",          "confusion_matrices.png",
         "XGBoost identifies 90.5% of actual churners at 92.7% precision."),
        ("Feature Importance (XGBoost)","feature_importance.png",
         "Gain-based importance. Compare with SHAP for direction-aware version."),
        ("Model Comparison Scorecard",  "model_comparison.png",
         "XGBoost outperforms on every metric. Largest gap on F1 (+0.26)."),
    ]

    for i in range(0, len(plot_groups), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(plot_groups):
                title, fname, caption = plot_groups[i + j]
                with col:
                    st.markdown(f'<p class="section-header">{title}</p>',
                                unsafe_allow_html=True)
                    show_image(load_figure(fname))
                    st.caption(caption)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — SHAP EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════

def page_shap_explainability():
    st.title("🔍 SHAP Explainability")
    st.markdown("*Why the model makes each prediction — direction-aware feature contributions.*")

    shap_values, _, feature_names = load_shap_artifacts()

    st.markdown(
        '<div class="insight-box">🧠 <b>What is SHAP?</b> '
        'Every prediction = baseline rate + sum of feature contributions. '
        'SHAP values are <b>direction-aware</b> (positive → churn, negative → stay) '
        'and <b>locally faithful</b> — each individual prediction is explained exactly.</div>',
        unsafe_allow_html=True
    )

    st.markdown('<p class="section-header">Global Feature Importance (SHAP)</p>',
                unsafe_allow_html=True)

    mean_abs = np.abs(shap_values).mean(axis=0)
    mean_dir = shap_values.mean(axis=0)
    imp_df   = pd.DataFrame({
        "Feature"     : feature_names,
        "Mean |SHAP|" : mean_abs.round(4),
        "Direction"   : ["↑ Toward Churn" if v > 0 else "↓ Away from Churn"
                         for v in mean_dir],
    }).sort_values("Mean |SHAP|", ascending=False).reset_index(drop=True)
    imp_df.index += 1

    col_t, col_b = st.columns([1.2, 1])
    with col_t:
        st.dataframe(imp_df.head(15), use_container_width=True, height=420)
    with col_b:
        fig_imp = go.Figure(go.Bar(
            x=imp_df["Mean |SHAP|"].head(12)[::-1],
            y=imp_df["Feature"].head(12)[::-1],
            orientation="h",
            marker_color=CLR_ACCENT, opacity=0.85,
            text=imp_df["Mean |SHAP|"].head(12)[::-1].round(3),
            textposition="outside",
            textfont=dict(color="#E2E8F0"),
        ))
        fig_imp.update_layout(
            **PLOTLY_BASE,
            title=dict(text="Top 12 by Mean |SHAP|",
                       font=dict(color="#E2E8F0")),
            xaxis=dict(title="Mean |SHAP Value|", color="#A0AEC0",
                       gridcolor="#2D3748"),
            yaxis=dict(color="#E2E8F0"),
            margin=dict(t=40, b=10, l=10, r=70),
            height=400,
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("---")

    shap_plots = [
        ("SHAP Summary — Beeswarm",            "shap_summary.png",
         "Y = importance. X = SHAP direction. Colour = feature value. Each dot = one customer."),
        ("SHAP Global Importance Bar",         "shap_importance_bar.png",
         "Mean |SHAP| per feature — honest ranking based on actual prediction impact."),
        ("SHAP Waterfall — Highest Risk",      "shap_waterfall_high_risk.png",
         "Local explanation for the highest-risk churner. Predicted probability: 100%."),
        ("SHAP Dependence — Transaction Count","shap_dependence_trans_ct.png",
         "Threshold effect around 40-50 transactions — the business intervention trigger."),
    ]

    for i in range(0, len(shap_plots), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(shap_plots):
                title, fname, caption = shap_plots[i + j]
                with col:
                    st.markdown(f'<p class="section-header">{title}</p>',
                                unsafe_allow_html=True)
                    show_image(load_figure(fname))
                    st.caption(caption)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — EDA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════

def page_eda_explorer():
    st.title("🔬 EDA Explorer")
    st.markdown("*Exploratory Data Analysis — understanding the dataset before modelling.*")

    df = load_ingested_data()

    st.markdown('<p class="section-header">Dataset Overview</p>',
                unsafe_allow_html=True)
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Total Customers", f"{len(df):,}")
    s2.metric("Features",        f"{df.shape[1] - 1}")
    s3.metric("Churned",         f"{df['churn'].sum():,}")
    s4.metric("Churn Rate",      f"{df['churn'].mean()*100:.1f}%")
    s5.metric("Avg Age",         f"{df['Customer_Age'].mean():.0f} yrs")

    st.markdown('<p class="section-header">Interactive Feature Analysis</p>',
                unsafe_allow_html=True)

    numeric_cols = [c for c in df.select_dtypes(include="number").columns
                    if c not in ["churn", "CLIENTNUM"]]

    sel_col, dist_col = st.columns([1, 3])
    with sel_col:
        chosen_feat   = st.selectbox("Select feature", numeric_cols, index=4)
        show_by_churn = st.checkbox("Split by churn status", value=True)

    with dist_col:
        fig_dist = go.Figure()
        if show_by_churn:
            for churn_val, name, colour in [
                (0, "Retained", CLR_RETAIN), (1, "Churned", CLR_CHURN)
            ]:
                fig_dist.add_trace(go.Histogram(
                    x=df[df["churn"] == churn_val][chosen_feat],
                    name=name, marker_color=colour, opacity=0.65, nbinsx=30,
                ))
            fig_dist.update_layout(barmode="overlay")
        else:
            fig_dist.add_trace(go.Histogram(
                x=df[chosen_feat], nbinsx=30,
                marker_color=CLR_ACCENT, opacity=0.8,
            ))

        fig_dist.update_layout(
            **PLOTLY_BASE,
            height=240,
            title=dict(text=f"{chosen_feat} Distribution",
                       font=dict(color="#E2E8F0")),
            xaxis=dict(color="#A0AEC0", gridcolor="#2D3748"),
            yaxis=dict(color="#A0AEC0", gridcolor="#2D3748"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        font=dict(color="#E2E8F0")),
            margin=dict(t=40, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown("---")
    st.markdown('<p class="section-header">EDA Plots</p>', unsafe_allow_html=True)

    eda_plots = [
        ("Churn Distribution",        "churn_distribution.png",
         "16.07% churn rate — class imbalance handled with scale_pos_weight."),
        ("Age Distribution",          "age_distribution.png",
         "Churned customers are slightly younger on average."),
        ("Churn by Age Group",        "churn_by_age.png",
         "Churn rate is consistent across age groups — age is not the primary driver."),
        ("Credit Limit Distribution", "credit_limit_dist.png",
         "Right-skewed — motivates log transform in feature engineering."),
        ("Transaction Count vs Churn","trans_count_vs_churn.png",
         "Strongest visual separator — churned customers transact far less."),
        ("Correlation Heatmap",       "correlation_heatmap.png",
         "Credit_Limit and Avg_Open_To_Buy are highly correlated."),
        ("Churn by Segment",          "churn_by_category.png",
         "Platinum card holders: 25% churn rate — highest of any segment."),
        ("Numeric Distributions",     "numeric_distributions.png",
         "Full overview of all features split by churn status."),
        ("Utilisation vs Churn",      "utilisation_vs_churn.png",
         "Churned customers cluster near zero utilisation."),
        ("Contact Frequency vs Churn","contacts_vs_churn.png",
         "Higher contact frequency = higher churn — dissatisfaction signal."),
    ]

    for i in range(0, len(eda_plots), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(eda_plots):
                title, fname, caption = eda_plots[i + j]
                with col:
                    st.markdown(f'<p class="section-header">{title}</p>',
                                unsafe_allow_html=True)
                    show_image(load_figure(fname))
                    st.caption(caption)

    st.markdown("---")
    st.markdown('<p class="section-header">Raw Data Preview</p>',
                unsafe_allow_html=True)

    col_toggle, _ = st.columns([1, 3])
    with col_toggle:
        show_churned_only = st.checkbox("Show churned customers only", value=False)

    preview_df = df[df["churn"] == 1] if show_churned_only else df
    st.dataframe(preview_df.head(100), use_container_width=True,
                 height=300, hide_index=True)
    st.caption(f"Showing first 100 of {len(preview_df):,} rows.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ROUTER
# ══════════════════════════════════════════════════════════════════════════════

def main():
    inject_css()
    page = render_sidebar()

    if   page == "📊 Executive Summary":   page_executive_summary()
    elif page == "🎯 Customer Risk Scorer": page_customer_risk_scorer()
    elif page == "📈 Model Performance":    page_model_performance()
    elif page == "🔍 SHAP Explainability":  page_shap_explainability()
    elif page == "🔬 EDA Explorer":         page_eda_explorer()


if __name__ == "__main__":
    main()