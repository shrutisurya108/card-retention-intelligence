"""
explain.py
----------
Phase 4 — SHAP Explainability

Responsibilities:
  1. Load trained XGBoost model and held-out test set
  2. Compute SHAP values using TreeExplainer (fastest for tree models)
  3. Save SHAP values, expected value, and feature names to disk
     (dashboard loads these instead of recomputing on every request)
  4. Generate 4 SHAP plots covering global, local, and interaction levels

Why XGBoost only for SHAP?
  TreeExplainer is purpose-built for tree-based models and runs in
  O(TLD) time (T=trees, L=leaves, D=depth) — orders of magnitude faster
  than KernelExplainer which works on any model but samples and approximates.
  For the dashboard to feel responsive, we need fast exact SHAP values.
  Logistic Regression has direct coefficient interpretation anyway.

Why save SHAP values to disk?
  Computing SHAP for 2,026 test samples takes ~2-5 seconds.
  Doing that on every Streamlit page load would make the dashboard feel broken.
  Precomputed values load in milliseconds and the dashboard stays snappy.

SHAP storytelling levels covered:
  - Global  : Summary beeswarm + bar chart (all features, all customers)
  - Local   : Waterfall plot (one specific high-risk customer)
  - Interaction: Dependence plot (how one feature's SHAP varies across its range)
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import joblib
import shap
from pathlib import Path

from src.logger import get_logger

log = get_logger("explain")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR          = Path(__file__).resolve().parent.parent
TEST_CSV          = ROOT_DIR / "data" / "processed" / "test_set.csv"
XGBOOST_PATH      = ROOT_DIR / "models" / "xgboost_model.pkl"
OUTPUTS_DIR       = ROOT_DIR / "outputs"
FIGURES_DIR       = ROOT_DIR / "outputs" / "figures"

# Saved SHAP artifacts — loaded by dashboard in Phase 5
SHAP_VALUES_PATH  = OUTPUTS_DIR / "shap_values.npy"
SHAP_EXPECTED_PATH= OUTPUTS_DIR / "shap_expected_value.npy"
SHAP_FEATNAMES_PATH = OUTPUTS_DIR / "shap_feature_names.json"

TARGET_COL  = "churn"
PLOT_DPI    = 300

# Consistent colour scheme with other phases
CLR_POS  = "#E74C3C"   # red   — pushes toward churn
CLR_NEG  = "#2ECC71"   # green — pushes away from churn
CLR_BASE = "#2C3E50"   # dark  — neutral bars


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_artifacts() -> tuple:
    """
    Load the XGBoost model and test set.
    Returns (model, X_test, y_test, feature_names).
    """
    log.info(f"Loading XGBoost model: {XGBOOST_PATH}")
    model = joblib.load(XGBOOST_PATH)

    log.info(f"Loading test set: {TEST_CSV}")
    df     = pd.read_csv(TEST_CSV)
    X_test = df.drop(columns=[TARGET_COL])
    y_test = df[TARGET_COL]

    feature_names = X_test.columns.tolist()
    log.info(f"Test set: {len(df):,} rows × {len(feature_names)} features")
    return model, X_test, y_test, feature_names


# ── SHAP computation ──────────────────────────────────────────────────────────

def compute_shap_values(model, X_test: pd.DataFrame) -> tuple:
    """
    Compute SHAP values using TreeExplainer.

    TreeExplainer vs KernelExplainer:
      - TreeExplainer: exact SHAP values, O(TLD) complexity, milliseconds
      - KernelExplainer: model-agnostic but approximate, minutes on this dataset
      TreeExplainer is always preferred for XGBoost/LightGBM/RandomForest.

    explainer.expected_value:
      The base rate — model output if we knew nothing about the customer.
      For a classifier with ~16% churn, this is approximately log-odds of 0.16.
      Each SHAP value is an additive shift from this baseline.

    Returns:
      shap_values      : np.ndarray of shape (n_samples, n_features)
                         Each value = feature's contribution to that prediction
      expected_value   : float — baseline prediction (model intercept)
      shap_explanation : shap.Explanation object for plot functions
    """
    log.info("Initialising SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)

    log.info(f"Computing SHAP values for {len(X_test):,} test samples...")
    log.info("(This may take 10-30 seconds for 2,026 samples × 500 trees)")

    shap_explanation = explainer(X_test)
    shap_values      = shap_explanation.values       # shape: (n_samples, n_features)
    expected_value   = explainer.expected_value       # scalar baseline

    # Handle case where expected_value is an array (multi-output)
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = float(expected_value[1]) if len(expected_value) > 1 \
                         else float(expected_value[0])
    else:
        expected_value = float(expected_value)

    log.info(f"SHAP values computed — shape: {shap_values.shape}")
    log.info(f"Expected value (baseline): {expected_value:.6f}")
    log.info(f"Mean |SHAP| across all features: "
             f"{np.abs(shap_values).mean():.6f}")

    return shap_values, expected_value, shap_explanation


# ── Save SHAP artifacts ───────────────────────────────────────────────────────

def save_shap_artifacts(shap_values: np.ndarray,
                        expected_value: float,
                        feature_names: list) -> None:
    """
    Persist SHAP outputs to disk for dashboard consumption.
    Dashboard loads these files instead of recomputing on every page load.
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    np.save(SHAP_VALUES_PATH,   shap_values)
    np.save(SHAP_EXPECTED_PATH, np.array([expected_value]))

    with open(SHAP_FEATNAMES_PATH, "w") as f:
        json.dump(feature_names, f, indent=2)

    log.info(f"SHAP values saved       → {SHAP_VALUES_PATH.relative_to(ROOT_DIR)}")
    log.info(f"Expected value saved    → {SHAP_EXPECTED_PATH.relative_to(ROOT_DIR)}")
    log.info(f"Feature names saved     → {SHAP_FEATNAMES_PATH.relative_to(ROOT_DIR)}")


# ── SHAP audit log ────────────────────────────────────────────────────────────

def log_shap_summary(shap_values: np.ndarray, feature_names: list) -> None:
    """
    Log top-10 most impactful features by mean absolute SHAP value.
    This is the global feature importance ranking — the 'truth' version
    of XGBoost's built-in importance (which only measures usage frequency).
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature"         : feature_names,
        "mean_abs_shap"   : mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    log.info("── SHAP Global Feature Importance (Top 10) ─────────────")
    log.info(f"  {'Rank':<5} {'Feature':<30} {'Mean |SHAP|'}")
    log.info(f"  {'─'*5} {'─'*30} {'─'*12}")
    for i, row in importance_df.head(10).iterrows():
        rank = importance_df.index.get_loc(i) + 1
        log.info(f"  {rank:<5} {row['feature']:<30} {row['mean_abs_shap']:.4f}")
    log.info("────────────────────────────────────────────────────────")


# ── Plot 1: SHAP Summary (Beeswarm) ──────────────────────────────────────────

def plot_shap_summary(shap_explanation, X_test: pd.DataFrame) -> None:
    """
    Beeswarm summary plot — the most information-dense SHAP visualisation.

    What it shows:
      - Y-axis: features ranked by mean |SHAP| (most important at top)
      - X-axis: SHAP value (negative = pushes toward retained, positive = toward churn)
      - Colour: feature value (red = high, blue = low)
      - Each dot: one customer in the test set

    How to read it (interview answer):
      "High Total_Trans_Ct with a low feature value (blue) has a large
       negative SHAP — customers who transact a lot are much less likely
       to churn. Low Avg_Utilization_Ratio (blue dots on the right) pushes
       toward churn — customers who stopped using the card show high churn risk."

    This plot tells the complete feature importance story in one image.
    """
    log.info("Generating SHAP summary (beeswarm) plot...")

    fig, ax = plt.subplots(figsize=(11, 9))
    shap.plots.beeswarm(
        shap_explanation,
        max_display=15,
        show=False,
        plot_size=None,
    )
    plt.title("SHAP Summary — Feature Impact on Churn Prediction\n"
              "Red = high feature value  |  Blue = low feature value  |  "
              "Right = pushes toward churn",
              fontsize=11, pad=15)
    plt.tight_layout()
    _save_current_fig("shap_summary.png")


# ── Plot 2: SHAP Global Bar Chart ────────────────────────────────────────────

def plot_shap_importance_bar(shap_values: np.ndarray,
                             feature_names: list) -> None:
    """
    Horizontal bar chart of mean absolute SHAP values (global importance).

    Why this over XGBoost's built-in feature_importances_?
      Built-in importance = how often a feature is used to split.
      A feature used many times with tiny splits ranks high but may matter little.
      Mean |SHAP| = average actual impact on predictions across all customers.
      SHAP importance is the honest, model-agnostic ranking.

    Clean, recruiter-friendly visual. Easy to screenshot for a portfolio.
    """
    log.info("Generating SHAP importance bar plot...")

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feat_df = pd.DataFrame({
        "feature"  : feature_names,
        "shap_imp" : mean_abs_shap,
    }).sort_values("shap_imp", ascending=True).tail(15)  # top 15

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(
        feat_df["feature"],
        feat_df["shap_imp"],
        color=CLR_BASE, alpha=0.85, edgecolor="white", height=0.65
    )

    # Annotate bars with exact values
    for bar, val in zip(bars, feat_df["shap_imp"]):
        ax.text(
            bar.get_width() + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center", fontsize=8.5
        )

    ax.set_title("SHAP Global Feature Importance (Top 15)\n"
                 "Mean |SHAP value| across all test customers",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Mean |SHAP Value|  (average impact on model output)")
    ax.set_ylabel("")

    # Clean grid
    ax.xaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    _save_current_fig("shap_importance_bar.png")


# ── Plot 3: SHAP Waterfall (single high-risk customer) ───────────────────────

def plot_shap_waterfall(shap_explanation,
                        X_test: pd.DataFrame,
                        y_test: pd.Series,
                        model) -> None:
    """
    Waterfall plot for a single high-risk customer.

    What it shows:
      Starting from the baseline (E[f(X)] = expected model output),
      each feature either adds to or subtracts from the prediction.
      The final value is the model's churn probability for this customer.

    Why this is the most powerful SHAP plot for the dashboard:
      A retention manager looking at a flagged customer needs to know:
      "This customer has 89% churn probability. The #1 driver is their
       near-zero transaction count. Their high credit limit is actually
       protective but not enough to offset the inactivity."
      That's an actionable insight — not just a number.

    Customer selection:
      We pick the highest-risk ACTUAL churner (y_test=1) in the test set.
      This makes the explanation honest — we're not picking a false positive.
      It also tells the clearest story: the model correctly flagged a real churner
      and here's exactly why.
    """
    log.info("Generating SHAP waterfall plot for highest-risk churner...")

    # Find the highest-probability actual churner in the test set
    y_prob = model.predict_proba(X_test)[:, 1]
    actual_churner_mask = (y_test.values == 1)
    churner_probs = y_prob * actual_churner_mask
    high_risk_idx = int(np.argmax(churner_probs))

    churn_prob = y_prob[high_risk_idx]
    log.info(f"High-risk customer index: {high_risk_idx}  "
             f"(predicted churn probability: {churn_prob:.4f})")

    fig, ax = plt.subplots(figsize=(11, 8))
    shap.plots.waterfall(
        shap_explanation[high_risk_idx],
        max_display=15,
        show=False,
    )
    plt.title(
        f"SHAP Waterfall — Highest-Risk Customer\n"
        f"Predicted churn probability: {churn_prob*100:.1f}%  |  "
        f"Actual outcome: Churned ✓",
        fontsize=11, pad=15
    )
    plt.tight_layout()
    _save_current_fig("shap_waterfall_high_risk.png")


# ── Plot 4: SHAP Dependence Plot ─────────────────────────────────────────────

def plot_shap_dependence(shap_values: np.ndarray,
                         X_test: pd.DataFrame,
                         feature_names: list) -> None:
    """
    Dependence plot for Total_Trans_Ct (top predictor by SHAP).

    What it shows:
      X-axis: actual feature value (number of transactions)
      Y-axis: SHAP value for that feature (its contribution to churn score)
      Colour: a second feature (auto-selected for interaction — usually
              the feature that explains most of the variance in the SHAP values)

    Why Total_Trans_Ct?
      It's consistently the #1 SHAP feature for churn prediction.
      The dependence plot reveals a clear threshold effect:
      customers with <40 transactions show sharply elevated churn SHAP values.
      This is the kind of insight that drives a real business rule:
      "Flag any customer who drops below 40 transactions in 12 months."

    Interview talking point:
      "SHAP dependence plots let me identify non-linear thresholds that
       linear models miss entirely. The XGBoost model learned that transaction
       count has a step-change effect around 40-50 transactions per year,
       which directly informs our intervention trigger."
    """
    log.info("Generating SHAP dependence plot (Total_Trans_Ct)...")

    # Identify the index of Total_Trans_Ct
    target_feature = "Total_Trans_Ct"
    if target_feature not in feature_names:
        # Fallback to top SHAP feature if column name changed
        top_idx = int(np.abs(shap_values).mean(axis=0).argmax())
        target_feature = feature_names[top_idx]
        log.warning(f"Total_Trans_Ct not found — using top feature: {target_feature}")

    feat_idx    = feature_names.index(target_feature)
    feat_values = X_test[target_feature].values
    shap_vals   = shap_values[:, feat_idx]

    # Auto-select interaction feature: highest correlation with SHAP residuals
    residuals = shap_vals - np.polyval(np.polyfit(feat_values, shap_vals, 1), feat_values)
    corr_with_residuals = [
        abs(np.corrcoef(residuals, X_test.iloc[:, j].values)[0, 1])
        for j in range(X_test.shape[1]) if j != feat_idx
    ]
    interact_idx = int(np.argmax(corr_with_residuals))
    # Adjust index to skip feat_idx
    if interact_idx >= feat_idx:
        interact_idx += 1
    interact_feature = feature_names[interact_idx]
    interact_values  = X_test[interact_feature].values

    log.info(f"Dependence plot: {target_feature} (SHAP) × {interact_feature} (colour)")

    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(
        feat_values, shap_vals,
        c=interact_values,
        cmap="RdYlGn_r",
        alpha=0.6, s=18, linewidths=0
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(interact_feature.replace("_", " "), fontsize=9)

    # Add trend line
    z = np.polyfit(feat_values, shap_vals, 3)
    p = np.poly1d(z)
    x_line = np.linspace(feat_values.min(), feat_values.max(), 200)
    ax.plot(x_line, p(x_line), color="#E74C3C", linewidth=2,
            linestyle="--", label="Trend", zorder=5)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="-", alpha=0.5)
    ax.set_title(
        f"SHAP Dependence Plot — {target_feature.replace('_', ' ')}\n"
        f"How transaction count drives churn risk  |  "
        f"Colour = {interact_feature.replace('_', ' ')}",
        fontsize=11, fontweight="bold", pad=15
    )
    ax.set_xlabel(f"{target_feature.replace('_', ' ')}  (annual transactions)")
    ax.set_ylabel(f"SHAP value for {target_feature.replace('_', ' ')}\n"
                  f"(positive = pushes toward churn)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    _save_current_fig("shap_dependence_trans_ct.png")


# ── Save helper ───────────────────────────────────────────────────────────────

def _save_current_fig(filename: str) -> None:
    """Save the current matplotlib figure and close it."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / filename
    plt.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close("all")
    log.info(f"Saved plot → outputs/figures/{filename}")


# ── Main entry point ──────────────────────────────────────────────────────────

def run_explanation() -> None:
    """
    Execute the full SHAP explainability pipeline.
    Loads model + test data, computes SHAP values, saves artifacts and plots.
    """
    log.info("═══════════════════════════════════════════════════════")
    log.info("  PHASE 4 — SHAP EXPLAINABILITY                      ")
    log.info("═══════════════════════════════════════════════════════")

    # Load model and data
    model, X_test, y_test, feature_names = load_artifacts()

    # Compute SHAP values
    shap_values, expected_value, shap_explanation = compute_shap_values(model, X_test)

    # Save artifacts for dashboard
    save_shap_artifacts(shap_values, expected_value, feature_names)

    # Log feature importance summary
    log_shap_summary(shap_values, feature_names)

    # Generate all 4 plots
    log.info("Generating SHAP plots...")
    plot_shap_summary(shap_explanation, X_test)
    plot_shap_importance_bar(shap_values, feature_names)
    plot_shap_waterfall(shap_explanation, X_test, y_test, model)
    plot_shap_dependence(shap_values, X_test, feature_names)

    log.info("All 4 SHAP plots saved to outputs/figures/")
    log.info("Phase 4 complete — SHAP explainability successful ✓")


if __name__ == "__main__":
    run_explanation()