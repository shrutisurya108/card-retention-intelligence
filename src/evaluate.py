"""
evaluate.py
-----------
Phase 3 — Model Evaluation

Responsibilities:
  1. Load saved models and held-out test set
  2. Generate predictions and probabilities for both models
  3. Compute full metrics suite: AUC, F1, Precision, Recall, Accuracy
  4. Save 5 evaluation plots to outputs/figures/
  5. Save final test-set metrics to outputs/test_metrics.csv
  6. Log business-framed interpretation of results

Plots produced:
  - roc_curves.png           : ROC curves for both models
  - pr_curves.png            : Precision-Recall curves (better for imbalance)
  - confusion_matrices.png   : Side-by-side confusion matrices
  - feature_importance.png   : XGBoost top-20 feature importances
  - model_comparison.png     : Metric scorecard table

Why test metrics are separate from CV metrics:
  Cross-validation (train.py) estimates generalisation.
  Test-set evaluation (here) is the FINAL, unbiased performance report.
  These numbers are what go on the resume and in the dashboard.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score, accuracy_score,
    confusion_matrix,
)

from src.logger import get_logger

log = get_logger("evaluate")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR      = Path(__file__).resolve().parent.parent
TEST_CSV      = ROOT_DIR / "data" / "processed" / "test_set.csv"
XGBOOST_PATH  = ROOT_DIR / "models" / "xgboost_model.pkl"
LOGREG_PATH   = ROOT_DIR / "models" / "logreg_model.pkl"
FIGURES_DIR   = ROOT_DIR / "outputs" / "figures"
METRICS_CSV   = ROOT_DIR / "outputs" / "test_metrics.csv"

TARGET_COL    = "churn"
CLR_XGB       = "#2C3E50"
CLR_LR        = "#E74C3C"
CLR_BASELINE  = "#95A5A6"
PLOT_DPI      = 300

sns.set_theme(style="whitegrid", font_scale=1.1)


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_test_set() -> tuple:
    """Load the held-out test set. Never used during training."""
    log.info(f"Loading test set: {TEST_CSV}")
    df = pd.read_csv(TEST_CSV)
    X_test = df.drop(columns=[TARGET_COL])
    y_test = df[TARGET_COL]
    log.info(f"Test set: {len(df):,} rows  (churn: {y_test.mean()*100:.2f}%)")
    return X_test, y_test


def load_models() -> tuple:
    """Load both saved models from models/ directory."""
    log.info("Loading saved models...")
    logreg = joblib.load(LOGREG_PATH)
    xgb    = joblib.load(XGBOOST_PATH)
    log.info("Models loaded ✓")
    return logreg, xgb


# ── Metrics computation ───────────────────────────────────────────────────────

def compute_metrics(model, model_name: str,
                    X_test: pd.DataFrame,
                    y_test: pd.Series) -> dict:
    """
    Compute full test-set metrics for one model.
    Uses 0.5 as classification threshold — appropriate for our use case
    since scale_pos_weight / class_weight already calibrate the model.
    """
    y_pred      = model.predict(X_test)
    y_prob      = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model"      : model_name,
        "roc_auc"    : round(roc_auc_score(y_test, y_prob), 4),
        "pr_auc"     : round(average_precision_score(y_test, y_prob), 4),
        "f1"         : round(f1_score(y_test, y_pred), 4),
        "precision"  : round(precision_score(y_test, y_pred), 4),
        "recall"     : round(recall_score(y_test, y_pred), 4),
        "accuracy"   : round(accuracy_score(y_test, y_pred), 4),
    }

    log.info(f"  {model_name} Test-Set Metrics:")
    for k, v in metrics.items():
        if k != "model":
            log.info(f"    {k:<12}: {v:.4f}")

    return metrics


# ── Plot 1: ROC Curves ────────────────────────────────────────────────────────

def plot_roc_curves(logreg, xgb, X_test, y_test) -> None:
    """
    ROC curves for both models on the test set.
    Baseline (random classifier) shown as dashed diagonal.
    AUC annotated directly on each curve.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for model, name, colour in [
        (logreg, "Logistic Regression", CLR_LR),
        (xgb,    "XGBoost",             CLR_XGB),
    ]:
        y_prob  = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc     = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, color=colour, linewidth=2.5,
                label=f"{name}  (AUC = {auc:.4f})")

    ax.plot([0, 1], [0, 1], color=CLR_BASELINE, linestyle="--",
            linewidth=1.5, label="Random Classifier (AUC = 0.50)")

    ax.set_title("ROC Curves — Test Set", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    sns.despine()

    _save(fig, "roc_curves.png")


# ── Plot 2: Precision-Recall Curves ──────────────────────────────────────────

def plot_pr_curves(logreg, xgb, X_test, y_test) -> None:
    """
    Precision-Recall curves — more informative than ROC for imbalanced data.
    Baseline = churn rate (a model predicting all positives).
    Interview point: "With 16% churn, ROC can be misleadingly optimistic.
    PR-AUC gives a truer picture of minority class performance."
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for model, name, colour in [
        (logreg, "Logistic Regression", CLR_LR),
        (xgb,    "XGBoost",             CLR_XGB),
    ]:
        y_prob = model.predict_proba(X_test)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        ax.plot(rec, prec, color=colour, linewidth=2.5,
                label=f"{name}  (PR-AUC = {ap:.4f})")

    baseline = y_test.mean()
    ax.axhline(baseline, color=CLR_BASELINE, linestyle="--",
               linewidth=1.5, label=f"Baseline (churn rate = {baseline:.2f})")

    ax.set_title("Precision-Recall Curves — Test Set",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="upper right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    sns.despine()

    _save(fig, "pr_curves.png")


# ── Plot 3: Confusion Matrices ────────────────────────────────────────────────

def plot_confusion_matrices(logreg, xgb, X_test, y_test) -> None:
    """
    Side-by-side confusion matrices normalised by true label (row).
    Normalisation makes it easy to compare recall/specificity visually
    regardless of class imbalance.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, model, name in [
        (axes[0], logreg, "Logistic Regression"),
        (axes[1], xgb,    "XGBoost"),
    ]:
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, normalize="true")
        cm_raw = confusion_matrix(y_test, y_pred)

        sns.heatmap(
            cm, annot=False, fmt=".2%", cmap="Blues",
            xticklabels=["Predicted\nRetained", "Predicted\nChurned"],
            yticklabels=["Actual\nRetained", "Actual\nChurned"],
            ax=ax, vmin=0, vmax=1, linewidths=0.5,
            cbar_kws={"format": mticker.PercentFormatter(xmax=1)}
        )

        # Annotate cells with both percentage and raw count
        for i in range(2):
            for j in range(2):
                ax.text(
                    j + 0.5, i + 0.5,
                    f"{cm[i,j]*100:.1f}%\n({cm_raw[i,j]:,})",
                    ha="center", va="center",
                    fontsize=12, fontweight="bold",
                    color="white" if cm[i, j] > 0.5 else "black"
                )

        recall_val = recall_score(y_test, y_pred)
        prec_val   = precision_score(y_test, y_pred)
        ax.set_title(
            f"{name}\nRecall: {recall_val:.3f}  |  Precision: {prec_val:.3f}",
            fontsize=11, fontweight="bold", pad=10
        )

    fig.suptitle("Confusion Matrices — Test Set (Row-Normalised)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, "confusion_matrices.png")


# ── Plot 4: XGBoost Feature Importance ───────────────────────────────────────

def plot_feature_importance(xgb, X_test: pd.DataFrame) -> None:
    """
    Top-20 XGBoost feature importances by 'gain'.

    Why 'gain' over 'weight'?
        'weight' = number of times a feature is used to split.
        'gain'   = average improvement in loss when a feature is used.
        Gain is the more meaningful metric — a feature used rarely but
        with large gain is more important than one used often with small gain.

    This plot is a preview of Phase 4 (SHAP). SHAP will give us
    direction-aware importances (which way each feature pushes predictions).
    """
    importances = xgb.feature_importances_
    feat_names  = X_test.columns.tolist()

    feat_df = pd.DataFrame({
        "feature"   : feat_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(
        feat_df["feature"][::-1],
        feat_df["importance"][::-1],
        color=CLR_XGB, alpha=0.85, edgecolor="white"
    )

    ax.set_title("XGBoost Feature Importance (Top 20, by Gain)",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Feature Importance (Gain)")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.3f}"))

    for bar, val in zip(bars, feat_df["importance"][::-1]):
        ax.text(bar.get_width() + 0.0002, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=8)

    sns.despine()
    _save(fig, "feature_importance.png")


# ── Plot 5: Model Comparison Scorecard ───────────────────────────────────────

def plot_model_comparison(logreg_metrics: dict, xgb_metrics: dict) -> None:
    """
    Visual scorecard comparing both models across all metrics.
    Bar chart makes the performance gap immediately obvious.
    """
    metrics_to_plot = ["roc_auc", "pr_auc", "f1", "precision", "recall", "accuracy"]
    x      = np.arange(len(metrics_to_plot))
    width  = 0.35

    lr_vals  = [logreg_metrics[m] for m in metrics_to_plot]
    xgb_vals = [xgb_metrics[m]   for m in metrics_to_plot]

    fig, ax = plt.subplots(figsize=(12, 6))

    bars_lr  = ax.bar(x - width/2, lr_vals,  width, label="Logistic Regression",
                      color=CLR_LR,  alpha=0.85, edgecolor="white")
    bars_xgb = ax.bar(x + width/2, xgb_vals, width, label="XGBoost",
                      color=CLR_XGB, alpha=0.85, edgecolor="white")

    for bars in [bars_lr, bars_xgb]:
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold"
            )

    ax.set_title("Model Comparison — Test Set Metrics",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n").upper() for m in metrics_to_plot])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.12)
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}"))
    sns.despine()

    _save(fig, "model_comparison.png")


# ── Business interpretation ───────────────────────────────────────────────────

def log_business_interpretation(logreg_metrics: dict,
                                xgb_metrics: dict,
                                y_test: pd.Series) -> None:
    """
    Translate model metrics into business language.
    This framing is what makes the project stand out in interviews
    and on the dashboard — not just numbers, but what they mean.
    """
    total_test    = len(y_test)
    actual_churners = y_test.sum()

    log.info("── Business Interpretation ─────────────────────────────")
    log.info(f"  Test set: {total_test:,} customers | {actual_churners:,} actual churners")
    log.info("")

    for metrics, name in [(logreg_metrics, "Logistic Regression"),
                          (xgb_metrics,    "XGBoost")]:
        caught    = int(metrics["recall"] * actual_churners)
        missed    = actual_churners - caught
        false_alg = int((metrics["precision"] and
                         caught / metrics["precision"] * (1 - metrics["precision"]))
                         if metrics["precision"] > 0 else 0)
        log.info(f"  {name}:")
        log.info(f"    Churners correctly identified : {caught:,} / {actual_churners:,} "
                 f"({metrics['recall']*100:.1f}% recall)")
        log.info(f"    Churners missed               : {missed:,}")
        log.info(f"    ROC-AUC                       : {metrics['roc_auc']:.4f}")
        log.info(f"    F1-Score                      : {metrics['f1']:.4f}")
        log.info("")

    log.info("  Business framing:")
    log.info("    Each correctly identified churner = opportunity to intervene")
    log.info("    Avg customer acquisition cost ~$300 (industry estimate)")
    xgb_caught = int(xgb_metrics["recall"] * actual_churners)
    log.info(f"    XGBoost catching {xgb_caught} churners ≈ "
             f"${xgb_caught * 300:,} retention value per cycle")
    log.info("────────────────────────────────────────────────────────")


# ── Save helper ───────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, filename: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved plot → outputs/figures/{filename}")


# ── Main entry point ──────────────────────────────────────────────────────────

def run_evaluation() -> None:
    """Load saved models and test set, generate all metrics and plots."""
    log.info("═══════════════════════════════════════════════════════")
    log.info("  PHASE 3b — MODEL EVALUATION                        ")
    log.info("═══════════════════════════════════════════════════════")

    X_test, y_test  = load_test_set()
    logreg, xgb     = load_models()

    log.info("── Computing test-set metrics ──────────────────────────")
    logreg_metrics = compute_metrics(logreg, "LogisticRegression", X_test, y_test)
    xgb_metrics    = compute_metrics(xgb,    "XGBoost",            X_test, y_test)

    # Save metrics CSV
    metrics_df = pd.DataFrame([logreg_metrics, xgb_metrics])
    METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(METRICS_CSV, index=False)
    log.info(f"Test metrics saved → {METRICS_CSV.relative_to(ROOT_DIR)}")

    log.info("── Generating evaluation plots ─────────────────────────")
    plot_roc_curves(logreg, xgb, X_test, y_test)
    plot_pr_curves(logreg, xgb, X_test, y_test)
    plot_confusion_matrices(logreg, xgb, X_test, y_test)
    plot_feature_importance(xgb, X_test)
    plot_model_comparison(logreg_metrics, xgb_metrics)

    log_business_interpretation(logreg_metrics, xgb_metrics, y_test)

    log.info("Phase 3b complete — evaluation successful ✓")


if __name__ == "__main__":
    run_evaluation()