"""
train.py
--------
Phase 3 — Model Training

Responsibilities:
  1. Load feature matrix from data/processed/customers_features.csv
  2. Split into train/test sets (80/20, stratified, reproducible)
  3. Save train/test sets for evaluate.py and dashboard reuse
  4. Train Logistic Regression with class_weight='balanced'
  5. Train XGBoost with scale_pos_weight for imbalance handling
  6. Run Stratified 5-Fold cross-validation on both models
  7. Save both trained models to models/
  8. Save CV comparison to outputs/model_comparison.csv

Design decisions:
  - Stratified split: preserves ~16% churn ratio in both train and test sets
  - random_state=42: full reproducibility across runs and machines
  - class_weight / scale_pos_weight: handles imbalance without SMOTE
    (SMOTE risks leakage if not handled carefully inside CV folds)
  - Cross-validation run on TRAIN set only — test set never touched
    until final evaluation in evaluate.py
  - Both models saved with joblib for fast loading in dashboard
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from src.logger import get_logger

log = get_logger("train")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR     = Path(__file__).resolve().parent.parent
FEATURES_CSV = ROOT_DIR / "data" / "processed" / "customers_features.csv"
TRAIN_CSV    = ROOT_DIR / "data" / "processed" / "train_set.csv"
TEST_CSV     = ROOT_DIR / "data" / "processed" / "test_set.csv"
MODELS_DIR   = ROOT_DIR / "models"
XGBOOST_PATH = MODELS_DIR / "xgboost_model.pkl"
LOGREG_PATH  = MODELS_DIR / "logreg_model.pkl"

# ── Constants ─────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.20
CV_FOLDS     = 5
TARGET_COL   = "churn"

CV_SCORING = {
    "roc_auc"  : "roc_auc",
    "f1"       : "f1",
    "precision": "precision",
    "recall"   : "recall",
}


# ── Data loading & splitting ──────────────────────────────────────────────────

def load_and_split() -> tuple:
    """
    Load feature matrix and create stratified train/test split.

    Stratify=y ensures both splits maintain the ~16% churn ratio.
    Test set is saved to CSV so evaluate.py and the dashboard can
    load held-out predictions without needing to retrain.
    """
    log.info(f"Loading feature matrix: {FEATURES_CSV}")
    df = pd.read_csv(FEATURES_CSV)
    log.info(f"Loaded {len(df):,} rows × {df.shape[1]} columns")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    log.info(f"Features : {X.shape[1]} columns")
    log.info(f"Target   : {(y==0).sum():,} retained | {(y==1).sum():,} churned "
             f"({y.mean()*100:.2f}% churn rate)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    log.info(f"Train    : {len(X_train):,} rows  (churn: {y_train.mean()*100:.2f}%)")
    log.info(f"Test     : {len(X_test):,} rows  (churn: {y_test.mean()*100:.2f}%)")

    # Save splits — used by evaluate.py and the Streamlit dashboard
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = X_train.copy()
    train_df[TARGET_COL] = y_train.values
    train_df.to_csv(TRAIN_CSV, index=False)

    test_df = X_test.copy()
    test_df[TARGET_COL] = y_test.values
    test_df.to_csv(TEST_CSV, index=False)

    log.info(f"Saved train set → {TRAIN_CSV.relative_to(ROOT_DIR)}")
    log.info(f"Saved test set  → {TEST_CSV.relative_to(ROOT_DIR)}")

    return X_train, X_test, y_train, y_test


# ── Cross-validation ──────────────────────────────────────────────────────────

def run_cross_validation(model, model_name: str,
                         X_train: pd.DataFrame,
                         y_train: pd.Series) -> dict:
    """
    Run Stratified K-Fold CV on the training set only.

    StratifiedKFold ensures each fold preserves the class ratio —
    essential when minority class is only 16% of data.
    n_jobs=-1 uses all available CPU cores.
    """
    log.info(f"Running {CV_FOLDS}-fold stratified CV for {model_name}...")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = cross_validate(
        model, X_train, y_train,
        cv=cv,
        scoring=CV_SCORING,
        return_train_score=False,
        n_jobs=-1
    )

    summary = {}
    log.info(f"  {model_name} CV results:")
    for metric in CV_SCORING:
        scores = results[f"test_{metric}"]
        mean, std = scores.mean(), scores.std()
        summary[metric] = {"mean": round(mean, 4), "std": round(std, 4)}
        log.info(f"    {metric:<12}: {mean:.4f} ± {std:.4f}")

    return summary


# ── Model builders ────────────────────────────────────────────────────────────

def build_logistic_regression() -> LogisticRegression:
    """
    Logistic Regression baseline model.

    class_weight='balanced': weights each class inversely to its frequency.
    For 16% churn → churned customers receive weight ~5.2x higher.
    This prevents the model from ignoring the minority class.

    max_iter=1000: default 100 often fails to converge on scaled multi-feature data.
    C=1.0: default L2 regularisation — appropriate after StandardScaler.
    solver='lbfgs': best general-purpose solver for binary L2 classification.
    """
    return LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )


def build_xgboost(y_train: pd.Series) -> XGBClassifier:
    """
    XGBoost primary model.

    scale_pos_weight = count(negative) / count(positive)
        Tells XGBoost each churned customer is worth ~5.2 retained customers.
        Equivalent to class_weight='balanced' for gradient boosting.

    n_estimators=500 + learning_rate=0.05:
        Small learning rate with more trees generalises better than
        large rate with few trees.

    subsample=0.8, colsample_bytree=0.8:
        Row and column subsampling reduce overfitting (similar to Random Forest).

    eval_metric='auc': aligns training objective with our primary metric.
    verbosity=0: suppresses XGBoost's internal output (we use loguru).
    """
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = round(neg / pos, 4)
    log.info(f"XGBoost scale_pos_weight = {neg:,} / {pos:,} = {spw}")

    return XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        eval_metric="auc",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )


# ── Training functions ────────────────────────────────────────────────────────

def train_logistic_regression(X_train, y_train) -> tuple:
    """Run CV, then fit Logistic Regression on full training set."""
    log.info("── Logistic Regression ─────────────────────────────────")
    model = build_logistic_regression()
    cv_results = run_cross_validation(model, "LogisticRegression", X_train, y_train)
    log.info("Fitting on full training set...")
    model.fit(X_train, y_train)
    log.info("Logistic Regression training complete ✓")
    return model, cv_results


def train_xgboost(X_train, y_train) -> tuple:
    """Run CV, then fit XGBoost on full training set."""
    log.info("── XGBoost ─────────────────────────────────────────────")
    model = build_xgboost(y_train)
    cv_results = run_cross_validation(model, "XGBoost", X_train, y_train)
    log.info("Fitting on full training set...")
    model.fit(X_train, y_train)
    log.info("XGBoost training complete ✓")
    return model, cv_results


# ── Save & compare ────────────────────────────────────────────────────────────

def save_models(logreg, xgb) -> None:
    """Persist both fitted models to models/ directory."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(logreg, LOGREG_PATH)
    joblib.dump(xgb,    XGBOOST_PATH)
    log.info(f"Saved LogisticRegression → {LOGREG_PATH.relative_to(ROOT_DIR)}")
    log.info(f"Saved XGBoost            → {XGBOOST_PATH.relative_to(ROOT_DIR)}")


def save_cv_comparison(logreg_cv: dict, xgb_cv: dict) -> None:
    """Save cross-validation comparison table to outputs/model_comparison.csv."""
    rows = []
    for metric in CV_SCORING:
        rows.append({
            "metric"      : metric,
            "logreg_mean" : logreg_cv[metric]["mean"],
            "logreg_std"  : logreg_cv[metric]["std"],
            "xgboost_mean": xgb_cv[metric]["mean"],
            "xgboost_std" : xgb_cv[metric]["std"],
        })
    df = pd.DataFrame(rows)
    out = ROOT_DIR / "outputs" / "model_comparison.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    log.info(f"CV comparison saved → {out.relative_to(ROOT_DIR)}")


def log_cv_comparison(logreg_cv: dict, xgb_cv: dict) -> None:
    """Print side-by-side CV comparison to log."""
    log.info("── CV Comparison Summary ───────────────────────────────")
    log.info(f"  {'Metric':<12}  {'LogReg mean±std':<22}  {'XGBoost mean±std'}")
    log.info(f"  {'─'*12}  {'─'*22}  {'─'*20}")
    for metric in CV_SCORING:
        lr = logreg_cv[metric]
        xg = xgb_cv[metric]
        log.info(
            f"  {metric:<12}  "
            f"{lr['mean']:.4f} ± {lr['std']:.4f}        "
            f"{xg['mean']:.4f} ± {xg['std']:.4f}"
        )
    log.info("────────────────────────────────────────────────────────")


# ── Main entry point ──────────────────────────────────────────────────────────

def run_training() -> dict:
    """
    Execute the full training pipeline.
    Returns a results dict consumed directly by run_phase_3() in run_pipeline.py.
    """
    log.info("═══════════════════════════════════════════════════════")
    log.info("  PHASE 3a — MODEL TRAINING                          ")
    log.info("═══════════════════════════════════════════════════════")

    X_train, X_test, y_train, y_test = load_and_split()

    logreg, logreg_cv = train_logistic_regression(X_train, y_train)
    xgb,    xgb_cv    = train_xgboost(X_train, y_train)

    save_models(logreg, xgb)
    log_cv_comparison(logreg_cv, xgb_cv)
    save_cv_comparison(logreg_cv, xgb_cv)

    log.info("Phase 3a complete — model training successful ✓")

    return {
        "logreg"    : logreg,
        "xgb"       : xgb,
        "X_train"   : X_train,
        "X_test"    : X_test,
        "y_train"   : y_train,
        "y_test"    : y_test,
        "logreg_cv" : logreg_cv,
        "xgb_cv"    : xgb_cv,
    }


if __name__ == "__main__":
    run_training()