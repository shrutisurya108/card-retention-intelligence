"""
test_train.py
-------------
Phase 3 Tests — Model Training & Evaluation

Tests verify:
  1.  Both model files exist after training
  2.  Both models are loadable with joblib
  3.  Both models can call .predict() and .predict_proba()
  4.  Prediction shapes match test set size
  5.  Probabilities are in valid range [0, 1]
  6.  XGBoost ROC-AUC > 0.90 on test set
  7.  Logistic Regression ROC-AUC > 0.80 on test set
  8.  Train and test CSV files exist
  9.  No data leakage: train and test sets don't overlap
  10. model_comparison.csv exists
  11. test_metrics.csv exists
  12. All 5 evaluation plots exist
  13. XGBoost outperforms Logistic Regression on ROC-AUC
  14. Both models have correct sklearn types

Run with:
    python -m pytest tests/test_train.py -v
"""

import pytest
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR     = Path(__file__).resolve().parent.parent
MODELS_DIR   = ROOT_DIR / "models"
OUTPUTS_DIR  = ROOT_DIR / "outputs"
FIGURES_DIR  = OUTPUTS_DIR / "figures"

XGBOOST_PATH = MODELS_DIR / "xgboost_model.pkl"
LOGREG_PATH  = MODELS_DIR / "logreg_model.pkl"
TRAIN_CSV    = ROOT_DIR / "data" / "processed" / "train_set.csv"
TEST_CSV     = ROOT_DIR / "data" / "processed" / "test_set.csv"
METRICS_CSV  = OUTPUTS_DIR / "test_metrics.csv"
CV_CSV       = OUTPUTS_DIR / "model_comparison.csv"

TARGET_COL   = "churn"

REQUIRED_PLOTS = [
    "roc_curves.png",
    "pr_curves.png",
    "confusion_matrices.png",
    "feature_importance.png",
    "model_comparison.png",
]

# Performance thresholds — conservative minimums, real results will exceed these
MIN_XGB_AUC    = 0.90
MIN_LOGREG_AUC = 0.80


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def logreg():
    assert LOGREG_PATH.exists(), (
        f"Logistic Regression model not found: {LOGREG_PATH}\n"
        "Run `python run_pipeline.py --phase 3` first."
    )
    return joblib.load(LOGREG_PATH)


@pytest.fixture(scope="module")
def xgb():
    assert XGBOOST_PATH.exists(), (
        f"XGBoost model not found: {XGBOOST_PATH}\n"
        "Run `python run_pipeline.py --phase 3` first."
    )
    return joblib.load(XGBOOST_PATH)


@pytest.fixture(scope="module")
def test_data():
    assert TEST_CSV.exists(), f"Test set not found: {TEST_CSV}"
    df = pd.read_csv(TEST_CSV)
    X  = df.drop(columns=[TARGET_COL])
    y  = df[TARGET_COL]
    return X, y


@pytest.fixture(scope="module")
def train_data():
    assert TRAIN_CSV.exists(), f"Train set not found: {TRAIN_CSV}"
    return pd.read_csv(TRAIN_CSV)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestModelFiles:

    def test_logreg_file_exists(self):
        assert LOGREG_PATH.exists(), f"Missing: {LOGREG_PATH}"

    def test_xgboost_file_exists(self):
        assert XGBOOST_PATH.exists(), f"Missing: {XGBOOST_PATH}"

    def test_logreg_correct_type(self, logreg):
        assert isinstance(logreg, LogisticRegression), (
            f"Expected LogisticRegression, got {type(logreg)}"
        )

    def test_xgboost_correct_type(self, xgb):
        assert isinstance(xgb, XGBClassifier), (
            f"Expected XGBClassifier, got {type(xgb)}"
        )

    def test_logreg_is_fitted(self, logreg):
        """Fitted LogisticRegression has coef_ attribute."""
        assert hasattr(logreg, "coef_"), "Logistic Regression is not fitted."

    def test_xgboost_is_fitted(self, xgb):
        """Fitted XGBoost has feature_importances_ attribute."""
        assert hasattr(xgb, "feature_importances_"), "XGBoost is not fitted."


class TestPredictions:

    def test_logreg_predict_shape(self, logreg, test_data):
        X, y = test_data
        preds = logreg.predict(X)
        assert len(preds) == len(y), (
            f"Prediction length {len(preds)} != test set length {len(y)}"
        )

    def test_xgboost_predict_shape(self, xgb, test_data):
        X, y = test_data
        preds = xgb.predict(X)
        assert len(preds) == len(y)

    def test_logreg_predict_proba_shape(self, logreg, test_data):
        X, y = test_data
        proba = logreg.predict_proba(X)
        assert proba.shape == (len(y), 2), (
            f"Expected shape ({len(y)}, 2), got {proba.shape}"
        )

    def test_xgboost_predict_proba_shape(self, xgb, test_data):
        X, y = test_data
        proba = xgb.predict_proba(X)
        assert proba.shape == (len(y), 2)

    def test_logreg_probabilities_valid_range(self, logreg, test_data):
        X, _ = test_data
        proba = logreg.predict_proba(X)[:, 1]
        assert proba.min() >= 0.0, "Probabilities below 0"
        assert proba.max() <= 1.0, "Probabilities above 1"

    def test_xgboost_probabilities_valid_range(self, xgb, test_data):
        X, _ = test_data
        proba = xgb.predict_proba(X)[:, 1]
        assert proba.min() >= 0.0
        assert proba.max() <= 1.0

    def test_predictions_are_binary(self, xgb, test_data):
        """Model predictions must be 0 or 1 only."""
        X, _ = test_data
        preds = xgb.predict(X)
        assert set(np.unique(preds)).issubset({0, 1}), (
            f"Predictions contain non-binary values: {np.unique(preds)}"
        )


class TestModelPerformance:

    def test_xgboost_auc_above_threshold(self, xgb, test_data):
        """
        XGBoost must achieve ROC-AUC > 0.90 on the test set.
        A well-tuned XGBoost on this dataset typically achieves ~0.97.
        This test catches model degradation or data issues.
        """
        X, y = test_data
        proba = xgb.predict_proba(X)[:, 1]
        auc   = roc_auc_score(y, proba)
        assert auc > MIN_XGB_AUC, (
            f"XGBoost AUC {auc:.4f} is below minimum threshold {MIN_XGB_AUC}."
        )

    def test_logreg_auc_above_threshold(self, logreg, test_data):
        """
        Logistic Regression must achieve ROC-AUC > 0.80 on the test set.
        This is the baseline bar — anything below suggests a pipeline issue.
        """
        X, y = test_data
        proba = logreg.predict_proba(X)[:, 1]
        auc   = roc_auc_score(y, proba)
        assert auc > MIN_LOGREG_AUC, (
            f"Logistic Regression AUC {auc:.4f} is below minimum {MIN_LOGREG_AUC}."
        )

    def test_xgboost_outperforms_logreg(self, logreg, xgb, test_data):
        """
        XGBoost must have higher ROC-AUC than Logistic Regression.
        If this fails, something is wrong with either model's training.
        """
        X, y = test_data
        lr_auc  = roc_auc_score(y, logreg.predict_proba(X)[:, 1])
        xgb_auc = roc_auc_score(y, xgb.predict_proba(X)[:, 1])
        assert xgb_auc > lr_auc, (
            f"XGBoost AUC ({xgb_auc:.4f}) should exceed "
            f"Logistic Regression AUC ({lr_auc:.4f})."
        )

    def test_both_models_beat_majority_class_baseline(self, logreg, xgb, test_data):
        """
        Both models must beat the naive majority-class baseline.
        Majority class baseline AUC = 0.50 (random).
        Any AUC <= 0.50 means the model is worse than random.
        """
        X, y = test_data
        for model, name in [(logreg, "LogReg"), (xgb, "XGBoost")]:
            auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
            assert auc > 0.50, f"{name} AUC {auc:.4f} does not beat random baseline."


class TestDataSplits:

    def test_train_csv_exists(self):
        assert TRAIN_CSV.exists(), f"Missing: {TRAIN_CSV}"

    def test_test_csv_exists(self):
        assert TEST_CSV.exists(), f"Missing: {TEST_CSV}"

    def test_no_data_leakage(self, train_data, test_data):
        """
        Train and test sets must not share any rows.
        We check this by verifying no overlap in all feature values combined.
        Data leakage would cause artificially inflated metrics.
        """
        X_test, _ = test_data
        # Use a subset of columns as a fingerprint
        train_fingerprints = set(
            train_data.drop(columns=[TARGET_COL])
            .round(6).astype(str).agg("_".join, axis=1)
        )
        test_fingerprints = set(
            X_test.round(6).astype(str).agg("_".join, axis=1)
        )
        overlap = train_fingerprints & test_fingerprints
        assert len(overlap) == 0, (
            f"Data leakage detected: {len(overlap)} rows appear in both "
            "train and test sets."
        )

    def test_train_test_split_ratio(self, train_data, test_data):
        """Test set should be approximately 20% of total data."""
        X_test, _ = test_data
        total = len(train_data) + len(X_test)
        test_ratio = len(X_test) / total
        assert 0.18 <= test_ratio <= 0.22, (
            f"Test split ratio {test_ratio:.2%} is outside expected 18-22% range."
        )

    def test_stratification_preserved(self, train_data, test_data):
        """Both splits should have ~16% churn rate (stratification check)."""
        X_test, y_test = test_data
        train_rate = train_data[TARGET_COL].mean()
        test_rate  = y_test.mean()
        assert abs(train_rate - test_rate) < 0.02, (
            f"Churn rate differs between train ({train_rate:.3f}) "
            f"and test ({test_rate:.3f}) — stratification may have failed."
        )


class TestOutputFiles:

    def test_metrics_csv_exists(self):
        assert METRICS_CSV.exists(), f"Missing: {METRICS_CSV}"

    def test_cv_comparison_csv_exists(self):
        assert CV_CSV.exists(), f"Missing: {CV_CSV}"

    def test_metrics_csv_has_both_models(self):
        df = pd.read_csv(METRICS_CSV)
        assert len(df) == 2, f"Expected 2 model rows, got {len(df)}"

    def test_metrics_csv_has_required_columns(self):
        df = pd.read_csv(METRICS_CSV)
        required = ["model", "roc_auc", "f1", "precision", "recall", "accuracy"]
        missing  = [c for c in required if c not in df.columns]
        assert len(missing) == 0, f"Missing columns in metrics CSV: {missing}"

    @pytest.mark.parametrize("plot_file", REQUIRED_PLOTS)
    def test_evaluation_plot_exists(self, plot_file):
        path = FIGURES_DIR / plot_file
        assert path.exists(), f"Missing evaluation plot: {plot_file}"