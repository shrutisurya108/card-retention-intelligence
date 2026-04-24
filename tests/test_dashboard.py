"""
test_dashboard.py
-----------------
Phase 5 Tests — Dashboard Pre-flight Checks

These tests verify all assets the dashboard depends on exist and are
loadable before launching. They serve as a pre-flight checklist.

Unlike other phases, the dashboard itself is tested manually by running
it with Streamlit. Automated UI testing of Streamlit apps requires
additional tooling (playwright / selenium) which is out of scope here.

Tests verify:
  1.  XGBoost model is loadable
  2.  Logistic Regression model is loadable
  3.  StandardScaler is loadable
  4.  SHAP values are loadable and correct shape
  5.  SHAP expected value is loadable
  6.  SHAP feature names JSON is loadable
  7.  Feature count consistency: scaler, SHAP, and test set agree
  8.  Test metrics CSV is loadable and has both models
  9.  CV comparison CSV is loadable
  10. All 10 EDA plots exist
  11. All 5 evaluation plots exist
  12. All 4 SHAP plots exist
  13. Models produce valid probabilities on test set
  14. Feature names in SHAP JSON match test set columns

Run with:
    python -m pytest tests/test_dashboard.py -v
"""

import pytest
import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR     = Path(__file__).resolve().parent.parent
MODELS_DIR   = ROOT_DIR / "models"
OUTPUTS_DIR  = ROOT_DIR / "outputs"
FIGURES_DIR  = ROOT_DIR / "outputs" / "figures"
DATA_DIR     = ROOT_DIR / "data" / "processed"

XGBOOST_PATH        = MODELS_DIR / "xgboost_model.pkl"
LOGREG_PATH         = MODELS_DIR / "logreg_model.pkl"
SCALER_PATH         = MODELS_DIR / "scaler.pkl"
SHAP_VALUES_PATH    = OUTPUTS_DIR / "shap_values.npy"
SHAP_EXPECTED_PATH  = OUTPUTS_DIR / "shap_expected_value.npy"
SHAP_FEATNAMES_PATH = OUTPUTS_DIR / "shap_feature_names.json"
METRICS_CSV         = OUTPUTS_DIR / "test_metrics.csv"
CV_CSV              = OUTPUTS_DIR / "model_comparison.csv"
TEST_CSV            = DATA_DIR    / "test_set.csv"
TARGET_COL          = "churn"

EDA_PLOTS = [
    "churn_distribution.png", "age_distribution.png",
    "churn_by_age.png", "credit_limit_dist.png",
    "trans_count_vs_churn.png", "correlation_heatmap.png",
    "churn_by_category.png", "numeric_distributions.png",
    "utilisation_vs_churn.png", "contacts_vs_churn.png",
]
EVAL_PLOTS = [
    "roc_curves.png", "pr_curves.png", "confusion_matrices.png",
    "feature_importance.png", "model_comparison.png",
]
SHAP_PLOTS = [
    "shap_summary.png", "shap_importance_bar.png",
    "shap_waterfall_high_risk.png", "shap_dependence_trans_ct.png",
]


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def xgb():
    return joblib.load(XGBOOST_PATH)


@pytest.fixture(scope="module")
def logreg():
    return joblib.load(LOGREG_PATH)


@pytest.fixture(scope="module")
def scaler():
    return joblib.load(SCALER_PATH)


@pytest.fixture(scope="module")
def shap_values():
    return np.load(SHAP_VALUES_PATH)


@pytest.fixture(scope="module")
def feature_names():
    with open(SHAP_FEATNAMES_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def test_data():
    df = pd.read_csv(TEST_CSV)
    return df.drop(columns=[TARGET_COL]), df[TARGET_COL]


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestModelAssets:

    def test_xgboost_loadable(self, xgb):
        assert xgb is not None

    def test_logreg_loadable(self, logreg):
        assert logreg is not None

    def test_scaler_loadable(self, scaler):
        assert scaler is not None

    def test_scaler_is_fitted(self, scaler):
        assert hasattr(scaler, "mean_")
        assert len(scaler.mean_) > 0

    def test_xgboost_produces_valid_proba(self, xgb, test_data):
        X, _ = test_data
        proba = xgb.predict_proba(X[:10])[:, 1]
        assert proba.min() >= 0.0
        assert proba.max() <= 1.0

    def test_logreg_produces_valid_proba(self, logreg, test_data):
        X, _ = test_data
        proba = logreg.predict_proba(X[:10])[:, 1]
        assert proba.min() >= 0.0
        assert proba.max() <= 1.0


class TestSHAPAssets:

    def test_shap_values_loadable(self, shap_values):
        assert isinstance(shap_values, np.ndarray)
        assert shap_values.ndim == 2

    def test_shap_expected_value_loadable(self):
        ev = np.load(SHAP_EXPECTED_PATH)
        assert ev is not None
        assert np.isfinite(float(ev.flat[0]))

    def test_shap_feature_names_loadable(self, feature_names):
        assert isinstance(feature_names, list)
        assert len(feature_names) > 0

    def test_shap_values_no_nan(self, shap_values):
        assert np.isnan(shap_values).sum() == 0

    def test_shap_values_no_inf(self, shap_values):
        assert np.isinf(shap_values).sum() == 0


class TestFeatureConsistency:
    """
    Critical: all components must agree on the number and order of features.
    Mismatches here would cause silent errors in the dashboard scorer.
    """

    def test_scaler_feature_count(self, scaler, feature_names):
        """Scaler must have same number of features as SHAP feature names."""
        assert len(scaler.mean_) == len(feature_names), (
            f"Scaler has {len(scaler.mean_)} features, "
            f"SHAP has {len(feature_names)} feature names."
        )

    def test_shap_columns_match_test_set(self, shap_values, feature_names, test_data):
        """SHAP values columns must match test set columns."""
        X_test, _ = test_data
        assert shap_values.shape[1] == len(X_test.columns), (
            f"SHAP cols {shap_values.shape[1]} != test cols {len(X_test.columns)}"
        )

    def test_feature_names_match_test_columns(self, feature_names, test_data):
        """Feature names list must exactly match test set column order."""
        X_test, _ = test_data
        assert feature_names == X_test.columns.tolist(), (
            "Feature names in SHAP JSON do not match test set column order. "
            "Dashboard scorer will produce incorrect predictions."
        )

    def test_xgboost_feature_count(self, xgb, feature_names):
        """XGBoost model must expect same number of features."""
        assert xgb.n_features_in_ == len(feature_names), (
            f"XGBoost expects {xgb.n_features_in_} features, "
            f"got {len(feature_names)} feature names."
        )


class TestMetricsCSVs:

    def test_metrics_csv_exists(self):
        assert METRICS_CSV.exists()

    def test_metrics_csv_has_both_models(self):
        df = pd.read_csv(METRICS_CSV)
        assert len(df) == 2

    def test_metrics_csv_has_roc_auc(self):
        df = pd.read_csv(METRICS_CSV)
        assert "roc_auc" in df.columns

    def test_cv_csv_exists(self):
        assert CV_CSV.exists()

    def test_cv_csv_has_correct_metrics(self):
        df = pd.read_csv(CV_CSV)
        assert "roc_auc" in df["metric"].values


class TestPlotAssets:

    @pytest.mark.parametrize("plot", EDA_PLOTS)
    def test_eda_plot_exists(self, plot):
        assert (FIGURES_DIR / plot).exists(), f"Missing EDA plot: {plot}"

    @pytest.mark.parametrize("plot", EVAL_PLOTS)
    def test_eval_plot_exists(self, plot):
        assert (FIGURES_DIR / plot).exists(), f"Missing eval plot: {plot}"

    @pytest.mark.parametrize("plot", SHAP_PLOTS)
    def test_shap_plot_exists(self, plot):
        assert (FIGURES_DIR / plot).exists(), f"Missing SHAP plot: {plot}"

    @pytest.mark.parametrize("plot", EDA_PLOTS + EVAL_PLOTS + SHAP_PLOTS)
    def test_plot_is_not_empty(self, plot):
        path = FIGURES_DIR / plot
        if path.exists():
            assert path.stat().st_size > 5_000, (
                f"Plot '{plot}' is suspiciously small — may be corrupt."
            )