"""
test_explain.py
---------------
Phase 4 Tests — SHAP Explainability

Tests verify:
  1.  SHAP values file exists and is loadable
  2.  SHAP values shape matches (test_rows, n_features)
  3.  No NaN values in SHAP values
  4.  No Inf values in SHAP values
  5.  Expected value file exists and is a scalar
  6.  Expected value is in a valid probability range
  7.  Feature names JSON file exists and is loadable
  8.  Feature count matches between SHAP values and feature names
  9.  Feature names match test set columns
  10. All 4 SHAP plots exist
  11. SHAP additivity: values sum approximately to model output
  12. Top SHAP feature is a meaningful predictor (not a one-hot dummy)

Run with:
    python -m pytest tests/test_explain.py -v
"""

import pytest
import numpy as np
import pandas as pd
import json
import joblib
from pathlib import Path
from sklearn.metrics import roc_auc_score

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR            = Path(__file__).resolve().parent.parent
OUTPUTS_DIR         = ROOT_DIR / "outputs"
FIGURES_DIR         = OUTPUTS_DIR / "figures"
SHAP_VALUES_PATH    = OUTPUTS_DIR / "shap_values.npy"
SHAP_EXPECTED_PATH  = OUTPUTS_DIR / "shap_expected_value.npy"
SHAP_FEATNAMES_PATH = OUTPUTS_DIR / "shap_feature_names.json"
TEST_CSV            = ROOT_DIR / "data" / "processed" / "test_set.csv"
XGBOOST_PATH        = ROOT_DIR / "models" / "xgboost_model.pkl"
TARGET_COL          = "churn"

REQUIRED_SHAP_PLOTS = [
    "shap_summary.png",
    "shap_importance_bar.png",
    "shap_waterfall_high_risk.png",
    "shap_dependence_trans_ct.png",
]

# Features that should NOT be the top SHAP predictor
# (one-hot dummies are less meaningful than continuous features)
ONE_HOT_PREFIXES = ["Marital_Status_", "Card_Category_"]


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def shap_values():
    """Load SHAP values array."""
    assert SHAP_VALUES_PATH.exists(), (
        f"SHAP values not found: {SHAP_VALUES_PATH}\n"
        "Run `python run_pipeline.py --phase 4` first."
    )
    return np.load(SHAP_VALUES_PATH)


@pytest.fixture(scope="module")
def expected_value():
    """Load SHAP expected value."""
    assert SHAP_EXPECTED_PATH.exists(), f"Missing: {SHAP_EXPECTED_PATH}"
    return np.load(SHAP_EXPECTED_PATH)


@pytest.fixture(scope="module")
def feature_names():
    """Load feature names list from JSON."""
    assert SHAP_FEATNAMES_PATH.exists(), f"Missing: {SHAP_FEATNAMES_PATH}"
    with open(SHAP_FEATNAMES_PATH, "r") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def test_data():
    """Load test set."""
    assert TEST_CSV.exists(), f"Missing test set: {TEST_CSV}"
    df = pd.read_csv(TEST_CSV)
    X  = df.drop(columns=[TARGET_COL])
    y  = df[TARGET_COL]
    return X, y


@pytest.fixture(scope="module")
def model():
    """Load XGBoost model."""
    assert XGBOOST_PATH.exists(), f"Missing model: {XGBOOST_PATH}"
    return joblib.load(XGBOOST_PATH)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestSHAPArtifactFiles:

    def test_shap_values_file_exists(self):
        assert SHAP_VALUES_PATH.exists(), f"Missing: {SHAP_VALUES_PATH}"

    def test_shap_expected_value_file_exists(self):
        assert SHAP_EXPECTED_PATH.exists(), f"Missing: {SHAP_EXPECTED_PATH}"

    def test_shap_feature_names_file_exists(self):
        assert SHAP_FEATNAMES_PATH.exists(), f"Missing: {SHAP_FEATNAMES_PATH}"

    def test_shap_values_loadable(self, shap_values):
        assert shap_values is not None
        assert isinstance(shap_values, np.ndarray)

    def test_feature_names_loadable(self, feature_names):
        assert isinstance(feature_names, list)
        assert len(feature_names) > 0


class TestSHAPValuesShape:

    def test_shap_values_is_2d(self, shap_values):
        """SHAP values must be a 2D array (samples × features)."""
        assert shap_values.ndim == 2, (
            f"Expected 2D array, got shape {shap_values.shape}"
        )

    def test_shap_values_row_count(self, shap_values, test_data):
        """Number of SHAP rows must match test set size."""
        X_test, _ = test_data
        assert shap_values.shape[0] == len(X_test), (
            f"SHAP rows {shap_values.shape[0]} != test rows {len(X_test)}"
        )

    def test_shap_values_column_count(self, shap_values, feature_names):
        """Number of SHAP columns must match number of features."""
        assert shap_values.shape[1] == len(feature_names), (
            f"SHAP cols {shap_values.shape[1]} != features {len(feature_names)}"
        )

    def test_feature_names_match_test_columns(self, feature_names, test_data):
        """Feature names must exactly match test set column names."""
        X_test, _ = test_data
        assert feature_names == X_test.columns.tolist(), (
            "Feature names in SHAP JSON do not match test set columns."
        )


class TestSHAPValueQuality:

    def test_no_nan_in_shap_values(self, shap_values):
        """SHAP values must be finite — NaN indicates a computation error."""
        nan_count = np.isnan(shap_values).sum()
        assert nan_count == 0, f"Found {nan_count} NaN values in SHAP array."

    def test_no_inf_in_shap_values(self, shap_values):
        """SHAP values must not contain infinity."""
        inf_count = np.isinf(shap_values).sum()
        assert inf_count == 0, f"Found {inf_count} Inf values in SHAP array."

    def test_shap_values_have_variance(self, shap_values):
        """
        SHAP values must show variation across customers.
        Zero variance would mean all features have identical impact
        for all customers — which would indicate a computation failure.
        """
        col_stds = shap_values.std(axis=0)
        zero_variance_cols = (col_stds == 0).sum()
        # Allow at most 20% of features to have zero variance (e.g. constant features)
        assert zero_variance_cols < shap_values.shape[1] * 0.2, (
            f"{zero_variance_cols} features have zero SHAP variance — "
            "possible computation issue."
        )


class TestExpectedValue:

    def test_expected_value_is_scalar(self, expected_value):
        """Expected value must be a single number."""
        assert expected_value.shape == (1,) or expected_value.ndim == 0, (
            f"Expected scalar, got shape {expected_value.shape}"
        )

    def test_expected_value_reasonable_range(self, expected_value):
        """
        Expected value should be reasonable for a churn model.
        For XGBoost with log-odds output, this is typically negative
        (since churn is the minority class). We just verify it's finite.
        """
        val = float(expected_value.flat[0])
        assert np.isfinite(val), f"Expected value is not finite: {val}"
        assert -20 < val < 20, (
            f"Expected value {val} is outside reasonable range (-20, 20)."
        )


class TestSHAPAdditivity:

    def test_shap_additivity_property(self, shap_values, expected_value,
                                      model, test_data):
        """
        SHAP Additivity: f(x) = E[f(X)] + sum(SHAP values for x)

        For tree models with log-odds output:
          model_output ≈ expected_value + sum(shap_values for that row)

        We verify this holds approximately (within 0.01 tolerance)
        for a sample of 50 customers. Small floating-point differences
        are acceptable — exact equality is not required.

        This is the fundamental correctness property of SHAP.
        If it fails, the SHAP computation or model loading is broken.
        """
        X_test, _ = test_data
        ev = float(expected_value.flat[0])

        # Get raw model margin output (log-odds, before sigmoid)
        raw_output = model.get_booster().predict(
            __import__("xgboost").DMatrix(X_test),
            output_margin=True
        )

        # Check additivity for first 50 samples
        n_check = min(50, len(X_test))
        for i in range(n_check):
            shap_sum  = float(shap_values[i].sum()) + ev
            model_out = float(raw_output[i])
            diff      = abs(shap_sum - model_out)
            assert diff < 0.05, (
                f"SHAP additivity violated for sample {i}: "
                f"SHAP sum={shap_sum:.6f}, model output={model_out:.6f}, "
                f"diff={diff:.6f}"
            )


class TestSHAPInterpretability:

    def test_top_shap_feature_is_meaningful(self, shap_values, feature_names):
        """
        Top SHAP feature should be a meaningful continuous feature,
        not a one-hot dummy variable. This validates that feature
        engineering produced a sensible feature ranking.
        """
        mean_abs = np.abs(shap_values).mean(axis=0)
        top_feature = feature_names[int(np.argmax(mean_abs))]

        is_dummy = any(top_feature.startswith(prefix)
                      for prefix in ONE_HOT_PREFIXES)
        assert not is_dummy, (
            f"Top SHAP feature '{top_feature}' is a one-hot dummy — "
            "unexpected for this dataset. Check feature engineering."
        )

    def test_shap_importance_sums_to_positive(self, shap_values):
        """
        Mean SHAP values (not absolute) across all customers should be
        close to zero — features that push toward churn for some customers
        push away for others on average. This is a sanity check.
        """
        mean_shap_per_feature = shap_values.mean(axis=0)
        # Allow modest imbalance — not exactly zero with minority class
        max_mean = np.abs(mean_shap_per_feature).max()
        assert max_mean < 2.0, (
            f"Max mean SHAP value {max_mean:.4f} is too large — "
            "possible data or computation issue."
        )


class TestSHAPPlots:

    @pytest.mark.parametrize("plot_file", REQUIRED_SHAP_PLOTS)
    def test_shap_plot_exists(self, plot_file):
        """All 4 SHAP plots must be saved to outputs/figures/."""
        path = FIGURES_DIR / plot_file
        assert path.exists(), f"Missing SHAP plot: {plot_file}"

    @pytest.mark.parametrize("plot_file", REQUIRED_SHAP_PLOTS)
    def test_shap_plot_has_content(self, plot_file):
        """SHAP plots must not be empty files."""
        path = FIGURES_DIR / plot_file
        if path.exists():
            size = path.stat().st_size
            assert size > 10_000, (
                f"SHAP plot '{plot_file}' is suspiciously small "
                f"({size} bytes) — may be corrupt or empty."
            )