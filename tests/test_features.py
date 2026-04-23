"""
test_features.py
----------------
Phase 2 Tests — Feature Engineering

Tests verify:
  1.  Output CSV file exists
  2.  Row count is preserved (10,127)
  3.  No string/object columns remain (all encoded)
  4.  No null values anywhere in the feature matrix
  5.  CLIENTNUM was dropped (ID column)
  6.  Original Attrition_Flag not present
  7.  All 3 engineered features are present
  8.  Target column 'churn' is present and still binary
  9.  Scaler file was saved to models/scaler.pkl
  10. Scaler can be loaded and used to transform data

Run with:
    python -m pytest tests/test_features.py -v
"""

import pytest
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR     = Path(__file__).resolve().parent.parent
FEATURES_CSV = ROOT_DIR / "data" / "processed" / "customers_features.csv"
SCALER_PATH  = ROOT_DIR / "models" / "scaler.pkl"
EXPECTED_ROWS = 10127

# The 3 engineered features that must exist
ENGINEERED_FEATURES = [
    "transaction_velocity",
    "inactivity_risk",
    "credit_usage_gap",
]


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def features_df():
    """Load the feature matrix CSV once for all tests in this module."""
    assert FEATURES_CSV.exists(), (
        f"Features CSV not found at {FEATURES_CSV}.\n"
        "Run `python run_pipeline.py --phase 2` first."
    )
    return pd.read_csv(FEATURES_CSV)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestOutputFiles:

    def test_features_csv_exists(self):
        """Feature matrix CSV must be created by feature engineering."""
        assert FEATURES_CSV.exists(), f"Missing: {FEATURES_CSV}"

    def test_scaler_file_exists(self):
        """Fitted scaler must be saved to models/scaler.pkl."""
        assert SCALER_PATH.exists(), f"Missing scaler: {SCALER_PATH}"


class TestRowIntegrity:

    def test_row_count_preserved(self, features_df):
        """Feature engineering must not drop or duplicate any rows."""
        assert len(features_df) == EXPECTED_ROWS, (
            f"Expected {EXPECTED_ROWS:,} rows, got {len(features_df):,}."
        )


class TestEncodingQuality:

    def test_no_string_columns(self, features_df):
        """
        All categorical columns must be encoded.
        No object/string dtype columns should remain in the feature matrix.
        """
        object_cols = list(features_df.select_dtypes(include="object").columns)
        assert len(object_cols) == 0, (
            f"String columns still present (not encoded): {object_cols}"
        )

    def test_no_null_values(self, features_df):
        """Feature matrix must be completely null-free."""
        null_counts = features_df.isnull().sum()
        null_cols   = null_counts[null_counts > 0]
        assert null_cols.empty, (
            f"Null values found in columns:\n{null_cols}"
        )

    def test_clientnum_dropped(self, features_df):
        """CLIENTNUM (ID column) must not be in the feature matrix."""
        assert "CLIENTNUM" not in features_df.columns, (
            "CLIENTNUM is still present — ID columns must be removed before modelling."
        )

    def test_attrition_flag_not_present(self, features_df):
        """Original string target column must not be present."""
        assert "Attrition_Flag" not in features_df.columns


class TestEngineeredFeatures:

    def test_transaction_velocity_exists(self, features_df):
        """transaction_velocity feature must exist."""
        assert "transaction_velocity" in features_df.columns

    def test_inactivity_risk_exists(self, features_df):
        """inactivity_risk feature must exist."""
        assert "inactivity_risk" in features_df.columns

    def test_credit_usage_gap_exists(self, features_df):
        """credit_usage_gap feature must exist."""
        assert "credit_usage_gap" in features_df.columns

    def test_all_engineered_features_present(self, features_df):
        """All 3 engineered features must be present."""
        missing = [f for f in ENGINEERED_FEATURES if f not in features_df.columns]
        assert len(missing) == 0, f"Missing engineered features: {missing}"

    def test_engineered_features_are_numeric(self, features_df):
        """All engineered features must be numeric (float or int)."""
        for feat in ENGINEERED_FEATURES:
            if feat in features_df.columns:
                assert pd.api.types.is_numeric_dtype(features_df[feat]), (
                    f"Engineered feature '{feat}' is not numeric."
                )

    def test_no_inf_values_in_engineered_features(self, features_df):
        """Engineered features must not contain infinite values."""
        for feat in ENGINEERED_FEATURES:
            if feat in features_df.columns:
                inf_count = np.isinf(features_df[feat]).sum()
                assert inf_count == 0, (
                    f"Feature '{feat}' contains {inf_count} infinite values."
                )


class TestTargetColumn:

    def test_churn_column_present(self, features_df):
        """Target column 'churn' must survive feature engineering."""
        assert "churn" in features_df.columns

    def test_churn_still_binary(self, features_df):
        """'churn' must remain binary (0 and 1 only) after all transformations."""
        unique_vals = set(features_df["churn"].unique())
        assert unique_vals == {0, 1}, (
            f"'churn' contains unexpected values after feature engineering: {unique_vals}"
        )

    def test_churn_rate_unchanged(self, features_df):
        """Churn rate must remain ~16% — feature engineering must not alter the target."""
        churn_rate = features_df["churn"].mean()
        assert 0.10 <= churn_rate <= 0.25, (
            f"Churn rate shifted unexpectedly to {churn_rate:.2%}."
        )


class TestScaler:

    def test_scaler_loadable(self):
        """Saved scaler must be loadable with joblib."""
        scaler = joblib.load(SCALER_PATH)
        assert scaler is not None

    def test_scaler_has_correct_type(self):
        """Scaler must be a StandardScaler instance."""
        from sklearn.preprocessing import StandardScaler
        scaler = joblib.load(SCALER_PATH)
        assert isinstance(scaler, StandardScaler), (
            f"Expected StandardScaler, got {type(scaler)}"
        )

    def test_scaler_is_fitted(self):
        """Scaler must be fitted (has mean_ attribute)."""
        scaler = joblib.load(SCALER_PATH)
        assert hasattr(scaler, "mean_"), "Scaler is not fitted — call fit_transform first."
        assert len(scaler.mean_) > 0