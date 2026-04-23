"""
features.py
-----------
Phase 2 — Feature Engineering

Responsibilities:
  1. Load ingested data from data/processed/customers_ingested.csv
  2. Drop ID column (CLIENTNUM)
  3. Encode categorical features (binary, ordinal, one-hot)
  4. Log-transform right-skewed numeric columns
  5. Engineer 3 new domain-informed features
  6. Scale all numeric features with StandardScaler
  7. Save ML-ready DataFrame to data/processed/customers_features.csv
  8. Save fitted scaler to models/scaler.pkl for reuse in dashboard

Every decision here is deliberate and interview-defensible.
See inline comments for reasoning on each transformation.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from src.logger import get_logger

log = get_logger("features")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR       = Path(__file__).resolve().parent.parent
INGESTED_CSV   = ROOT_DIR / "data" / "processed" / "customers_ingested.csv"
FEATURES_CSV   = ROOT_DIR / "data" / "processed" / "customers_features.csv"
SCALER_PATH    = ROOT_DIR / "models" / "scaler.pkl"

# ── Encoding maps ─────────────────────────────────────────────────────────────

# Ordinal: Education has a natural progression — encoding preserves that order
EDUCATION_ORDER = {
    "Uneducated"    : 0,
    "High School"   : 1,
    "College"       : 2,
    "Graduate"      : 3,
    "Post-Graduate" : 4,
    "Doctorate"     : 5,
    "Unknown"       : -1,   # treat unknown as missing, handled separately
}

# Ordinal: Income has a natural progression
INCOME_ORDER = {
    "Less than $40K" : 0,
    "$40K - $60K"    : 1,
    "$60K - $80K"    : 2,
    "$80K - $120K"   : 3,
    "$120K +"        : 4,
    "Unknown"        : -1,
}

# Columns to log-transform (right-skewed, confirmed in EDA plot 4)
LOG_TRANSFORM_COLS = ["Credit_Limit", "Avg_Open_To_Buy", "Total_Trans_Amt"]

# Columns to one-hot encode (no natural order)
OHE_COLS = ["Marital_Status", "Card_Category"]

# All numeric columns to scale (excludes target 'churn')
# Populated dynamically after all transformations


# ── Transformation functions ──────────────────────────────────────────────────

def drop_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop CLIENTNUM — it's a customer identifier with zero predictive value.
    Including it would cause the model to memorise IDs, not learn patterns.
    """
    df = df.drop(columns=["CLIENTNUM"], errors="ignore")
    log.info("Dropped CLIENTNUM (ID column — no predictive value)")
    return df


def encode_gender(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary encode Gender: M → 1, F → 0.
    Simple and interpretable. No information lost.
    """
    df["Gender"] = (df["Gender"] == "M").astype(int)
    log.info("Encoded Gender: M=1, F=0")
    return df


def encode_ordinal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ordinal encode Education_Level and Income_Category.
    These have a natural ordering — ordinal encoding preserves that
    information which one-hot encoding would discard.
    Unknown values are encoded as -1 and will be handled by the scaler.
    """
    df["Education_Level"] = df["Education_Level"].map(EDUCATION_ORDER)
    df["Income_Category"]  = df["Income_Category"].map(INCOME_ORDER)

    unknown_edu    = (df["Education_Level"] == -1).sum()
    unknown_income = (df["Income_Category"]  == -1).sum()
    log.info(f"Ordinal encoded Education_Level ({unknown_edu} unknowns → -1)")
    log.info(f"Ordinal encoded Income_Category ({unknown_income} unknowns → -1)")
    return df


def encode_onehot_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode Marital_Status and Card_Category.
    These have no natural order — ordinal encoding would impose a false
    hierarchy (e.g. implying Platinum > Gold > Silver mathematically).
    drop_first=True avoids the dummy variable trap (multicollinearity).
    """
    before_cols = df.shape[1]
    df = pd.get_dummies(df, columns=OHE_COLS, drop_first=True, dtype=int)
    after_cols = df.shape[1]
    new_cols = [c for c in df.columns if any(c.startswith(base) for base in OHE_COLS)]
    log.info(f"One-hot encoded {OHE_COLS} → added {after_cols - before_cols} columns: {new_cols}")
    return df


def apply_log_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Log-transform right-skewed numeric columns using log1p (log(1+x)).
    log1p is used instead of log to safely handle any zero values.
    Confirmed right-skewed in EDA plot 4 (credit_limit_dist.png).
    Transformed columns are renamed with '_log' suffix for clarity.
    """
    for col in LOG_TRANSFORM_COLS:
        if col in df.columns:
            new_col = f"{col}_log"
            df[new_col] = np.log1p(df[col])
            df = df.drop(columns=[col])
            log.info(f"Log-transformed {col} → {new_col}")
    return df


def engineer_new_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create 3 new domain-informed features.

    1. transaction_velocity: Average spend per transaction.
       Insight: Churned customers don't just transact less — they
       also spend less per transaction. This compound signal is stronger
       than either component alone.

    2. inactivity_risk: Months inactive × contact count.
       Insight: A customer who is inactive AND calling the bank frequently
       is a very high churn risk (frustrated + disengaged).

    3. credit_usage_gap: Credit limit minus open-to-buy (available credit).
       Insight: Directly captures how much of the credit limit the
       customer is actually using — a cleaner signal than utilisation ratio alone.
    """
    # Feature 1: transaction velocity (avg spend per transaction)
    df["transaction_velocity"] = (
        df["Total_Trans_Amt_log"] / (df["Total_Trans_Ct"] + 1)
        if "Total_Trans_Amt_log" in df.columns
        else df["Total_Trans_Amt"] / (df["Total_Trans_Ct"] + 1)
    )
    log.info("Engineered feature: transaction_velocity = Total_Trans_Amt / (Total_Trans_Ct + 1)")

    # Feature 2: inactivity risk score
    df["inactivity_risk"] = df["Months_Inactive_12_mon"] * df["Contacts_Count_12_mon"]
    log.info("Engineered feature: inactivity_risk = Months_Inactive × Contacts_Count")

    # Feature 3: credit usage gap
    # Use log versions if available, otherwise raw
    credit_col  = "Credit_Limit_log"     if "Credit_Limit_log"     in df.columns else "Credit_Limit"
    open_to_buy = "Avg_Open_To_Buy_log"  if "Avg_Open_To_Buy_log"  in df.columns else "Avg_Open_To_Buy"
    df["credit_usage_gap"] = df[credit_col] - df[open_to_buy]
    log.info("Engineered feature: credit_usage_gap = Credit_Limit_log - Avg_Open_To_Buy_log")

    return df


def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    StandardScaler on all numeric features except the target column 'churn'.
    StandardScaler is chosen over MinMaxScaler because:
      - It handles outliers better (not sensitive to min/max)
      - Required for logistic regression (gradient-based optimiser)
      - XGBoost doesn't need scaling but it doesn't hurt
    The fitted scaler is saved to models/scaler.pkl so the dashboard
    can scale new input data consistently at inference time.
    """
    feature_cols = [c for c in df.select_dtypes(include="number").columns if c != "churn"]

    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)

    log.info(f"StandardScaler fitted on {len(feature_cols)} features")
    log.info(f"Scaler saved → {SCALER_PATH.relative_to(ROOT_DIR)}")
    return df


def audit_features(df: pd.DataFrame) -> None:
    """Log a summary of the final feature matrix."""
    log.info("── Feature Matrix Audit ────────────────────────────────")
    log.info(f"  Shape         : {df.shape}")
    log.info(f"  Numeric cols  : {df.select_dtypes('number').shape[1]}")
    log.info(f"  String cols   : {df.select_dtypes('object').shape[1]}  ← must be 0")
    log.info(f"  Null values   : {df.isnull().sum().sum()}  ← must be 0")
    log.info(f"  Churn balance : {df['churn'].mean()*100:.2f}% positive class")
    log.info(f"  Columns       : {list(df.columns)}")
    log.info("────────────────────────────────────────────────────────")


# ── Main entry point ──────────────────────────────────────────────────────────

def run_feature_engineering() -> pd.DataFrame:
    """
    Execute the full feature engineering pipeline.
    Returns the ML-ready DataFrame.
    """
    log.info("═══════════════════════════════════════════════════════")
    log.info("  PHASE 2b — FEATURE ENGINEERING                     ")
    log.info("═══════════════════════════════════════════════════════")

    log.info(f"Loading ingested data from: {INGESTED_CSV}")
    df = pd.read_csv(INGESTED_CSV)
    log.info(f"Loaded {len(df):,} rows × {df.shape[1]} columns")

    # Apply transformations in correct order
    df = drop_id_column(df)
    df = encode_gender(df)
    df = encode_ordinal_features(df)
    df = apply_log_transforms(df)
    df = engineer_new_features(df)
    df = encode_onehot_features(df)
    df = scale_features(df)

    audit_features(df)

    # Save final feature matrix
    FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FEATURES_CSV, index=False)
    log.info(f"Feature matrix saved → {FEATURES_CSV.relative_to(ROOT_DIR)}")
    log.info("Phase 2b complete — feature engineering successful ✓")

    return df


if __name__ == "__main__":
    run_feature_engineering()