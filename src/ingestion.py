"""
ingestion.py
------------
Phase 1 — Data Ingestion

Responsibilities:
  1. Read raw CSV from data/raw/BankChurners.csv
  2. Drop 2 junk Naive Bayes columns left by the dataset author
  3. Rename target column to clean binary 'churn' (1 = churned, 0 = retained)
  4. Log data quality summary (shape, nulls, class balance)
  5. Write cleaned data into SQLite table 'customers' via SQLAlchemy
  6. Save a processed CSV copy to data/processed/customers_ingested.csv

Why SQLite?
  Demonstrates SQL ingestion on the resume without requiring a running
  database server. SQLAlchemy abstracts the connection so swapping to
  PostgreSQL later requires only a connection string change.

Why drop Naive Bayes columns?
  The dataset author leaked two model-output columns into the raw file.
  Including them would cause data leakage (they directly encode the target).
  Dropping them is the correct data science decision and a good interview talking point.
"""

import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine, text

from src.logger import get_logger

log = get_logger("ingestion")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR       = Path(__file__).resolve().parent.parent
RAW_CSV        = ROOT_DIR / "data" / "raw" / "BankChurners.csv"
PROCESSED_CSV  = ROOT_DIR / "data" / "processed" / "customers_ingested.csv"
DB_PATH        = ROOT_DIR / "database" / "churn.db"
DB_URL         = f"sqlite:///{DB_PATH}"
TABLE_NAME     = "customers"

# Columns to drop — Naive Bayes leakage columns from original dataset
JUNK_COLUMNS = [
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon"
    "_Dependent_count_Education_Level_Months_Inactive_12_mon_1",
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon"
    "_Dependent_count_Education_Level_Months_Inactive_12_mon_2",
]

# Target column mapping
TARGET_RAW    = "Attrition_Flag"
TARGET_CLEAN  = "churn"
CHURN_VALUE   = "Attrited Customer"   # string value that means churned


# ── Helper functions ──────────────────────────────────────────────────────────

def load_raw_csv(path: Path) -> pd.DataFrame:
    """Read the raw CSV and return a DataFrame."""
    log.info(f"Loading raw CSV from: {path}")
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data file not found at '{path}'.\n"
            "Please download BankChurners.csv from Kaggle and place it in data/raw/"
        )
    df = pd.read_csv(path)
    log.info(f"Raw data loaded — shape: {df.shape}")
    return df


def drop_junk_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop the two Naive Bayes leakage columns injected by the dataset author.
    Uses partial matching to handle any minor column name variations.
    """
    cols_to_drop = [c for c in df.columns if "Naive_Bayes" in c]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        log.info(f"Dropped {len(cols_to_drop)} Naive Bayes leakage column(s): {cols_to_drop}")
    else:
        log.warning("No Naive Bayes columns found — skipping drop step.")
    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert string target column to binary integer:
      'Attrited Customer'  → 1  (churned)
      'Existing Customer'  → 0  (retained)
    Renames column from 'Attrition_Flag' to 'churn'.
    """
    if TARGET_RAW not in df.columns:
        raise KeyError(f"Expected target column '{TARGET_RAW}' not found in data.")

    df[TARGET_CLEAN] = (df[TARGET_RAW] == CHURN_VALUE).astype(int)
    df = df.drop(columns=[TARGET_RAW])

    churn_rate = df[TARGET_CLEAN].mean() * 100
    class_counts = df[TARGET_CLEAN].value_counts().to_dict()
    log.info(f"Target encoded — churn rate: {churn_rate:.2f}%")
    log.info(f"Class distribution — {class_counts}  (0 = retained, 1 = churned)")
    return df


def audit_data_quality(df: pd.DataFrame) -> None:
    """Log a data quality summary — shape, dtypes, nulls, duplicates."""
    log.info("── Data Quality Audit ──────────────────────────────────")
    log.info(f"  Rows        : {df.shape[0]:,}")
    log.info(f"  Columns     : {df.shape[1]}")

    null_counts = df.isnull().sum()
    null_cols   = null_counts[null_counts > 0]
    if null_cols.empty:
        log.info("  Nulls       : None — dataset is complete ✓")
    else:
        log.warning(f"  Nulls found :\n{null_cols}")

    duplicates = df.duplicated().sum()
    log.info(f"  Duplicates  : {duplicates}")
    log.info(f"  Numeric cols: {df.select_dtypes('number').shape[1]}")
    log.info(f"  Object cols : {df.select_dtypes('object').shape[1]}")
    log.info("────────────────────────────────────────────────────────")


def save_to_sqlite(df: pd.DataFrame, db_url: str, table: str) -> None:
    """
    Write DataFrame to SQLite using SQLAlchemy.
    Uses 'replace' to ensure idempotency — re-running ingestion
    always produces a clean, consistent table state.
    """
    log.info(f"Connecting to SQLite database: {db_url}")
    engine = create_engine(db_url, echo=False)

    df.to_sql(table, con=engine, if_exists="replace", index=False)
    log.info(f"Written {len(df):,} rows to table '{table}'")

    # Verify write with a quick SQL query
    with engine.connect() as conn:
        result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
        count  = result.scalar()
    log.info(f"SQL verification — SELECT COUNT(*) FROM {table} = {count:,} ✓")

    engine.dispose()


def save_processed_csv(df: pd.DataFrame, path: Path) -> None:
    """Save a CSV copy of the ingested data for downstream modules."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info(f"Processed CSV saved to: {path}")


# ── Main entry point ──────────────────────────────────────────────────────────

def run_ingestion() -> pd.DataFrame:
    """
    Execute the full ingestion pipeline.
    Returns the cleaned DataFrame for use in run_pipeline.py.
    """
    log.info("═══════════════════════════════════════════════════════")
    log.info("  PHASE 1 — DATA INGESTION                            ")
    log.info("═══════════════════════════════════════════════════════")

    # Ensure database directory exists
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Step-by-step ingestion
    df = load_raw_csv(RAW_CSV)
    df = drop_junk_columns(df)
    df = encode_target(df)
    audit_data_quality(df)
    save_to_sqlite(df, DB_URL, TABLE_NAME)
    save_processed_csv(df, PROCESSED_CSV)

    log.info("Phase 1 complete — ingestion successful ✓")
    return df


if __name__ == "__main__":
    run_ingestion()
