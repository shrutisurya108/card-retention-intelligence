"""
test_ingestion.py
-----------------
Phase 1 Tests — Data Ingestion
"""

import pytest
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine, inspect, text

ROOT_DIR      = Path(__file__).resolve().parent.parent
DB_PATH       = ROOT_DIR / "database" / "churn.db"
DB_URL        = f"sqlite:///{DB_PATH}"
PROCESSED_CSV = ROOT_DIR / "data" / "processed" / "customers_ingested.csv"
TABLE_NAME    = "customers"
EXPECTED_ROWS = 10127

EXPECTED_COLS = [
    "CLIENTNUM", "Customer_Age", "Gender", "Dependent_count",
    "Education_Level", "Marital_Status", "Income_Category", "Card_Category",
    "Months_on_book", "Total_Relationship_Count", "Months_Inactive_12_mon",
    "Contacts_Count_12_mon", "Credit_Limit", "Total_Revolving_Bal",
    "Avg_Open_To_Buy", "Total_Amt_Chng_Q4_Q1", "Total_Trans_Amt",
    "Total_Trans_Ct", "Total_Ct_Chng_Q4_Q1", "Avg_Utilization_Ratio", "churn",
]


@pytest.fixture(scope="module")
def engine():
    assert DB_PATH.exists(), f"Database not found at {DB_PATH}. Run python run_pipeline.py first."
    eng = create_engine(DB_URL, echo=False)
    yield eng
    eng.dispose()


@pytest.fixture(scope="module")
def customers_df(engine):
    return pd.read_sql(f"SELECT * FROM {TABLE_NAME}", con=engine)


class TestDatabaseCreation:
    def test_db_file_exists(self):
        assert DB_PATH.exists(), f"DB file missing: {DB_PATH}"

    def test_customers_table_exists(self, engine):
        inspector = inspect(engine)
        assert TABLE_NAME in inspector.get_table_names()


class TestRowCount:
    def test_row_count(self, engine):
        with engine.connect() as conn:
            count = conn.execute(text(f"SELECT COUNT(*) FROM {TABLE_NAME}")).scalar()
        assert count == EXPECTED_ROWS, f"Expected {EXPECTED_ROWS:,} rows, got {count:,}."


class TestColumnIntegrity:
    def test_naive_bayes_columns_dropped(self, customers_df):
        naive_bayes_cols = [c for c in customers_df.columns if "Naive_Bayes" in c]
        assert len(naive_bayes_cols) == 0, f"Leakage columns still present: {naive_bayes_cols}"

    def test_original_target_column_removed(self, customers_df):
        assert "Attrition_Flag" not in customers_df.columns

    def test_expected_columns_present(self, customers_df):
        missing = [c for c in EXPECTED_COLS if c not in customers_df.columns]
        assert len(missing) == 0, f"Missing columns: {missing}"


class TestTargetColumn:
    def test_churn_column_exists(self, customers_df):
        assert "churn" in customers_df.columns

    def test_churn_is_binary(self, customers_df):
        assert set(customers_df["churn"].unique()) == {0, 1}

    def test_churn_rate_realistic(self, customers_df):
        churn_rate = customers_df["churn"].mean()
        assert 0.10 <= churn_rate <= 0.25, f"Unexpected churn rate: {churn_rate:.2%}"


class TestDataQuality:
    def test_no_fully_null_columns(self, customers_df):
        fully_null = [c for c in customers_df.columns if customers_df[c].isnull().all()]
        assert len(fully_null) == 0

    def test_no_duplicate_rows(self, customers_df):
        assert customers_df.duplicated().sum() == 0

    def test_clientnum_unique(self, customers_df):
        assert customers_df["CLIENTNUM"].nunique() == len(customers_df)


class TestProcessedCSV:
    def test_processed_csv_exists(self):
        assert PROCESSED_CSV.exists(), f"Processed CSV missing: {PROCESSED_CSV}"

    def test_processed_csv_row_count(self):
        df = pd.read_csv(PROCESSED_CSV)
        assert len(df) == EXPECTED_ROWS, f"CSV has {len(df):,} rows, expected {EXPECTED_ROWS:,}."
