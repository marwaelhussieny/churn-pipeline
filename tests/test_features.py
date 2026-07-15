import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "serving"))

from features import engineer_features  # noqa: E402


@pytest.fixture
def sample_raw_df():
    return pd.DataFrame({
        "customerID": ["A", "B", "C"],
        "gender": ["Female", "Male", "Female"],
        "SeniorCitizen": [0, 1, 0],
        "Partner": ["Yes", "No", "Yes"],
        "Dependents": ["No", "No", "Yes"],
        "tenure": [1, 34, 0],
        "PhoneService": ["No", "Yes", "Yes"],
        "MultipleLines": ["No phone service", "No", "Yes"],
        "InternetService": ["DSL", "Fiber optic", "No"],
        "OnlineSecurity": ["No", "Yes", "No internet service"],
        "OnlineBackup": ["Yes", "No", "No internet service"],
        "DeviceProtection": ["No", "Yes", "No internet service"],
        "TechSupport": ["No", "No", "No internet service"],
        "StreamingTV": ["No", "Yes", "No internet service"],
        "StreamingMovies": ["No", "Yes", "No internet service"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "PaperlessBilling": ["Yes", "No", "Yes"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Credit card (automatic)"],
        "MonthlyCharges": [29.85, 56.95, 20.0],
        "TotalCharges": ["29.85", "1889.5", " "],  # blank string - the real bug case
        "Churn": ["No", "No", "Yes"],
    })


def test_engineer_features_fills_blank_total_charges(sample_raw_df):
    result = engineer_features(sample_raw_df)
    assert result["TotalCharges"].iloc[2] == 0.0
    assert result["TotalCharges"].dtype == float


def test_engineer_features_collapses_no_internet_service(sample_raw_df):
    result = engineer_features(sample_raw_df)
    # row C had "No internet service" for OnlineSecurity - should collapse to 0
    assert result["OnlineSecurity"].iloc[2] == 0


def test_engineer_features_binary_columns_are_01(sample_raw_df):
    result = engineer_features(sample_raw_df)
    for col in ["Partner", "Dependents", "PhoneService", "Churn"]:
        assert set(result[col].unique()).issubset({0, 1})


def test_engineer_features_one_hot_encodes_categoricals(sample_raw_df):
    result = engineer_features(sample_raw_df)
    assert "Contract_Month-to-month" in result.columns
    assert "InternetService_Fiber optic" in result.columns
    assert "PaymentMethod_Electronic check" in result.columns


def test_engineer_features_preserves_row_count(sample_raw_df):
    result = engineer_features(sample_raw_df)
    assert len(result) == len(sample_raw_df)


def test_full_pipeline_on_real_sample_data():
    """Runs feature engineering against the real committed data sample, not synthetic data."""
    path = Path(__file__).resolve().parent.parent / "data" / "telco_churn_sample.csv"
    raw = pd.read_csv(path)
    result = engineer_features(raw)
    assert len(result) == len(raw)
    assert result["TotalCharges"].isna().sum() == 0
    assert set(result["Churn"].unique()).issubset({0, 1})
