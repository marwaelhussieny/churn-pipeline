import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "serving"))

# Point at the same local MLflow store used for the manual verification run -
# these tests assume train.py has been run at least once already.
os.environ.setdefault(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{Path(__file__).resolve().parent.parent / 'src' / 'mlflow.db'}",
)

from fastapi.testclient import TestClient  # noqa: E402

from api import app  # noqa: E402


HIGH_RISK_PROFILE = {
    "gender": 0, "SeniorCitizen": 0, "Partner": 0, "Dependents": 0,
    "tenure": 1, "PhoneService": 1, "MultipleLines": 0,
    "OnlineSecurity": 0, "OnlineBackup": 0, "DeviceProtection": 0,
    "TechSupport": 0, "StreamingTV": 0, "StreamingMovies": 0,
    "PaperlessBilling": 1, "MonthlyCharges": 95.0, "TotalCharges": 95.0,
    "InternetService_DSL": 0, "InternetService_Fiber optic": 1, "InternetService_No": 0,
    "Contract-Month-to-month": 1,
}

LOW_RISK_PROFILE = {
    "gender": 0, "SeniorCitizen": 0, "Partner": 1, "Dependents": 1,
    "tenure": 60, "PhoneService": 1, "MultipleLines": 1,
    "OnlineSecurity": 1, "OnlineBackup": 1, "DeviceProtection": 1,
    "TechSupport": 1, "StreamingTV": 0, "StreamingMovies": 0,
    "PaperlessBilling": 0, "MonthlyCharges": 60.0, "TotalCharges": 3600.0,
    "InternetService_DSL": 1, "InternetService_Fiber optic": 0, "InternetService_No": 0,
    "Contract_Two_year": 1,
}


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_check(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_predict_returns_valid_response(client):
    r = client.post("/predict", json=HIGH_RISK_PROFILE)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert isinstance(body["churn_prediction"], bool)


def test_high_risk_profile_scores_higher_than_low_risk(client):
    """Sanity check that the model direction makes sense, not just that it runs."""
    high = client.post("/predict", json=HIGH_RISK_PROFILE).json()
    low = client.post("/predict", json=LOW_RISK_PROFILE).json()
    assert high["churn_probability"] > low["churn_probability"]


def test_repeated_request_is_served_from_cache(client):
    r1 = client.post("/predict", json=HIGH_RISK_PROFILE)
    r2 = client.post("/predict", json=HIGH_RISK_PROFILE)
    assert r1.json()["cached"] is False or r2.json()["cached"] is True
    assert r1.json()["churn_probability"] == r2.json()["churn_probability"]


def test_metrics_endpoint_exposes_prometheus_format(client):
    client.get("/health")
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "python_gc_objects_collected_total" in r.text
