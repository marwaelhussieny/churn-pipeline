"""
FastAPI serving layer for the churn model.

- Loads the latest registered model + scaler from MLflow at startup.
- Caches predictions in Redis, keyed by a hash of the input features - a
  repeat request for the same customer profile returns instantly from cache
  instead of re-running inference, which matters once this is handling real
  traffic volume.
- Exposes Prometheus metrics at /metrics via prometheus-fastapi-instrumentator,
  so request latency, error rate, and throughput are observable out of the box.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os

import mlflow
import redis
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_NAME = os.environ.get("MODEL_NAME", "telco-churn-classifier")
MODEL_STAGE = os.environ.get("MODEL_STAGE", "latest")
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", 3600))

from contextlib import asynccontextmanager


def load_model():
    global _model, _scaler
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    _model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")

    # The scaler is versioned as a separate MLflow run artifact from training -
    # load it from that same run so scaling at inference time matches training exactly.
    client = mlflow.MlflowClient()
    model_version = client.get_model_version_by_alias(MODEL_NAME, MODEL_STAGE) \
        if MODEL_STAGE not in ("latest",) else None
    if model_version is None:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        run_id = sorted(versions, key=lambda v: int(v.version))[-1].run_id
    else:
        run_id = model_version.run_id
    _scaler = mlflow.sklearn.load_model(f"runs:/{run_id}/scaler")

    logger.info("Loaded model %s@%s and its matching scaler (run %s)", MODEL_NAME, MODEL_STAGE, run_id)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="Telco Churn Prediction API", lifespan=lifespan)
Instrumentator().instrument(app).expose(app)  # adds /metrics for Prometheus scraping

_model = None
_scaler = None
_redis_client = None


class CustomerFeatures(BaseModel):
    gender: int = Field(..., ge=0, le=1)
    SeniorCitizen: int = Field(..., ge=0, le=1)
    Partner: int = Field(..., ge=0, le=1)
    Dependents: int = Field(..., ge=0, le=1)
    tenure: int = Field(..., ge=0)
    PhoneService: int = Field(..., ge=0, le=1)
    MultipleLines: int = Field(..., ge=0, le=1)
    OnlineSecurity: int = Field(..., ge=0, le=1)
    OnlineBackup: int = Field(..., ge=0, le=1)
    DeviceProtection: int = Field(..., ge=0, le=1)
    TechSupport: int = Field(..., ge=0, le=1)
    StreamingTV: int = Field(..., ge=0, le=1)
    StreamingMovies: int = Field(..., ge=0, le=1)
    PaperlessBilling: int = Field(..., ge=0, le=1)
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)
    InternetService_DSL: int = Field(0, ge=0, le=1)
    InternetService_Fiber_optic: int = Field(0, ge=0, le=1, alias="InternetService_Fiber optic")
    InternetService_No: int = Field(0, ge=0, le=1)
    Contract_Month_to_month: int = Field(0, ge=0, le=1, alias="Contract_Month-to-month")
    Contract_One_year: int = Field(0, ge=0, le=1, alias="Contract_One year")
    Contract_Two_year: int = Field(0, ge=0, le=1, alias="Contract_Two year")
    PaymentMethod_Bank_transfer: int = Field(0, ge=0, le=1, alias="PaymentMethod_Bank transfer (automatic)")
    PaymentMethod_Credit_card: int = Field(0, ge=0, le=1, alias="PaymentMethod_Credit card (automatic)")
    PaymentMethod_Electronic_check: int = Field(0, ge=0, le=1, alias="PaymentMethod_Electronic check")
    PaymentMethod_Mailed_check: int = Field(0, ge=0, le=1, alias="PaymentMethod_Mailed check")

    model_config = {"populate_by_name": True}


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: bool
    cached: bool


def _cache_key(features: dict) -> str:
    payload = json.dumps(features, sort_keys=True)
    return f"churn_pred:{hashlib.sha256(payload.encode()).hexdigest()}"


def get_redis() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    return _redis_client


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures):
    feature_dict = features.model_dump(by_alias=True)
    key = _cache_key(feature_dict)

    r = get_redis()
    cached = r.get(key)
    if cached:
        result = json.loads(cached)
        return PredictionResponse(**result, cached=True)

    import pandas as pd

    X = pd.DataFrame([feature_dict])
    X_scaled = _scaler.transform(X)
    proba = float(_model.predict_proba(X_scaled)[0][1]) if hasattr(_model, "predict_proba") else float(_model.predict(X_scaled)[0])
    prediction = proba >= 0.5

    result = {"churn_probability": round(proba, 4), "churn_prediction": prediction}
    r.set(key, json.dumps(result), ex=CACHE_TTL_SECONDS)

    return PredictionResponse(**result, cached=False)
