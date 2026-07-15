"""
Model training layer, tracked with MLflow.

Trains a simple, interpretable baseline (logistic regression) - published
analyses of this exact dataset consistently find logistic regression
competitive with or better than more complex models (churn here is largely
linearly separable by contract type and tenure), so it's a reasonable
default rather than reaching for complexity the data doesn't need.
"""
from __future__ import annotations

import logging

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"
EXPERIMENT_NAME = "telco-churn"
MODEL_NAME = "telco-churn-classifier"


def train_model(features_df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> dict:
    X = features_df.drop(columns=[TARGET_COLUMN, ID_COLUMN])
    y = features_df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }

        mlflow.log_params({
            "model_type": "LogisticRegression",
            "class_weight": "balanced",
            "test_size": test_size,
            "n_features": X.shape[1],
        })
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            model, name="model",
            registered_model_name=MODEL_NAME,
            input_example=X_train.head(3),
        )

        # Log the scaler too - needed at inference time, and versioning it
        # alongside the model avoids a mismatched-scaler bug in production.
        mlflow.sklearn.log_model(scaler, name="scaler")

        logger.info("Training complete. Metrics: %s", metrics)
        return metrics


if __name__ == "__main__":
    from pathlib import Path

    from features import engineer_features

    logging.basicConfig(level=logging.INFO)
    path = Path(__file__).resolve().parent.parent / "data" / "telco_churn.csv"
    raw = pd.read_csv(path)
    features = engineer_features(raw)
    train_model(features)
