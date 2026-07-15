"""
Feature engineering layer.

Handles the real, well-documented messiness of this specific dataset:
- TotalCharges arrives as a string, and 11 rows have it as a blank/whitespace
  string rather than a number - these are all brand-new customers (tenure=0)
  who haven't been billed yet, so TotalCharges is genuinely undefined, not
  actually missing/corrupted data. Filled with 0 rather than dropped, since
  a new customer with $0 total charges is a real, meaningful data point.
- Several categorical columns use "No internet service"/"No phone service"
  as a third category that's really just a different flavor of "No" -
  collapsed to binary Yes/No for a cleaner feature space.
- Target and binary Yes/No columns are converted to 0/1 explicitly rather
  than relying on implicit string encoding downstream.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BINARY_YES_NO_COLUMNS = [
    "Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn",
]

COLLAPSE_TO_NO = [
    "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

CATEGORICAL_COLUMNS = ["InternetService", "Contract", "PaymentMethod", "gender"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n_before = len(df)

    n_blank_charges = (df["TotalCharges"].astype(str).str.strip() == "").sum()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    logger.info("Filled %d blank TotalCharges values (new customers, tenure=0) with 0", n_blank_charges)

    for col in COLLAPSE_TO_NO:
        df[col] = df[col].replace({"No internet service": "No", "No phone service": "No"})

    for col in BINARY_YES_NO_COLUMNS + COLLAPSE_TO_NO:
        df[col] = df[col].map({"Yes": 1, "No": 0}).astype(int)

    df["gender"] = df["gender"].map({"Male": 1, "Female": 0}).astype(int)

    df = pd.get_dummies(
        df, columns=["InternetService", "Contract", "PaymentMethod"], prefix_sep="_"
    )

    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    n_after = len(df)
    logger.info("Feature engineering: %d -> %d rows (%d dropped)", n_before, n_after, n_before - n_after)
    return df


if __name__ == "__main__":
    from pathlib import Path

    logging.basicConfig(level=logging.INFO)
    path = Path(__file__).resolve().parent.parent / "data" / "telco_churn.csv"
    raw = pd.read_csv(path)
    features = engineer_features(raw)
    print(features.shape)
    print(features.dtypes)
