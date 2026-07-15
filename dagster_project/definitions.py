"""
Dagster assets for the churn pipeline: raw_churn_data -> churn_model.

The serving API (serving/api.py) is deliberately NOT a Dagster asset - it's
a long-running service, not a batch job. Dagster owns the batch side
(ingestion + training); the FastAPI service picks up whatever the latest
registered MLflow model is independently.
"""
import sys
from pathlib import Path

import dagster as dg

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


@dg.asset(group_name="raw")
def raw_churn_data(context: dg.AssetExecutionContext) -> None:
    """Loads the raw Telco churn CSV into Redshift."""
    from ingest import load_raw

    n = load_raw()
    context.add_output_metadata({"rows_loaded": n})


@dg.asset(group_name="model", deps=[raw_churn_data])
def churn_model(context: dg.AssetExecutionContext):
    """Engineers features and trains a new model version, tracked in MLflow."""
    import pandas as pd

    from features import engineer_features
    from train import train_model
    from pathlib import Path as P

    # In production this would read back from Redshift; reading the local
    # CSV directly here keeps this asset runnable without a live Redshift
    # connection for local dev/testing.
    path = P(__file__).resolve().parent.parent / "data" / "telco_churn.csv"
    raw = pd.read_csv(path)
    features = engineer_features(raw)
    metrics = train_model(features)

    context.add_output_metadata({k: float(v) for k, v in metrics.items()})
    return metrics


weekly_schedule = dg.ScheduleDefinition(
    name="weekly_churn_retrain",
    cron_schedule="0 3 * * 1",  # Monday 3am - retrain weekly as new data accumulates
    job=dg.define_asset_job("churn_pipeline_job", selection=[raw_churn_data, churn_model]),
)

defs = dg.Definitions(
    assets=[raw_churn_data, churn_model],
    schedules=[weekly_schedule],
)
