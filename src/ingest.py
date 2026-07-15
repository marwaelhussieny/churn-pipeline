"""
Ingestion layer: loads the raw Telco churn CSV into Redshift, as-is.

No cleaning here - bronze/raw philosophy consistent with the rest of this
portfolio. Feature engineering (features.py) is where the real data quality
handling happens.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "telco_churn.csv"
RAW_TABLE = "raw_telco_churn"


def get_connection():
    conn_string = os.environ.get("REDSHIFT_CONN_STRING")
    if not conn_string:
        raise RuntimeError("REDSHIFT_CONN_STRING environment variable is not set.")
    return psycopg2.connect(conn_string)


def load_raw(path: Path = DATA_PATH) -> int:
    df = pd.read_csv(path)
    logger.info("Loaded %d raw rows from %s", len(df), path)

    columns = list(df.columns)
    col_defs = ", ".join(f'"{c}" VARCHAR(256)' for c in columns)  # load as text - typing happens in features.py

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(f'DROP TABLE IF EXISTS {RAW_TABLE}')
            cur.execute(f'CREATE TABLE {RAW_TABLE} ({col_defs})')

            values = [tuple(row) for row in df.itertuples(index=False, name=None)]
            insert_sql = f'INSERT INTO {RAW_TABLE} ({", ".join(f\'"{c}"\' for c in columns)}) VALUES %s'
            execute_values(cur, insert_sql, values, page_size=1000)
        conn.commit()
    finally:
        conn.close()

    logger.info("Loaded %d rows into Redshift table %s", len(df), RAW_TABLE)
    return len(df)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    load_raw()
