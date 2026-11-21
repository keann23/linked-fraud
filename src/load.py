"""Phase 1: load the Kaggle CSVs into DuckDB, write one parquet file, report basics."""
from __future__ import annotations

import time
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
SQL = ROOT / "sql"
BASE_PARQUET = PROCESSED / "base.parquet"


def build_base(force: bool = False) -> Path:
    """CSV -> DuckDB join -> parquet. Slow once (CSV parse), fast after."""
    if BASE_PARQUET.exists() and not force:
        return BASE_PARQUET
    for f in ("train_transaction.csv", "train_identity.csv"):
        if not (RAW / f).exists():
            raise FileNotFoundError(f"Missing {RAW / f}. See README 'How to run'.")
    PROCESSED.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute((SQL / "01_base.sql").read_text().replace("{raw}", str(RAW)))
    con.execute(f"COPY base TO '{BASE_PARQUET}' (FORMAT parquet, COMPRESSION zstd)")
    con.close()
    return BASE_PARQUET


def connect() -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with a `base` view over the parquet file."""
    con = duckdb.connect()
    con.execute(f"CREATE VIEW base AS SELECT * FROM read_parquet('{BASE_PARQUET}')")
    return con


def summary(con: duckdb.DuckDBPyConnection) -> dict[str, pd.DataFrame]:
    overall = con.sql("""
        SELECT COUNT(*)                          AS transactions,
               SUM(isFraud)                      AS fraud_txns,
               ROUND(100 * AVG(isFraud), 2)      AS fraud_rate_pct,
               ROUND(100 * AVG(has_identity::INT), 1) AS pct_with_identity,
               MIN(day) AS first_day, MAX(day) AS last_day,
               MAX(day) - MIN(day) + 1           AS span_days,
               ROUND(SUM(TransactionAmt))        AS total_amt,
               ROUND(SUM(TransactionAmt * isFraud)) AS fraud_amt
        FROM base
    """).df()
    by_product = con.sql("""
        SELECT ProductCD,
               COUNT(*) AS transactions,
               ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS share_pct,
               ROUND(100 * AVG(isFraud), 2) AS fraud_rate_pct,
               ROUND(100 * AVG(has_identity::INT), 1) AS pct_with_identity
        FROM base GROUP BY ProductCD ORDER BY transactions DESC
    """).df()
    return {"overall": overall, "by_product": by_product}


if __name__ == "__main__":
    t0 = time.time()
    build_base()
    print(f"parquet ready in {time.time() - t0:.1f}s")
    t0 = time.time()
    con = connect()
    out = summary(con)
    print(f"summary from parquet in {time.time() - t0:.1f}s")
    for name, df in out.items():
        print(f"\n== {name} ==\n{df.to_string(index=False)}")
