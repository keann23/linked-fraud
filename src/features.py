"""Phase 3: SQL velocity features (DuckDB window functions)."""
from __future__ import annotations

from pathlib import Path

import duckdb

from identity import IDENTITY_PARQUET
from load import BASE_PARQUET, PROCESSED, SQL

VELOCITY_PARQUET = PROCESSED / "velocity.parquet"


def build_velocity(force: bool = False) -> Path:
    if VELOCITY_PARQUET.exists() and not force:
        return VELOCITY_PARQUET
    con = duckdb.connect()
    con.execute(f"CREATE VIEW base AS SELECT * FROM read_parquet('{BASE_PARQUET}')")
    con.execute(f"CREATE VIEW identity AS SELECT * FROM read_parquet('{IDENTITY_PARQUET}')")
    con.execute((SQL / "03_velocity.sql").read_text())
    con.execute(f"COPY velocity TO '{VELOCITY_PARQUET}' (FORMAT parquet)")
    con.close()
    return VELOCITY_PARQUET


if __name__ == "__main__":
    import time
    t = time.time()
    build_velocity(force=True)
    print(f"velocity built in {time.time() - t:.1f}s")
    con = duckdb.connect()
    print(con.sql(f"SELECT * FROM read_parquet('{VELOCITY_PARQUET}')").df().describe().T.round(2).to_string())
