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


NETWORK_PARQUET = PROCESSED / "network_features.parquet"
LABEL_LAG = 30
NETWORK_COLS = ["net_degree", "net_device_degree", "net_card_degree", "nbr_fraud_uids",
                "nbr_known_txns", "nbr_fraud_txns", "nbr_fraud_rate", "nbr_device_fraud_uids",
                "clu_size", "clu_fraud_uids", "clu_fraud_rate"]


def build_network(force: bool = False) -> Path:
    """As-of network features. Needs graph.run() to have written clusters.parquet."""
    from graph import CLUSTERS_PARQUET, HUB_MAX
    from model import TRAIN_LAST_DAY
    if NETWORK_PARQUET.exists() and not force:
        return NETWORK_PARQUET
    con = duckdb.connect()
    con.execute(f"CREATE VIEW base AS SELECT * FROM read_parquet('{BASE_PARQUET}')")
    con.execute(f"CREATE VIEW identity AS SELECT * FROM read_parquet('{IDENTITY_PARQUET}')")
    con.execute(f"CREATE VIEW clusters AS SELECT * FROM read_parquet('{CLUSTERS_PARQUET}')")
    sql = ((SQL / "05_network_features.sql").read_text()
           .replace("{hub_max}", str(HUB_MAX))
           .replace("{train_last_day}", str(TRAIN_LAST_DAY))
           .replace("{label_lag}", str(LABEL_LAG)))
    con.execute(sql)
    con.execute(f"COPY network_features TO '{NETWORK_PARQUET}' (FORMAT parquet)")
    con.close()
    return NETWORK_PARQUET


if __name__ == "__main__":
    import time
    t = time.time()
    build_velocity(force=True)
    print(f"velocity built in {time.time() - t:.1f}s")
    con = duckdb.connect()
    print(con.sql(f"SELECT * FROM read_parquet('{VELOCITY_PARQUET}')").df().describe().T.round(2).to_string())
