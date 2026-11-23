"""Phase 5: same 2% review budget, four ways of choosing what to review.

1. Baseline model        transaction + velocity features
2. Ring flag alone       test txns whose identity sits in a ring (train labels)
3. Model + ring alert    ring txns go to review first, model fills the rest
4. Model + network       baseline features + as-of network features, retrained

Each model is trained with 3 seeds; we report the mean and the spread.
"""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

import graph
import model
from features import NETWORK_COLS, NETWORK_PARQUET

SEEDS = [1, 2, 3]
RESULTS_PARQUET = model.PROCESSED / "phase5_scores.parquet"


def load_frame() -> pd.DataFrame:
    df = model.load_frame()
    net = duckdb.sql(f"SELECT * FROM read_parquet('{NETWORK_PARQUET}')").df()
    return df.merge(net, on="TransactionID", how="left")


def ring_flag(test: pd.DataFrame) -> np.ndarray:
    c = pd.read_parquet(model.PROCESSED / "cluster_table.parquet")
    u = pd.read_parquet(graph.CLUSTERS_PARQUET)[["uid", "cluster_id"]]
    rings = set(c.index[graph.is_ring(c)])
    cid = test[["uid"]].astype({"uid": str}).merge(u, on="uid", how="left").cluster_id
    return cid.isin(rings).to_numpy()


def ring_first(ring: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Ring transactions jump to the top of the review queue."""
    return np.asarray(scores) + ring.astype(float) * 10.0


def train_scores(df: pd.DataFrame) -> pd.DataFrame:
    train, test = model.split(df)
    out = pd.DataFrame({"TransactionID": test.TransactionID.values})
    for seed in SEEDS:
        model.PARAMS["seed"] = seed
        for name, extra in [("baseline", []), ("network", NETWORK_COLS)]:
            cols = model.feature_cols(df, velocity=True, extra=extra)
            m = model.fit(train, cols)
            out[f"{name}_s{seed}"] = m.predict(test[cols])
            print(f"seed {seed} {name}: trees={m.best_iteration_probe}", flush=True)
            if seed == SEEDS[0] and name == "network":
                m.save_model(str(model.PROCESSED / "network_model.txt"))
    model.PARAMS["seed"] = model.SEED
    out.to_parquet(RESULTS_PARQUET, index=False)
    return out


def compare(test: pd.DataFrame, scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = test.isFraud.to_numpy().astype(bool)
    amt = test.TransactionAmt.to_numpy()
    ring = ring_flag(test)
    rows, missed_rows = [], []
    for seed in SEEDS:
        base = scores[f"baseline_s{seed}"].to_numpy()
        flags = {
            "Baseline model": model.flag_top(base),
            "Ring flag alone": ring,
            "Model + ring alert": model.flag_top(ring_first(ring, base)),
            "Model + network features": model.flag_top(scores[f"network_s{seed}"].to_numpy()),
        }
        missed = y & ~flags["Baseline model"]
        for name, f in flags.items():
            caught = f & y
            rows.append({"approach": name, "seed": seed, "reviewed": int(f.sum()),
                         "fraud_caught": int(caught.sum()), "recall": caught.sum() / y.sum(),
                         "fraud_amt_caught": amt[caught].sum(),
                         "precision": caught.sum() / max(f.sum(), 1)})
            if name != "Baseline model":
                lost = y & flags["Baseline model"] & ~f
                missed_rows.append({"approach": name, "seed": seed,
                                    "baseline_missed": int(missed.sum()),
                                    "baseline_missed_amt": amt[missed].sum(),
                                    "newly_caught": int((f & missed).sum()),
                                    "pct_of_missed_caught": (f & missed).sum() / missed.sum(),
                                    "newly_caught_amt": amt[f & missed].sum(),
                                    "baseline_catches_lost": int(lost.sum()),
                                    "lost_amt": amt[lost].sum(),
                                    "net_fraud_gain": int((f & missed).sum() - lost.sum()),
                                    "net_amt_gain": amt[f & missed].sum() - amt[lost].sum()})
    order = list(flags)
    r = pd.DataFrame(rows).groupby("approach").agg(["mean", "std"]).drop(columns="seed").reindex(order)
    m = pd.DataFrame(missed_rows).groupby("approach").agg(["mean", "std"]).drop(columns="seed").reindex(order[1:])
    return r, m


if __name__ == "__main__":
    df = load_frame()
    train, test = model.split(df)
    scores = train_scores(df)
    r, m = compare(test, scores)
    pd.set_option("display.width", 250)
    print(r.round(4).to_string())
    print(m.round(4).to_string())
    r.to_csv(model.PROCESSED / "phase5_results.csv")
    m.to_csv(model.PROCESSED / "phase5_missed.csv")
