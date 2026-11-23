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


def coverage(test: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """Share of each fraud segment the baseline model already flags (mean over seeds)."""
    y = test.isFraud.to_numpy().astype(bool)
    segments = {
        "All test fraud": y,
        "Fraud with a known-fraud neighbor": y & (test.nbr_fraud_uids.fillna(0).to_numpy() >= 1),
        "Fraud inside a ring": y & ring_flag(test),
    }
    rows = []
    for seed in SEEDS:
        flag = model.flag_top(scores[f"baseline_s{seed}"].to_numpy())
        for name, seg in segments.items():
            rows.append({"segment": name, "seed": seed, "fraud": int(seg.sum()),
                         "caught": int((seg & flag).sum()), "share": (seg & flag).sum() / seg.sum()})
    out = pd.DataFrame(rows).groupby("segment", sort=False).agg(
        fraud=("fraud", "first"), caught_mean=("caught", "mean"), share=("share", "mean"),
        share_min=("share", "min"), share_max=("share", "max"))
    return out


def plot_headline(r: pd.DataFrame, cov: pd.DataFrame):
    import matplotlib.pyplot as plt
    import viz

    viz.use_style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.6})

    # left: recall at the same budget
    names = list(r.index)[::-1]
    rec = r[("recall", "mean")].reindex(names)
    sd = r[("recall", "std")].reindex(names).fillna(0)
    colors = [viz.BLUE if n == "Model + network features" else viz.MUTED for n in names]
    a1.barh(names, rec, color=colors, height=0.6, xerr=sd, error_kw={"ecolor": viz.TEXT_2, "lw": 1.2, "capsize": 3})
    for i, v in enumerate(rec):
        a1.text(v + sd.iloc[i] + 0.012, i, f"{v:.1%}", va="center", fontsize=12, color=viz.TEXT)
    a1.set_xlim(0, 0.55)
    a1.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    a1.grid(axis="y", visible=False)
    a1.grid(axis="x", visible=True)
    a1.set_title("Share of test fraud caught, same 2% review budget", fontsize=13)

    # right: what the baseline already catches
    segs = list(cov.index)[::-1]
    sh = cov.share.reindex(segs)
    lo = (sh - cov.share_min.reindex(segs)).to_numpy()
    hi = (cov.share_max.reindex(segs) - sh).to_numpy()
    colors = [viz.MUTED if s == "All test fraud" else viz.FRAUD for s in segs]
    a2.barh(segs, sh, color=colors, height=0.6, xerr=[lo, hi], error_kw={"ecolor": viz.TEXT_2, "lw": 1.2, "capsize": 3})
    for i, s in enumerate(segs):
        a2.text(sh[s] + hi[i] + 0.03, i, f"{sh[s]:.0%}  (of {cov.fraud[s]:,})", va="center", fontsize=12, color=viz.TEXT)
    a2.set_xlim(0, 1.25)
    a2.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    a2.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    a2.grid(axis="y", visible=False)
    a2.grid(axis="x", visible=True)
    a2.set_title("Share the baseline model already catches", fontsize=13)

    gain = r.loc["Model + network features", ("recall", "mean")] - r.loc["Baseline model", ("recall", "mean")]
    viz.headline(fig, "The transaction model already catches most ring fraud",
                 f"Adding network features moved recall by {gain * 100:+.1f} points, inside the seed-to-seed noise.")
    viz.source_note(fig, "Review budget: top 2% of test transactions (3,520 of 175,998, last 2 months). "
                         "Mean of 3 model seeds; whiskers show the spread.\n"
                         "Data: IEEE-CIS Fraud Detection (Vesta), 590,540 transactions")
    return fig


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
