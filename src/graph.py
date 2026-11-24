"""Phase 4: link identities into a network and find clusters (candidate rings).

Leakage rules:
  * edges use attributes only, never labels
  * clusters are scored with labels from the training period (days 1-120) only
  * for any one uid, "neighbor risk" excludes that uid's own labels, so the ring
    signal measures guilt by association, not the customer's own history
"""
from __future__ import annotations

import duckdb
import networkx as nx
import numpy as np
import pandas as pd

from identity import IDENTITY_PARQUET
from load import BASE_PARQUET, PROCESSED, SQL
from model import TRAIN_LAST_DAY

HUB_MAX = 10   # tuned: at 20, chained links merge 20K uids into one component
LOUVAIN_MIN = 30      # components bigger than this get split with Louvain
SEED = 42
CLUSTERS_PARQUET = PROCESSED / "clusters.parquet"
EDGES_PARQUET = PROCESSED / "edges.parquet"


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"CREATE VIEW base AS SELECT * FROM read_parquet('{BASE_PARQUET}')")
    con.execute(f"CREATE VIEW identity AS SELECT * FROM read_parquet('{IDENTITY_PARQUET}')")
    return con


def build_edges(con, hub_max: int = HUB_MAX) -> pd.DataFrame:
    con.execute((SQL / "04_edges.sql").read_text().replace("{hub_max}", str(hub_max)))
    return con.sql("SELECT * FROM edges").df()


def find_clusters(edges: pd.DataFrame, louvain_min: int = LOUVAIN_MIN) -> pd.DataFrame:
    """Connected components; components larger than louvain_min are split by Louvain."""
    # Integer node ids + sorted insertion so Louvain is reproducible run to run
    # (sets of strings iterate in a different order under each Python hash seed).
    names = np.array(sorted(set(edges.src) | set(edges.dst)))
    idx = {n: i for i, n in enumerate(names)}
    e = edges.assign(s=edges.src.map(idx), d=edges.dst.map(idx)).sort_values(["s", "d"])
    G = nx.Graph()
    G.add_nodes_from(range(len(names)))
    G.add_weighted_edges_from(e[["s", "d", "weight"]].itertuples(index=False))
    G.remove_nodes_from([n for n in range(len(names)) if G.degree(n) == 0])
    rows, cid = [], 0
    for comp in sorted(nx.connected_components(G), key=lambda c: (-len(c), min(c))):
        if len(comp) > louvain_min:
            parts = nx.community.louvain_communities(G.subgraph(sorted(comp)), weight="weight", seed=SEED)
            parts = sorted(parts, key=lambda p: (-len(p), min(p)))
            method = "louvain"
        else:
            parts, method = [comp], "component"
        for p in parts:
            rows += [(names[u], cid, len(comp), method) for u in p]
            cid += 1
    out = pd.DataFrame(rows, columns=["uid", "cluster_id", "component_size", "method"])
    out["cluster_size"] = out.groupby("cluster_id").uid.transform("size")
    return out


def graph_shape(edges: pd.DataFrame, clusters: pd.DataFrame) -> dict:
    comp = clusters.drop_duplicates("cluster_id")
    return {
        "linked_uids": len(clusters),
        "edges": len(edges),
        "largest_component": int(clusters.component_size.max()),
        "clusters": int(clusters.cluster_id.nunique()),
        "clusters_3plus": int((comp.cluster_size >= 3).sum()),
        "largest_cluster": int(comp.cluster_size.max()),
        "edges_via_device": int(edges.via_device.sum()),
        "edges_via_card": int(edges.via_card.sum()),
        "edges_via_email": int(edges.via_email.sum()),
    }


def uid_stats(con) -> pd.DataFrame:
    """Per uid: train-period txns/fraud (labels allowed) and test-period txns."""
    return con.sql(f"""
        SELECT uid,
               COUNT(*) FILTER (WHERE day <= {TRAIN_LAST_DAY})                 AS train_txns,
               COALESCE(SUM(isFraud) FILTER (WHERE day <= {TRAIN_LAST_DAY}), 0) AS train_fraud,
               COUNT(*) FILTER (WHERE day >  {TRAIN_LAST_DAY})                 AS test_txns,
               SUM(TransactionAmt)                                             AS total_amt
        FROM identity WHERE uid IS NOT NULL GROUP BY 1
    """).df()


def score_clusters(clusters: pd.DataFrame, ustats: pd.DataFrame, edges: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster table (train labels only) + per-uid leave-one-out neighbor risk."""
    u = clusters.merge(ustats, on="uid", how="left").fillna({"train_txns": 0, "train_fraud": 0, "test_txns": 0})
    u["train_fraud_uid"] = (u.train_fraud > 0).astype(int)
    g = u.groupby("cluster_id")
    c = pd.DataFrame({
        "n_uids": g.size(),
        "train_txns": g.train_txns.sum(),
        "train_fraud": g.train_fraud.sum(),
        "train_fraud_uids": g.train_fraud_uid.sum(),
        "uids_with_train": g.train_txns.apply(lambda s: int((s > 0).sum())),
        "test_txns": g.test_txns.sum(),
        "total_amt": g.total_amt.sum().round(0),
    })
    c["train_fraud_rate"] = (c.train_fraud / c.train_txns.replace(0, np.nan)).round(4)

    # link mix inside each cluster
    e = edges.merge(clusters[["uid", "cluster_id"]], left_on="src", right_on="uid")
    e = e.merge(clusters[["uid", "cluster_id"]], left_on="dst", right_on="uid", suffixes=("", "_d"))
    e = e[e.cluster_id == e.cluster_id_d]
    mix = e.groupby("cluster_id")[["via_device", "via_card", "via_email"]].mean().round(2)
    c = c.join(mix).sort_values(["train_fraud_rate", "train_fraud_uids", "n_uids"], ascending=False)

    # leave-one-out: risk of everyone else in my cluster
    u = u.join(c[["train_txns", "train_fraud", "train_fraud_uids"]].add_prefix("c_"), on="cluster_id")
    u["nbr_train_txns"] = u.c_train_txns - u.train_txns
    u["nbr_train_fraud"] = u.c_train_fraud - u.train_fraud
    u["nbr_fraud_uids"] = u.c_train_fraud_uids - u.train_fraud_uid
    u["nbr_fraud_rate"] = u.nbr_train_fraud / u.nbr_train_txns.replace(0, np.nan)
    keep = ["uid", "cluster_id", "cluster_size", "component_size", "method", "train_txns", "train_fraud",
            "test_txns", "nbr_train_txns", "nbr_train_fraud", "nbr_fraud_uids", "nbr_fraud_rate"]
    return c, u[keep]


def test_view(con, uid_scores: pd.DataFrame) -> pd.DataFrame:
    """Test-period transactions with their uid's cluster info and the baseline flag."""
    con.register("uid_scores", uid_scores)
    return con.sql(f"""
        SELECT i.TransactionID, i.day, i.isFraud, i.TransactionAmt, i.uid,
               s.* EXCLUDE (uid), p.flag_baseline, p.score_baseline
        FROM identity i
        JOIN read_parquet('{PROCESSED / "predictions.parquet"}') p USING (TransactionID)
        LEFT JOIN uid_scores s USING (uid)
    """).df()


def is_ring(c: pd.DataFrame, min_uids: int = 3, min_fraud_uids: int = 2, min_rate: float = 0.5) -> pd.Series:
    """A high-risk cluster, judged on train labels only: 3+ uids, fraud spread
    across 2+ different uids, and at least half of train transactions fraud."""
    return (c.n_uids >= min_uids) & (c.train_fraud_uids >= min_fraud_uids) & (c.train_fraud_rate >= min_rate)


def gate_metrics(tv: pd.DataFrame, c: pd.DataFrame) -> dict:
    ring_ids = set(c.index[is_ring(c)])
    y = tv.isFraud == 1
    missed = y & ~tv.flag_baseline
    in_ring = tv.cluster_id.isin(ring_ids)
    # guilt by association only: the uid itself had no train fraud
    assoc = in_ring & (tv.train_fraud.fillna(0) == 0)
    nbr = (tv.nbr_fraud_uids.fillna(0) >= 1) & (tv.train_fraud.fillna(0) == 0)
    amt = tv.TransactionAmt

    def block(mask):
        return {"txns": int(mask.sum()), "fraud_rate": round(float(y[mask].mean()), 3) if mask.any() else None,
                "pct_of_test_fraud": round(100 * float((mask & y).sum() / y.sum()), 1),
                "pct_of_missed_fraud": round(100 * float((mask & missed).sum() / missed.sum()), 1),
                "missed_fraud_amt": round(float(amt[mask & missed].sum()))}
    return {
        "ring_clusters": len(ring_ids),
        "uids_in_rings": int(c.loc[list(ring_ids), "n_uids"].sum()),
        "test_base_fraud_rate": round(float(y.mean()), 4),
        "in_ring (any)": block(in_ring),
        "in_ring, own history clean": block(assoc),
        "any fraud neighbor, own history clean": block(nbr),
    }


def run(hub_max: int = HUB_MAX):
    con = connect()
    edges = build_edges(con, hub_max)
    cl = find_clusters(edges)
    c, u = score_clusters(cl, uid_stats(con), edges)
    edges.to_parquet(EDGES_PARQUET, index=False)
    u.to_parquet(CLUSTERS_PARQUET, index=False)
    c.to_parquet(PROCESSED / "cluster_table.parquet")
    return con, edges, cl, c, u


def plot_size_vs_fraud(c: pd.DataFrame):
    import matplotlib.pyplot as plt
    import viz

    viz.use_style()
    # every ring, plus other groups with enough training history to have a stable rate
    d = c[((c.n_uids >= 2) & (c.train_txns >= 3)) | is_ring(c)].copy()
    ring = is_ring(d)
    rng = np.random.default_rng(SEED)
    x = d.n_uids * np.exp(rng.uniform(-0.12, 0.12, len(d)))
    y = (d.train_fraud_rate + rng.uniform(-0.012, 0.012, len(d))).clip(-0.02, 1.02)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x[~ring], y[~ring], s=14, color=viz.MUTED, alpha=0.35, linewidths=0, label="Linked group")
    ax.scatter(x[ring], y[ring], s=46, color=viz.FRAUD, edgecolors=viz.SURFACE, linewidths=1.2,
               label="Ring: 3+ identities, 2+ with fraud, 50%+ fraud", zorder=3)
    ax.set_xscale("log")
    ax.set_xticks([2, 3, 5, 10, 20, 50, 100, 200], ["2", "3", "5", "10", "20", "50", "100", "200"])
    ax.set_xlabel("Identities in the linked group (log scale)")
    ax.set_ylabel("Fraud rate, training period")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="center right", fontsize=11, markerscale=1.2)
    n_rings, n_groups = int(ring.sum()), len(d)
    viz.headline(fig, f"Most linked groups are clean; {n_rings} small rings are almost all fraud",
                 f"{n_groups:,} groups of identities linked by shared devices, cards or rare email domains. "
                 f"Scored with training-period labels only.")
    viz.source_note(fig)
    fig.tight_layout()
    return fig
