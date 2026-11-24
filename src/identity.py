"""Phase 2: rebuild customer identities (uids) and check how label-pure they are.

Why purity works as a check: in this dataset, once a customer is caught, their
later transactions tend to be labeled fraud too. So if a rebuilt identity really
is one customer, its transactions should be all fraud or all clean.

Caveat we report alongside every number: a group of 1 is pure by definition,
and with a 3.5% base rate most small groups are pure by chance. So we compare
against a shuffled-label baseline (same groups, labels randomly reassigned).
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from load import BASE_PARQUET, PROCESSED, SQL

IDENTITY_PARQUET = PROCESSED / "identity.parquet"
GROUPINGS = {"card1 alone": "uid_card1", "uid": "uid", "uid + email": "uid_email"}


def build_identity(force: bool = False) -> Path:
    if IDENTITY_PARQUET.exists() and not force:
        return IDENTITY_PARQUET
    con = duckdb.connect()
    con.execute(f"CREATE VIEW base AS SELECT * FROM read_parquet('{BASE_PARQUET}')")
    con.execute((SQL / "02_identity.sql").read_text())
    con.execute(f"COPY identity TO '{IDENTITY_PARQUET}' (FORMAT parquet)")
    con.close()
    return IDENTITY_PARQUET


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"CREATE VIEW identity AS SELECT * FROM read_parquet('{IDENTITY_PARQUET}')")
    return con


def group_table(con, col: str, label: str = "isFraud") -> str:
    """SQL for one row per group: size and fraud share."""
    return f"""
        SELECT {col} AS g, COUNT(*) AS n, AVG({label}) AS fraud_share
        FROM identity_labeled WHERE {col} IS NOT NULL GROUP BY 1
    """


def purity_stats(con, col: str, shuffled: bool = False, seed: float = 0.42) -> dict:
    """Purity for one grouping. shuffled=True reassigns labels at random first."""
    if shuffled:
        con.execute(f"SELECT setseed({seed})")
        con.execute("""
            CREATE OR REPLACE TEMP TABLE identity_labeled AS
            WITH rnd AS MATERIALIZED (SELECT isFraud, random() AS u FROM identity),
                 lab AS (SELECT isFraud, ROW_NUMBER() OVER (ORDER BY u) AS r FROM rnd),
                 ids AS (SELECT * EXCLUDE (isFraud), ROW_NUMBER() OVER (ORDER BY TransactionID) AS r FROM identity)
            SELECT ids.*, lab.isFraud FROM ids JOIN lab USING (r)
        """)
    else:
        con.execute("CREATE OR REPLACE TEMP TABLE identity_labeled AS SELECT * FROM identity")

    row = con.sql(f"""
        WITH g AS ({group_table(con, col)})
        SELECT
            COUNT(*)                                                   AS groups,
            SUM(n)                                                     AS txns_covered,
            ROUND(AVG(n), 2)                                           AS txns_per_group_mean,
            MEDIAN(n)                                                  AS txns_per_group_median,
            ROUND(100 * AVG((n = 1)::INT), 1)                          AS pct_singletons,
            -- purity among groups with 2+ transactions (singletons are trivially pure)
            COUNT(*) FILTER (WHERE n >= 2)                             AS groups_2plus,
            ROUND(100 * AVG((fraud_share IN (0, 1))::INT) FILTER (WHERE n >= 2), 1)
                                                                       AS pct_pure_2plus,
            -- the sharper test: groups that contain any fraud, are they all fraud?
            COUNT(*) FILTER (WHERE n >= 2 AND fraud_share > 0)         AS fraud_groups_2plus,
            ROUND(100 * AVG((fraud_share = 1)::INT) FILTER (WHERE n >= 2 AND fraud_share > 0), 1)
                                                                       AS pct_all_fraud_given_any,
            -- transaction-weighted: share of fraud txns that sit in an all-fraud group of 2+
            ROUND(100 * SUM(n * fraud_share) FILTER (WHERE n >= 2 AND fraud_share = 1)
                  / SUM(n * fraud_share), 1)                           AS pct_fraud_txns_in_pure_fraud_groups
        FROM g
    """).df().iloc[0].to_dict()
    return row


def compare_groupings(con) -> pd.DataFrame:
    rows = []
    for name, col in GROUPINGS.items():
        for shuffled in (False, True):
            r = purity_stats(con, col, shuffled=shuffled)
            r.update(grouping=name, labels="shuffled" if shuffled else "real")
            rows.append(r)
    df = pd.DataFrame(rows).set_index(["grouping", "labels"])
    return df


def fraud_share_distribution(con, col: str) -> pd.DataFrame:
    con.execute("CREATE OR REPLACE TEMP TABLE identity_labeled AS SELECT * FROM identity")
    return con.sql(f"""
        WITH g AS ({group_table(con, col)})
        SELECT n, fraud_share FROM g WHERE n >= 2
    """).df()


def plot_purity(con, stats: pd.DataFrame):
    """Two panels: among multi-transaction groups that contain any fraud,
    what share of each group's transactions is fraud?"""
    import numpy as np
    import matplotlib.pyplot as plt
    import viz

    viz.use_style()
    bins = np.linspace(0, 1, 11)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    panels = [("card1 alone", "uid_card1", viz.MUTED), ("Rebuilt identity (uid)", "uid", viz.BLUE)]
    for ax, (label, col, color) in zip(axes, panels):
        d = fraud_share_distribution(con, col)
        d = d[d.fraud_share > 0]
        counts, _ = np.histogram(d.fraud_share, bins=bins)
        pct = 100 * counts / counts.sum()
        ax.bar(bins[:-1] + 0.05, pct, width=0.1 * 0.9, color=color, linewidth=0)
        name = "uid" if col == "uid" else "card1 alone"
        all_fraud = stats.loc[(name, "real"), "pct_all_fraud_given_any"]
        ax.annotate(f"{all_fraud:.0f}% are\n100% fraud", xy=(0.95, pct[-1]), xytext=(0.95, pct[-1] + 6),
                    ha="center", va="bottom", fontsize=12, color=viz.TEXT, fontweight="bold")
        ax.set_title(f"{label}  ({len(d):,} groups)", fontsize=13)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
        ax.set_xlabel("Share of the group's transactions that are fraud")
        ax.set_xlim(-0.02, 1.02)
    axes[0].set_ylabel("% of groups")
    axes[0].set_ylim(0, max(axes[0].get_ylim()[1], axes[1].get_ylim()[1]) + 10)
    axes[0].yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    shuffled = stats.loc[("uid", "shuffled"), "pct_all_fraud_given_any"]
    viz.headline(
        fig,
        "Rebuilt identities separate fraudsters from customers; card numbers alone don't",
        f"Groups with 2+ transactions and at least one fraud. With labels shuffled at random, "
        f"only {shuffled:.1f}% of uids would be 100% fraud.",
    )
    viz.source_note(fig)
    fig.tight_layout(w_pad=3)
    return fig


if __name__ == "__main__":
    build_identity(force=True)
    con = connect()
    pd.set_option("display.width", 250)
    print(compare_groupings(con).T.to_string())
