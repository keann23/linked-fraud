"""Phase 6: patterns worth telling about the rings found in Phase 4.

Rings are chosen with training-period labels only (graph.is_ring). The patterns
below then *describe* those rings using all labels, days 1 to 182. That is fine
for description; nothing here feeds a model or the Phase 5 comparison.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import graph

SEED = 42


def ring_frame(con) -> pd.DataFrame:
    """Every transaction with a segment: ring member, other fraud, or clean."""
    c = pd.read_parquet(graph.PROCESSED / "cluster_table.parquet")
    u = pd.read_parquet(graph.CLUSTERS_PARQUET)[["uid", "cluster_id"]]
    rings = c.index[graph.is_ring(c)]
    con.register("u", u)
    con.register("r", pd.DataFrame({"cluster_id": rings}))
    t = con.sql("""
        SELECT b.TransactionID, b.day, b.TransactionDT, b.isFraud, b.TransactionAmt, b.ProductCD,
               b.DeviceType, b.DeviceInfo, b.has_identity, i.uid, u.cluster_id,
               (r.cluster_id IS NOT NULL) AS in_ring
        FROM base b JOIN identity i USING (TransactionID)
        LEFT JOIN u USING (uid) LEFT JOIN r ON r.cluster_id = u.cluster_id
    """).df()
    t["hour"] = (t.TransactionDT % 86400) // 3600
    t["segment"] = np.where(t.in_ring, "Ring", np.where(t.isFraud == 1, "Other fraud", "Clean"))
    return t


def ring_shape(t: pd.DataFrame) -> dict:
    """Size, link type, how much each member identity is used, how long fraud runs."""
    c = pd.read_parquet(graph.PROCESSED / "cluster_table.parquet")
    rc = c[graph.is_ring(c)]
    rt = t[t.in_ring]
    rf = rt[rt.isFraud == 1]
    span = rf.groupby("cluster_id").day.agg(lambda d: d.max() - d.min() + 1)
    per_member = rf.groupby("uid").size()
    return {
        "rings": len(rc),
        "identities": int(rc.n_uids.sum()),
        "median_size": float(rc.n_uids.median()),
        "max_size": int(rc.n_uids.max()),
        "device_link_share": round(float(rc.via_device.mean()), 2),
        "card_link_share": round(float(rc.via_card.mean()), 2),
        "email_link_share": round(float(rc.via_email.mean()), 2),
        "ring_txns": len(rt),
        "ring_fraud_rate": round(float(rt.isFraud.mean()), 3),
        "median_fraud_txns_per_member": float(per_member.median()),
        "members_with_1_or_2_fraud_txns": round(float((per_member <= 2).mean()), 3),
        "median_fraud_days": float(span.median()),
        "rings_fraud_in_one_day": round(float((span == 1).mean()), 3),
    }


def coordination(t: pd.DataFrame, n_pairs: int = 50_000) -> pd.DataFrame:
    """Do ring members start their fraud together?

    For each ring, sort member identities by the day of their first fraud and take
    the gap to the previous member. Compare with random pairs of fraud identities.
    """
    rf = t[t.in_ring & (t.isFraud == 1)]
    first = rf.groupby(["cluster_id", "uid"]).day.min().reset_index().sort_values(["cluster_id", "day"])
    ring_gap = first.groupby("cluster_id").day.diff().dropna().to_numpy()

    all_first = t[t.isFraud == 1].groupby("uid").day.min().to_numpy()
    rng = np.random.default_rng(SEED)
    a, b = rng.choice(all_first, n_pairs), rng.choice(all_first, n_pairs)
    rand_gap = np.abs(a - b)

    rows = []
    for name, gap in [("Ring members", ring_gap), ("Random pairs of fraudsters", rand_gap)]:
        rows.append({"group": name, "pairs": len(gap),
                     "same_day": float((gap == 0).mean()),
                     "within_7_days": float((gap <= 7).mean()),
                     "median_gap_days": float(np.median(gap))})
    return pd.DataFrame(rows).set_index("group")


def mix(t: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Device type, product and timing mix. Restricted to rows with identity data
    for device and product, since rings need device data to exist at all."""
    idn = t[t.has_identity]
    order = ["Ring", "Other fraud", "Clean"]
    amt = t.groupby("segment").TransactionAmt.median().reindex(order).rename("median_amount")
    night = t.groupby("segment").hour.apply(lambda h: h.between(0, 6).mean()).reindex(order).rename("share_0_to_6h")
    return {
        "device": pd.crosstab(idn.segment, idn.DeviceType, normalize="index").reindex(order),
        "product": pd.crosstab(idn.segment, idn.ProductCD, normalize="index").reindex(order),
        "amount_and_hour": pd.concat([amt, night], axis=1),
    }


def pick_case_study() -> int:
    """Rule, written before looking at the stories: purely device-linked, 6+ identities,
    highest training-period fraud rate."""
    c = pd.read_parquet(graph.PROCESSED / "cluster_table.parquet")
    r = c[graph.is_ring(c) & (c.via_device == 1) & (c.n_uids >= 6)]
    return int(r.sort_values(["train_fraud_rate", "n_uids"], ascending=False).index[0])


def case_study(con, cluster_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    u = pd.read_parquet(graph.CLUSTERS_PARQUET)
    members = u.loc[u.cluster_id == cluster_id, "uid"]
    con.register("m", pd.DataFrame({"uid": members}))
    tx = con.sql("""
        SELECT i.uid, b.day, (b.TransactionDT % 86400) // 3600 AS hour, b.isFraud, b.TransactionAmt,
               b.ProductCD, b.card1, b.card4, b.addr1, b.P_emaildomain, b.DeviceInfo, b.id_31
        FROM base b JOIN identity i USING (TransactionID) JOIN m USING (uid)
        ORDER BY b.TransactionDT
    """).df()
    who = tx.groupby("uid").agg(txns=("day", "size"), fraud=("isFraud", "sum"), first_day=("day", "min"),
                                last_day=("day", "max"), amount=("TransactionAmt", "sum"),
                                card_brand=("card4", "first"))
    return tx, who.sort_values("first_day")


def plot_coordination(co: pd.DataFrame):
    import matplotlib.pyplot as plt
    import viz

    viz.use_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    cats = ["same_day", "within_7_days"]
    labels = ["Same day", "Within a week"]
    x = np.arange(len(cats))
    w = 0.36
    series = [("Ring members", viz.FRAUD), ("Random pairs of fraudsters", viz.MUTED)]
    for k, (name, color) in enumerate(series):
        vals = co.loc[name, cats].to_numpy(dtype=float)
        pos = x + (k - 0.5) * (w + 0.02)
        ax.bar(pos, vals, width=w, color=color, label=name)
        for p, v in zip(pos, vals):
            ax.text(p, v + 0.02, (f"{v:.1%}" if v < 0.01 else f"{v:.0%}"), ha="center", va="bottom", fontsize=13, color=viz.TEXT)
    ax.set_xticks(x, labels, fontsize=13)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_ylabel("Share of pairs starting fraud this close together")
    ax.tick_params(axis="x", length=0)
    ax.legend(loc="upper right", fontsize=12)
    n = int(co.loc["Ring members", "pairs"])
    viz.headline(fig, "Ring members start stealing on the same day",
                 f"Days between ring members' first frauds ({n} member-to-member gaps in 44 rings), "
                 "vs random pairs of fraud identities.")
    viz.source_note(fig)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    con = graph.connect()
    t = ring_frame(con)
    print(ring_shape(t))
    print(coordination(t).round(3))
    for k, v in mix(t).items():
        print(k, v.round(3), sep="\n")
    cid = pick_case_study()
    tx, who = case_study(con, cid)
    print(cid, who, tx.to_string(), sep="\n")
