"""Phase 7 hero image: the case-study ring as a network plus its purchase timeline.

Left: who is linked to what. Rebuilt identities sit on a circle around the one
phone they all used; each identity also links out to its card. Right: every
purchase the ring made, one row per identity, so the lockstep timing shows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import patterns
import viz


def _layout(who: pd.DataFrame, tx: pd.DataFrame):
    uids = list(who.index)  # ordered by first fraud day
    n = len(uids)
    ang = np.pi / 2 - np.arange(n) * 2 * np.pi / n
    pos = {u: (np.cos(a), np.sin(a)) for u, a in zip(uids, ang)}
    cards = tx.groupby("card1").uid.apply(lambda s: sorted(set(s), key=uids.index))
    cpos = {}
    for card, us in cards.items():
        a = np.mean([np.arctan2(pos[u][1], pos[u][0]) for u in us]) if len(us) == 1 else \
            np.arctan2(np.mean([pos[u][1] for u in us]), np.mean([pos[u][0] for u in us]))
        cpos[card] = (1.85 * np.cos(a), 1.85 * np.sin(a))
    return uids, pos, cards, cpos


def plot_hero(con, cluster_id: int | None = None):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    viz.use_style()
    cid = cluster_id if cluster_id is not None else patterns.pick_case_study()
    tx, who = patterns.case_study(con, cid)
    uids, pos, cards, cpos = _layout(who, tx)
    label = {u: f"ID {i + 1}" for i, u in enumerate(uids)}
    brand = tx.groupby("card1").card4.first().str.capitalize()
    device = tx.DeviceInfo.dropna().mode()[0]
    model = device.split(" Build")[0].title().replace("Samsung Sm-", "Samsung SM-")

    fig = plt.figure(figsize=(15, 7.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.25], wspace=0.08)
    ax = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # ---- network ----
    ax.set_aspect("equal")
    ax.axis("off")
    ax.grid(False)
    for u in uids:
        x, y = pos[u]
        ax.plot([0, x], [0, y], color=viz.BLUE, lw=2.4, zorder=1, solid_capstyle="round")
    for card, us in cards.items():
        cx, cy = cpos[card]
        for u in us:
            x, y = pos[u]
            ax.plot([x, cx], [y, cy], color=viz.MUTED, lw=1.8, ls=(0, (3, 2)), zorder=1)
    # phone hub
    ax.scatter([0], [0], s=2600, color=viz.TEXT, zorder=3, edgecolors=viz.SURFACE, linewidths=3)
    ax.text(0, 0, "1\nphone", ha="center", va="center", color=viz.SURFACE, fontsize=12, fontweight="bold", zorder=4)
    # identities
    for u in uids:
        x, y = pos[u]
        r = who.loc[u]
        clean = r.txns - r.fraud
        ax.scatter([x], [y], s=1250, color=viz.FRAUD, zorder=3, edgecolors=viz.SURFACE, linewidths=2.5)
        if clean:
            ax.scatter([x], [y], s=1250, facecolors="none", edgecolors=viz.CLEAN, linewidths=3, zorder=4)
        ax.text(x, y, label[u], ha="center", va="center", color="white", fontsize=11, fontweight="bold", zorder=5)
    # cards
    for card, (cx, cy) in cpos.items():
        ax.scatter([cx], [cy], s=520, marker="s", color=viz.SURFACE, edgecolors=viz.TEXT_2, linewidths=1.8, zorder=3)
        ha = "left" if cx > 0.3 else "right" if cx < -0.3 else "center"
        dx = 0.2 if ha == "left" else -0.2 if ha == "right" else 0
        dy = 0 if ha != "center" else (0.24 if cy > 0 else -0.24)
        va = "center" if ha != "center" else ("bottom" if cy > 0 else "top")
        ax.text(cx + dx, cy + dy, f"{brand[card]}\ncard {card}", ha=ha, va=va, fontsize=10.5, color=viz.TEXT_2, zorder=5)
    ax.set_xlim(-2.75, 2.75)
    ax.set_ylim(-2.3, 2.3)
    handles = [
        Line2D([], [], marker="o", ls="", ms=13, color=viz.FRAUD, label="Rebuilt identity (fraud)"),
        Line2D([], [], marker="o", ls="", ms=13, mfc=viz.FRAUD, mec=viz.CLEAN, mew=2.5,
               label="Identity that also made a legitimate purchase"),
        Line2D([], [], marker="s", ls="", ms=10, mfc=viz.SURFACE, mec=viz.TEXT_2, mew=1.6, label="Payment card"),
        Line2D([], [], color=viz.BLUE, lw=2.4, label=f"Same phone: {model}"),
        Line2D([], [], color=viz.MUTED, lw=1.8, ls=(0, (3, 2)), label="Paid with this card"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=1, fontsize=11,
              handletextpad=0.5, columnspacing=1.4)

    # ---- timeline ----
    rows = {u: i for i, u in enumerate(uids)}
    t = tx.assign(row=tx.uid.map(rows))
    t["x"] = t.day + t.groupby(["row", "day"]).cumcount() * 0.6
    for flag, color, name in [(1, viz.FRAUD, "Fraud purchase"), (0, viz.CLEAN, "Legitimate purchase")]:
        d = t[t.isFraud == flag]
        ax2.scatter(d.x, d.row, s=d.TransactionAmt * 1.1 + 30, color=color,
                    edgecolors=viz.SURFACE, linewidths=1.5, zorder=3, label=name)
    ax2.set_yticks(range(len(uids)), [label[u] for u in uids])
    ax2.invert_yaxis()
    ax2.set_ylim(len(uids) - 0.4, -0.9)
    ax2.grid(axis="y", visible=False)
    ax2.grid(axis="x", visible=True)
    ax2.set_xlabel("Day in the dataset")
    lo, hi = int(t.day.min()), int(t.day.max())
    ax2.set_xlim(lo - 3, hi + 4)
    # busiest day callout
    busy = t[t.isFraud == 1].groupby("day").uid.nunique().idxmax()
    nb = t[(t.day == busy) & (t.ProductCD != "W")]
    ax2.axvspan(busy - 0.8, busy + 2.2, color=viz.GRID, alpha=0.6, zorder=0)
    hrs = int(nb.hour.max() - nb.hour.min())
    ax2.text(busy + 0.7, -0.65, f"Day {busy}: {len(nb)} purchases on\n{nb.card1.nunique()} cards within {hrs} hours",
             ha="center", va="center", fontsize=10.5, color=viz.TEXT)
    ax2.legend(loc="lower right", fontsize=11, markerscale=0.6)
    ax2.set_title("Every purchase the ring made (dot size = amount)", fontsize=13)

    n_fraud, n_tx = int(tx.isFraud.sum()), len(tx)
    amt = tx.loc[tx.isFraud == 1, "TransactionAmt"].sum()
    n200 = int((tx.TransactionAmt == 200).sum())
    viz.headline(fig, f"One phone, {tx.card1.nunique()} cards, {len(uids)} identities: a fraud ring hiding in plain sight",
                 f"{n_fraud} of {n_tx} purchases were fraud (\\${amt:,.0f}) over {hi - lo + 1} days. "
                 f"{n200} were exactly \\$200. No single purchase looks unusual; the links give it away.")
    viz.source_note(fig, f"Ring {cid}, found by linking rebuilt customer identities that share a device or card. "
                         "Data: IEEE-CIS Fraud Detection (Vesta)")
    return fig


if __name__ == "__main__":
    import graph
    fig = plot_hero(graph.connect())
    print(viz.save(fig, "01_hero_ring"))
