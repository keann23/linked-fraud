"""Shared chart style so every figure in the repo looks like one family."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"

# Roles, not raw hex, throughout the charts.
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
MUTED = "#a3a29c"      # baselines / comparison series
GRID = "#e6e5e0"
BLUE = "#2a78d6"       # slot 1: our method / clean
ORANGE = "#eb6834"     # slot 2: fraud
AQUA = "#1baf7a"       # slot 3
FRAUD = ORANGE
CLEAN = BLUE


def use_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlecolor": TEXT,
        "axes.labelcolor": TEXT_2,
        "axes.labelsize": 12,
        "axes.edgecolor": GRID,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": TEXT_2,
        "ytick.color": TEXT_2,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "ytick.left": False,
        "legend.frameon": False,
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.3,
    })


def headline(fig, title: str, subtitle: str | None = None) -> None:
    """Takeaway title + plain-English subtitle, left aligned above all panels."""
    fig.text(0.0, 1.06, title, fontsize=17, fontweight="bold", color=TEXT,
             ha="left", va="bottom", transform=fig.transFigure)
    if subtitle:
        fig.text(0.0, 1.0, subtitle, fontsize=12, color=TEXT_2,
                 ha="left", va="bottom", transform=fig.transFigure)


def source_note(fig, text: str = "Data: IEEE-CIS Fraud Detection (Vesta), 590,540 transactions") -> None:
    fig.text(0.0, -0.04, text, fontsize=9.5, color=TEXT_2, ha="left", va="top",
             transform=fig.transFigure)


def save(fig, name: str) -> Path:
    FIGURES.mkdir(exist_ok=True)
    path = FIGURES / f"{name}.png"
    fig.savefig(path)
    return path
