#!/usr/bin/env python3
"""Figure 1: the target-access design.

Left: how the trials are divided.  Each held-out subject is split once into an
available half and a reserved test half; the source pool is split once into a
training set and a selection set.  Both divisions are identical in every
condition.

Right: the design matrix.  A filled marker means the condition receives that
input.  Three of the five columns are the same in all four rows, which is the
point of the figure: only the two procedure columns vary, and they are drawn
larger so that the reader sees this before reading any label.

Self-contained; only matplotlib is required.
Run:  python figures/gen_fig_design.py   ->  manuscript/figures/figure1.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "figures"))

import matplotlib                                   # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

from paper_plot_style import COLORS                 # noqa: E402

SIGNAL = COLORS["skyblue"]       # target signals, label-free
LABEL = COLORS["vermillion"]     # target labels
SOURCE = "#5c5c5c"
INK = "#1a1a1a"
MUTE = "#b8b8b8"
GREY = "#6a6a6a"

TITLE_Y = 0.935                  # figure coords, shared by both panel titles

# condition -> (name, gloss, uses alignment, uses supervision)
CONDITIONS = [
    ("SRC",    "uses neither",        False, False),
    ("EA",     "alignment only",      True,  False),
    ("SUP",    "supervision only",    False, True),
    ("EA+SUP", "both procedures",     True,  True),
]

# The two middle columns are procedures, not permissions: supervised
# training reads the available trials' signals as well as their labels, so
# this is not an orthogonal signals-versus-labels split.
COLUMNS = [
    ("source\ntraining", SOURCE, False),
    ("align on\n$C_s$", SIGNAL, True),
    ("supervise\non $C_s$", LABEL, True),
    ("source\nselection", SOURCE, False),
    ("scored on\n$T_s$", INK, False),
]


def bar(ax, x, y, w, h, fc, ec, label, pct, fs=7.0):
    """One segment of a division bar, with its share printed below it."""

    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec,
                           lw=0.9, zorder=2))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fs, color=INK, zorder=3)
    ax.text(x + w / 2, y - 0.052, pct, ha="center", va="center",
            fontsize=fs - 1.4, color=GREY, zorder=3)


def draw_split(ax):
    """Left panel: the two fixed divisions of the data."""

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    bar_w, bar_h = 0.80, 0.150
    note_x = 0.845

    # --- source pool -------------------------------------------------------
    ax.text(0.0, 0.855, "source pool  ($N{-}1$ subjects)", fontsize=7.2,
            color=INK, va="center")
    y0 = 0.650
    bar(ax, 0.0, y0, bar_w * 0.80, bar_h, "#ededed", SOURCE, "training", "80%")
    bar(ax, bar_w * 0.80, y0, bar_w * 0.20, bar_h, "#f8f8f8", SOURCE,
        "sel.", "20%")
    ax.text(note_x, y0 + bar_h / 2, "identical in\nall conditions",
            fontsize=6.2, color=GREY, va="center", ha="left", linespacing=1.4)

    # --- held-out subject --------------------------------------------------
    ax.text(0.0, 0.425, "held-out subject $s$", fontsize=7.2, color=INK,
            va="center")
    y1 = 0.220
    bar(ax, 0.0, y1, bar_w / 2, bar_h, "#ffffff", INK, "available $C_s$", "50%")
    bar(ax, bar_w / 2, y1, bar_w / 2, bar_h, "#ededed", INK,
        "reserved $T_s$", "50%")
    ax.text(note_x, y1 + bar_h / 2, "stratified,\nseeded per subject",
            fontsize=6.2, color=GREY, va="center", ha="left", linespacing=1.4)

    # --- the two roles, called out under the held-out bar ------------------
    # Clear of the percentage labels above them, which previously sat almost on
    # top of these bands.
    band_y = y1 - 0.108
    ax.plot([0.004, bar_w / 2 - 0.004], [band_y, band_y], color=SIGNAL,
            lw=2.4, solid_capstyle="butt", zorder=3)
    ax.text(bar_w / 4, band_y - 0.042, "the only material a condition may use",
            fontsize=6.2, color=SIGNAL, ha="center", va="top")
    ax.plot([bar_w / 2 + 0.004, bar_w - 0.004], [band_y, band_y], color=INK,
            lw=2.4, solid_capstyle="butt", zorder=3)
    ax.text(bar_w * 0.75, band_y - 0.042, "scored, never touched",
            fontsize=6.2, color=INK, ha="center", va="top")


def draw_matrix(ax):
    """Right panel: which inputs each condition receives."""

    n_rows, n_cols = len(CONDITIONS), len(COLUMNS)
    label_x = -1.62
    rule_x0, rule_x1 = label_x, n_cols - 0.55
    ax.set_xlim(label_x - 0.07, n_cols - 0.48)
    ax.set_ylim(-0.92, n_rows + 0.10)
    ax.axis("off")
    ax.invert_yaxis()

    # Highlight the two columns that vary, behind everything else.  The block
    # now starts below the spanning label and stops level with the last row
    # instead of ending in mid-air.
    top, bottom = -0.30, n_rows - 0.06
    for j, (_head, colour, varies) in enumerate(COLUMNS):
        if not varies:
            continue
        ax.add_patch(FancyBboxPatch(
            (j - 0.40, top), 0.80, bottom - top,
            boxstyle="round,pad=0,rounding_size=0.06",
            facecolor=colour, alpha=0.07, edgecolor="none", zorder=1))
    ax.text(1.5, -0.52, "the two procedures", ha="center", va="center",
            fontsize=6.4, color=GREY, zorder=3)

    for j, (head, colour, varies) in enumerate(COLUMNS):
        ax.text(j, -0.08, head, ha="center", va="center", fontsize=6.6,
                color=colour if varies else GREY, linespacing=1.35, zorder=3)

    ax.plot([rule_x0, rule_x1], [0.30, 0.30], color="#c9c9c9", lw=0.8,
            zorder=2)

    # Markers are drawn with scatter so they stay circular whatever the axes
    # aspect ratio turns out to be.  The three fixed columns are drawn smaller:
    # the eye should land on the two that change before it reads a header.
    on_pts, off_pts = {"x": [], "y": [], "c": [], "s": []}, {"x": [], "y": [],
                                                            "s": []}
    for i, (name, gloss, sig, lab) in enumerate(CONDITIONS):
        y = i + 0.78
        ax.text(label_x, y - 0.10, name, ha="left", va="center", fontsize=7.2,
                fontweight="bold", color=INK, zorder=3)
        ax.text(label_x, y + 0.19, gloss, ha="left", va="center", fontsize=6.1,
                color=GREY, zorder=3)
        for j, on in enumerate([True, sig, lab, True, True]):
            size = 66 if COLUMNS[j][2] else 46
            if on:
                on_pts["x"].append(j)
                on_pts["y"].append(y)
                on_pts["c"].append(COLUMNS[j][1])
                on_pts["s"].append(size)
            else:
                off_pts["x"].append(j)
                off_pts["y"].append(y)
                off_pts["s"].append(size)
        if i < n_rows - 1:
            ax.plot([rule_x0, rule_x1], [i + 1.30, i + 1.30],
                    color="#ededed", lw=0.7, zorder=2)

    ax.scatter(off_pts["x"], off_pts["y"], s=off_pts["s"], facecolors="white",
               edgecolors=MUTE, linewidths=0.9, zorder=4)
    ax.scatter(on_pts["x"], on_pts["y"], s=on_pts["s"], facecolors=on_pts["c"],
               edgecolors=on_pts["c"], linewidths=0.9, zorder=4)


def main():
    fig = plt.figure(figsize=(7.15, 2.05))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.95], wspace=0.10,
                          left=0.012, right=0.988, top=0.86, bottom=0.04)
    ax_l = fig.add_subplot(gs[0, 0])
    ax_r = fig.add_subplot(gs[0, 1])
    draw_split(ax_l)
    draw_matrix(ax_r)

    # Both titles are placed in figure coordinates at one shared height, each
    # at its own panel's left edge.  Setting them inside the axes let the two
    # halves drift apart, since the panels do not share a data range.
    for ax, text in ((ax_l, "How the trials are divided"),
                     (ax_r, "What each condition does with $C_s$")):
        fig.text(ax.get_position().x0, TITLE_Y, text, fontsize=8.2,
                 fontweight="bold", color=INK, va="center", ha="left")

    out = ROOT / "manuscript" / "figures" / "figure1.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    png = ROOT / "figures" / "figure1_preview.png"
    fig.savefig(png, dpi=230)
    print("[saved]", out)
    print("[saved]", png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
