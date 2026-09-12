#!/usr/bin/env python3
"""Figure: paired effect of each target-access mode, per dataset.

Left panel: change in accuracy against the source-only reference, paired within
subject, with percentile bootstrap intervals over subjects.  Right panel: the
ratio of between-subject standard deviations for the same comparison, so a
condition that moves the mean and one that changes the spread can be told
apart at a glance.

Run:
    python figures/gen_fig_v4_effects.py results/v4_stats.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "figures"))

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

from paper_plot_style import COLORS, save_fig          # noqa: E402

ORDER = ["EA", "SUP", "EASUP", "SEL", "POOL"]
LABEL = {"EA": "EA", "SUP": "SUP", "EASUP": "EA+SUP", "SEL": "SEL", "POOL": "POOL"}
MAIN = {"EA", "SUP", "EASUP"}


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    results = json.loads(Path(argv[1]).read_text(encoding="utf-8"))

    rows = []          # (ytick label, dataset, cond, delta, ratio, is_main)
    yticks, ylabels, seps = [], [], []
    y = 0.0
    for res in results:
        y -= 1.15
        seps.append((y + 0.42, res["dataset"]))
        by_cond = {r["condition"]: r for r in res["deltas"]["rows"]}
        for cond in ORDER:
            r = by_cond.get(cond)
            if r is None:
                continue
            y -= 1.0
            rows.append((y, res["dataset"], cond, r["delta"], r["sd_ratio"],
                         cond in MAIN))
            yticks.append(y)
            ylabels.append(LABEL[cond])

    fig, (axl, axr) = plt.subplots(
        1, 2, figsize=(7.1, 0.34 * len(rows) + 1.0),
        gridspec_kw=dict(width_ratios=[1.35, 1.0], wspace=0.28))

    for ax, kind in ((axl, "delta"), (axr, "ratio")):
        for y, ds, cond, delta, ratio, is_main in rows:
            src = delta if kind == "delta" else ratio
            scale = 100.0 if kind == "delta" else 1.0
            point, lo, hi = src["point"] * scale, src["lo"] * scale, src["hi"] * scale
            colour = COLORS["blue"] if is_main else "0.55"
            marker = "o" if is_main else "s"
            ax.plot([lo, hi], [y, y], color=colour, lw=1.4,
                    solid_capstyle="round", zorder=2)
            ax.plot([point], [y], marker=marker, ms=4.2, color=colour,
                    zorder=3, clip_on=False)
        ax.axvline(0.0 if kind == "delta" else 1.0, color="0.2", lw=0.8,
                   ls=(0, (4, 3)), zorder=1)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels if kind == "delta" else [""] * len(ylabels))
        ax.set_ylim(min(yticks) - 0.7, 0.35)
        lo_all = min(min(r[3]["lo"] if kind == "delta" else r[4]["lo"]
                         for r in rows) * scale, 0.0 if kind == "delta" else 1.0)
        hi_all = max(max(r[3]["hi"] if kind == "delta" else r[4]["hi"]
                         for r in rows) * scale, 0.0 if kind == "delta" else 1.0)
        pad = 0.08 * (hi_all - lo_all) or 0.05
        ax.set_xlim(lo_all - pad, hi_all + pad)
        ax.tick_params(labelsize=8)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)

    axl.set_xlabel("change in accuracy vs. source-only (pp)", fontsize=9)
    axr.set_xlabel("between-subject SD ratio vs. source-only", fontsize=9)

    for ypos, name in seps:
        axl.text(-0.085, ypos, name, transform=axl.get_yaxis_transform(),
                 ha="left", va="center", fontsize=8.5, style="italic",
                 clip_on=False)

    fig.subplots_adjust(left=0.085, right=0.985, top=0.985, bottom=0.115)
    # Written into manuscript/figures/ so the paper's relative include path
    # resolves; the repo-level figures/ directory holds generators, not output.
    save_fig(fig, "fig_v4_effects",
             output_dir=str(ROOT / "manuscript" / "figures"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
