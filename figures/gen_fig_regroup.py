#!/usr/bin/env python3
"""Figure: the fixed-prediction regrouping control.

One panel per dataset, two conditions per panel.  Each shaded distribution is
the between-group standard deviation obtained by shuffling that condition's
test-trial predictions into random groups of the true group sizes; the marked
line is the standard deviation of the same predictions grouped by true subject.
Nothing is retrained between the two, and the trial-weighted accuracy is
identical, so the gap is the part of the reported spread that belongs to the
subject rather than to how many trials each group happens to hold.

Both a source-only and a target-supervised arm are shown, which is what makes
this a training-information by aggregation grid rather than a single check.

Run:
    python figures/gen_fig_regroup.py "2a=results/v4_2a/*.json" ...
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "figures"))

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
from matplotlib.lines import Line2D                    # noqa: E402

from paper_plot_style import COLORS, save_fig          # noqa: E402
from analysis.v4_stats import load                     # noqa: E402

N_REPEAT = 2000
CONDITIONS = [("SRC", "source-only", COLORS["skyblue"]),
              ("SUP", "target-supervised", COLORS["orange"])]


def null_and_observed(table, cond, rng, n_repeat=N_REPEAT):
    """Return (null SDs, observed SD, pooled accuracy, group sizes) for one arm.

    Runs are pooled by concatenating each run's null draws: a subject's
    reserved test trials are the same objects in every run, and the question is
    about grouping rather than about run-to-run variance.
    """

    per_run_null, per_run_obs, pooled = [], [], []
    sizes_used = []
    for seed in sorted(table.get(cond, {})):
        subjects = table[cond][seed]
        correct, sizes = [], []
        for subj in sorted(subjects, key=lambda s: int(s)):
            rec = subjects[subj]
            if "test_predictions" not in rec:
                continue
            p = np.asarray(rec["test_predictions"])
            y = np.asarray(rec["test_labels"])
            correct.append(p == y)
            sizes.append(len(y))
        if not correct:
            continue
        flat = np.concatenate(correct)
        pooled.append(flat.mean())
        per_run_obs.append(np.std([c.mean() for c in correct], ddof=1))
        bounds = np.cumsum(sizes)[:-1]
        sds = np.empty(n_repeat)
        for r in range(n_repeat):
            shuffled = flat[rng.permutation(len(flat))]
            sds[r] = np.std([g.mean() for g in np.split(shuffled, bounds)],
                            ddof=1)
        per_run_null.append(sds)
        sizes_used = sizes
    if not per_run_null:
        return None
    return (np.concatenate(per_run_null), float(np.mean(per_run_obs)),
            float(np.mean(pooled)), sizes_used)


def main(argv):
    specs = [a for a in argv[1:] if "=" in a]
    if not specs:
        print(__doc__)
        return 2

    rng = np.random.default_rng(20260909)
    panels = []
    for spec in specs:
        name, _, pattern = spec.partition("=")
        table, logs = load(pattern)
        arms = {}
        for cond, _label, _colour in CONDITIONS:
            got = null_and_observed(table, cond, rng)
            if got is not None:
                arms[cond] = got
        if arms:
            panels.append((name, arms))
        else:
            print("[skip] %s: no stored per-trial predictions" % name)

    if not panels:
        print("nothing to plot")
        return 1

    fig, axes = plt.subplots(len(panels), 1, figsize=(3.4, 0.95 * len(panels)))
    if len(panels) == 1:
        axes = [axes]

    # One x-range for every panel.  The null shifts right as the trials per
    # group fall, and that shift is the point of the figure -- per-panel
    # autoscaling would hide it by re-centring each null in its own axes.
    all_lo = min(min(v[0].min(), v[1]) for _n, arms in panels
                 for v in arms.values())
    all_hi = max(max(v[0].max(), v[1]) for _n, arms in panels
                 for v in arms.values())
    bins = np.geomspace(all_lo * 0.85, all_hi * 1.15, 52)

    for ax, (name, arms) in zip(axes, panels):
        # Log axis: the observed spread sits several times above the null, so a
        # linear scale spends most of its width on empty space between them.
        ax.set_xscale("log")
        # The two observed lines can sit close together, so the ratio labels
        # are staggered vertically rather than left to collide.
        for slot, (cond, label, colour) in enumerate(CONDITIONS):
            if cond not in arms:
                continue
            null, obs, _pooled, sizes = arms[cond]
            ax.hist(null, bins=bins, color=colour, edgecolor=colour,
                    linewidth=0.5, alpha=0.45)
            ax.axvline(obs, color=colour, lw=1.5)
            ratio = obs / null.mean() if null.mean() else float("nan")
            ax.annotate(r"$%.1f{\times}$" % ratio,
                        xy=(obs, 0.08 + 0.36 * slot),
                        xycoords=("data", "axes fraction"),
                        xytext=(3, 0), textcoords="offset points",
                        ha="left", va="bottom", fontsize=6.8, color=colour)
        any_sizes = next(iter(arms.values()))[3]
        ax.set_ylabel(name, fontsize=8)
        ax.set_yticks([])
        ax.tick_params(labelsize=7)
        ax.set_xlim(bins[0], bins[-1])
        ax.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _p: ("%g" % v)))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.text(0.015, 0.06, "%d trials/group" % any_sizes[0],
                transform=ax.transAxes, fontsize=6.8, va="bottom", ha="left",
                color="0.35")

    axes[-1].set_xlabel("between-group SD of accuracy", fontsize=8)
    handles = [Line2D([0], [0], color=c, lw=1.5, label=l)
               for _cond, l, c in CONDITIONS]
    axes[0].legend(handles=handles, fontsize=6.4, frameon=False,
                   loc="upper left", handlelength=1.2, borderaxespad=0.2)
    axes[0].set_title("shaded: random regrouping of identical predictions;\n"
                      "line: the same predictions grouped by true subject",
                      fontsize=7, pad=4)
    fig.tight_layout(h_pad=0.7)
    # The paper includes figures by a path relative to manuscript/, so write
    # there rather than to the repo-level figures/ directory.
    save_fig(fig, "fig_regroup", output_dir=str(ROOT / "manuscript" / "figures"))

    for name, arms in panels:
        for cond, _label, _colour in CONDITIONS:
            if cond not in arms:
                continue
            null, obs, pooled, _sizes = arms[cond]
            print("%-12s %-5s pooled=%.4f  observed SD=%.4f  null SD=%.4f "
                  "[%.4f,%.4f]  ratio=%.2f  p=%.4f" % (
                      name, cond, pooled, obs, null.mean(),
                      np.percentile(null, 2.5), np.percentile(null, 97.5),
                      obs / null.mean(),
                      (np.sum(null >= obs) + 1) / (len(null) + 1)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
