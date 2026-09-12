"""Emit the unrounded per-run variance table for the supplement.

Two reviews asked for this: the paper quotes three-decimal standard deviations
computed under two different pooling orders, and at that precision a reader
cannot recompute the variance shares or check them against the binomial bound.
This writes every intermediate at full precision, states which pooling order
each column uses, and records the bound the noise term must satisfy.

Columns, all computed within a run and then pooled on the variance scale:

  raw_var        s^2 of the per-subject accuracies in that run
  noise          N^-1 sum_s p_s(1-p_s)/(m_s-1), the conditionally binomial
                 estimate of the finite-trial term
  bound          max_s 0.25/(m_s-1); noise cannot exceed this
  corrected_var  raw_var - noise
  benchmark_var  variance of the random-regrouping null for that run

Usage:
    python analysis/variance_table.py [--out results/variance_table.md]
"""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from analysis.v4_stats import load  # noqa: E402

DATASETS = [
    ("BCI IV 2a", "results/v4_2a/*.json"),
    ("BNCI2014-002", "results/v4_002/*.json"),
    ("PhysionetMI", "results/v4_physionet/*.json"),
]
CONDITIONS = ("SRC", "SUP")
N_REGROUP = 2000


def per_run(table, cond, rng):
    """One row per run: the raw, noise, bound and benchmark terms."""

    rows = []
    for seed in sorted(table.get(cond, {})):
        subs = table[cond][seed]
        acc, sizes = [], []
        for s in sorted(subs, key=lambda x: int(x)):
            rec = subs[s]
            if "test_predictions" not in rec:
                continue
            p = np.asarray(rec["test_predictions"])
            y = np.asarray(rec["test_labels"])
            acc.append(float((p == y).mean()))
            sizes.append(len(y))
        if not acc:
            continue
        acc = np.asarray(acc)
        m = np.asarray(sizes, dtype=float)

        raw = float(acc.var(ddof=1))
        noise = float(np.mean(acc * (1.0 - acc) / (m - 1.0)))
        bound = float(np.max(0.25 / (m - 1.0)))

        # Random-regrouping benchmark on the same fixed predictions.
        correct = []
        for s in sorted(subs, key=lambda x: int(x)):
            rec = subs[s]
            if "test_predictions" not in rec:
                continue
            p = np.asarray(rec["test_predictions"])
            y = np.asarray(rec["test_labels"])
            correct.append(p == y)
        flat = np.concatenate(correct)
        bounds = np.cumsum(sizes)[:-1]
        draws = np.empty(N_REGROUP)
        for r in range(N_REGROUP):
            shuffled = flat[rng.permutation(len(flat))]
            draws[r] = np.var([g.mean() for g in np.split(shuffled, bounds)],
                              ddof=1)

        rows.append(dict(
            seed=int(seed), n_subjects=len(acc),
            m_min=int(m.min()), m_max=int(m.max()),
            raw_var=raw, noise=noise, bound=bound,
            corrected_var=max(raw - noise, 0.0),
            benchmark_var=float(draws.mean()),
        ))
    return rows


def main(argv):
    out_path = None
    if "--out" in argv:
        out_path = argv[argv.index("--out") + 1]

    rng = np.random.default_rng(20260911)
    lines = [
        "# Per-run variance table",
        "",
        "All quantities are computed inside a single run and, where the paper",
        "quotes one number, pooled across runs on the **variance** scale.",
        "Table I of the paper instead reports the standard deviation of",
        "run-averaged per-subject accuracy, which is a different pooling order",
        "and a slightly smaller number; the two are not interchangeable.",
        "",
        "`noise` is the conditionally binomial estimate",
        "`N^-1 sum_s p_s(1-p_s)/(m_s-1)`; it cannot exceed `bound`,",
        "`max_s 0.25/(m_s-1)`. `benchmark_var` is the random-regrouping null,",
        "which removes the subject structure as well as the trial noise and so",
        "is not the same estimand.",
        "",
        "| dataset | cond | run | N | m | raw_var | noise | bound | corrected_var | benchmark_var |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    summary = []
    for label, pattern in DATASETS:
        table, logs = load(pattern)
        if not logs:
            print("[skip] %s: no logs" % label)
            continue
        for cond in CONDITIONS:
            rows = per_run(table, cond, rng)
            if not rows:
                continue
            for r in rows:
                m = ("%d" % r["m_min"] if r["m_min"] == r["m_max"]
                     else "%d-%d" % (r["m_min"], r["m_max"]))
                lines.append(
                    "| %s | %s | %d | %d | %s | %.8f | %.8f | %.8f | %.8f | %.8f |"
                    % (label, cond, r["seed"], r["n_subjects"], m,
                       r["raw_var"], r["noise"], r["bound"],
                       r["corrected_var"], r["benchmark_var"]))
                if r["noise"] > r["bound"]:
                    lines.append("| | | | | | **NOISE EXCEEDS BOUND** | | | | |")
            raw = float(np.mean([r["raw_var"] for r in rows]))
            noi = float(np.mean([r["noise"] for r in rows]))
            summary.append((label, cond, raw, noi, max(raw - noi, 0.0)))

    lines += ["", "## Pooled over runs (variance scale)", "",
              "| dataset | cond | raw SD | corrected SD | variance share |",
              "|---|---|---:|---:|---:|"]
    for label, cond, raw, noi, corr in summary:
        lines.append("| %s | %s | %.6f | %.6f | %.2f%% |"
                     % (label, cond, raw ** 0.5, corr ** 0.5, 100 * noi / raw))

    text = "\n".join(lines) + "\n"
    if out_path:
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(text)
        print("written:", out_path)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
