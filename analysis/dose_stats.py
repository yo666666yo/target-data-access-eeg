"""Calibration-budget dose analysis.

The main matrix leaves one quantity free by construction: with the epoch tied
to the source pool and the batch share declared, each available trial is drawn
``w*S/|C_s|`` times per epoch, and ``|C_s|`` differs 6.4-fold across our three
datasets.  Any cross-dataset ordering could therefore be a property of the
calibration budget rather than of the procedures.

This module reads the capped-budget runs, in which ``|C_s|`` is reduced while
the reserved test half is untouched, and reports each effect against the budget
that produced it.  Within a dataset the source pool, class count, channel count
and cohort are all fixed, so the budget and the repetition it implies are the
only things moving -- which is what makes the within-dataset curve, rather than
the cross-dataset comparison, the identifying one.

Usage:
    python analysis/dose_stats.py
"""

from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from analysis.v4_stats import (  # noqa: E402
    Resampler, condition_of, paired_delta, permutation_p,
)

# (label, full-budget glob, capped-budget glob, full |C_s|)
DATASETS = [
    ("BCI IV 2a", "results/v4_2a/*.json", "results/v4_dose_2a/*.json", 288),
    ("BNCI2014-002", "results/v4_002/*.json", "results/v4_dose_002/*.json", 80),
    ("PhysionetMI", "results/v4_physionet/*.json", None, 45),
]


def load_by_budget(pattern, full_budget):
    """Return {budget: {condition: {seed: {subject: accuracy}}}}."""

    out = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    if not pattern:
        return out
    for path in sorted(glob.glob(pattern)):
        if os.path.basename(path).startswith("_"):
            continue
        with open(path, encoding="utf-8") as fh:
            log = json.load(fh)
        if log.get("status") != "completed":
            continue
        cap = log["protocol"].get("available_trials_cap")
        budget = int(cap) if cap else int(full_budget)
        cond = condition_of(log)
        for subj, rec in log["per_subject"].items():
            out[budget][cond][log["seed"]][subj] = rec["main_accuracy"]
    return out


def repeats(patterns, budget, full_budget):
    """Mean realized draws of each available trial per epoch, from the logs.

    Only the supervised arms record this, since only they draw target trials;
    a run's budget is its cap when it has one and the full half otherwise.
    """

    vals = []
    for pattern in patterns:
        for path in glob.glob(pattern or ""):
            if os.path.basename(path).startswith("_"):
                continue
            with open(path, encoding="utf-8") as fh:
                log = json.load(fh)
            cap = log["protocol"].get("available_trials_cap")
            if (int(cap) if cap else int(full_budget)) != int(budget):
                continue
            batch = log.get("batch_size") or 0
            for rec in log["per_subject"].values():
                c = rec.get("controls") or {}
                n_t = c.get("n_target_train") or 0
                if not n_t:
                    continue
                if c.get("target_repeats_per_epoch"):
                    vals.append(c["target_repeats_per_epoch"])
                elif c.get("steps_per_epoch") and batch:
                    # The main-run logs predate this field; it is recoverable
                    # from what they do record.
                    vals.append(c["steps_per_epoch"] * batch
                                * c["realized_target_fraction"] / n_t)
    return float(np.mean(vals)) if vals else float("nan")


def seed_mean(table, cond):
    per = defaultdict(list)
    for _seed, subs in table.get(cond, {}).items():
        for s, a in subs.items():
            per[s].append(a)
    return {s: float(np.mean(v)) for s, v in per.items()}


def main():
    print("Calibration-budget dose curve. Effects are within-subject means over")
    print("runs, in percentage points; the reserved test half is identical at")
    print("every budget, so only the calibration set changes.\n")
    print("%-14s %7s %9s %22s %22s" % (
        "dataset", "|C_s|", "repeats", "SUP - SRC", "SUP - EA"))

    for label, full_glob, cap_glob, full_budget in DATASETS:
        # The source-only arm never touches the available half -- it is absent
        # from the training pool and from the scaler's fit set -- so it is the
        # same run at every budget and is deliberately not re-run capped.  Each
        # budget row therefore reuses the full-budget SRC, and only the arms
        # that consume C_s are replaced.
        merged = load_by_budget(full_glob, full_budget)
        for budget, table in load_by_budget(cap_glob, full_budget).items():
            for cond, seeds in table.items():
                merged[budget][cond] = seeds
            for cond in ("SRC",):
                if cond not in merged[budget] and cond in merged[full_budget]:
                    merged[budget][cond] = merged[full_budget][cond]

        for budget in sorted(merged, reverse=True):
            table = merged[budget]
            if not {"SRC", "EA", "SUP"} <= set(table):
                continue
            src, ea, sup = (seed_mean(table, c) for c in ("SRC", "EA", "SUP"))
            subs = sorted(set(src) & set(ea) & set(sup), key=lambda s: int(s))
            if not subs:
                continue
            R = Resampler(len(subs), 20260911)
            a = np.array([src[s] for s in subs])
            e = np.array([ea[s] for s in subs])
            u = np.array([sup[s] for s in subs])

            d1, d2 = paired_delta(a, u, R), paired_delta(e, u, R)
            print("%-14s %7d %9.2f  %+6.2f [%+6.2f,%+6.2f]  %+6.2f [%+6.2f,%+6.2f] %s" % (
                label, budget,
                repeats([full_glob, cap_glob], budget, full_budget),
                100 * d1["point"], 100 * d1["lo"], 100 * d1["hi"],
                100 * d2["point"], 100 * d2["lo"], 100 * d2["hi"],
                "p=%.4f" % permutation_p(e, u, R)))
    print()
    print("Read the within-dataset rows first: there the source pool, classes,")
    print("channels and cohort are fixed, so budget and repetition are the only")
    print("things moving. A flat SUP-EA column down a dataset's rows would mean")
    print("the cross-dataset reversal is not a budget effect.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
