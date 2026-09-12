"""Compare our EEGNet against braindecode's EEGNetv4 under the same protocol.

The decoder is the only thing that changes: the split, the alignment, the
standardization, the sampler, the optimizer and the stopping rule all come from
the same code path.  Two questions follow, and they are separate:

  1. Does the reference implementation lift the absolute accuracy?  A gap here
     would say our decoder is a weak carrier, which matters for how far the
     protocol claims travel.
  2. Do the effects keep their direction and rough size?  This is the question
     the paper's conclusions actually rest on; an effect that survives a change
     of decoder is a property of the protocol rather than of our model.

A reference implementation that scores higher but reports the same effects
would strengthen the paper, not weaken it.

Usage:
    python analysis/reference_check.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from analysis.v4_stats import (  # noqa: E402
    Resampler, load, paired_delta, permutation_p, seed_mean_accuracy,
)

PAIRS = [
    ("BCI IV 2a", "results/v4_2a/*.json", "results/v4_ref_2a/*.json"),
    ("BNCI2014-002", "results/v4_002/*.json", "results/v4_ref_002/*.json"),
    ("PhysionetMI", "results/v4_physionet/*.json",
     "results/v4_ref_physionet/*.json"),
]
ARMS = ("EA", "SUP", "SEL")


def effects(pattern):
    """Return (reference accuracy, {arm: (delta, ci_lo, ci_hi, p)})."""

    table, logs = load(pattern)
    if not logs or "SRC" not in table:
        return None, {}

    # Every arm must cover the same runs.  Averaging over whatever seeds happen
    # to have landed and then pairing the arms compares different runs to each
    # other, which silently produces a plausible-looking number from an
    # incomplete sweep.
    seeds = {arm: set(table[arm]) for arm in ("SRC",) + ARMS if arm in table}
    common = set.intersection(*seeds.values()) if seeds else set()
    incomplete = {a: sorted(s) for a, s in seeds.items() if set(s) != common}
    if incomplete:
        return None, {"__incomplete__": (seeds, common)}

    base = seed_mean_accuracy(table, "SRC")
    subs = sorted(base, key=lambda s: int(s))
    a = np.array([base[s] for s in subs])
    R = Resampler(len(subs), 20260911)

    out = {}
    for arm in ARMS:
        if arm not in table:
            continue
        got = seed_mean_accuracy(table, arm)
        if set(got) != set(base):
            continue
        b = np.array([got[s] for s in subs])
        d = paired_delta(a, b, R)
        out[arm] = (100 * d["point"], 100 * d["lo"], 100 * d["hi"],
                    permutation_p(a, b, R))
    return float(a.mean()), out


def main():
    print("Reference-implementation check: our EEGNet vs braindecode EEGNetv4,")
    print("identical protocol, decoder swapped.\n")
    any_ref = False
    for label, ours_glob, ref_glob in PAIRS:
        acc_o, eff_o = effects(ours_glob)
        acc_r, eff_r = effects(ref_glob)
        if acc_r is None:
            if "__incomplete__" in eff_r:
                seeds, common = eff_r["__incomplete__"]
                print("%-14s reference sweep incomplete, not comparable: %s"
                      % (label, {a: sorted(s) for a, s in seeds.items()}))
            else:
                print("%-14s reference runs not present yet" % label)
            continue
        any_ref = True
        print("%s  source-only: ours %.4f | braindecode %.4f  (%+.2f pp)"
              % (label, acc_o, acc_r, 100 * (acc_r - acc_o)))
        print("   %-5s %24s %24s %s" % ("arm", "ours", "braindecode", "agree?"))
        for arm in ARMS:
            if arm not in eff_o or arm not in eff_r:
                continue
            do, dr = eff_o[arm], eff_r[arm]
            same_sign = (do[0] > 0) == (dr[0] > 0)
            overlap = not (do[2] < dr[1] or dr[2] < do[1])
            # Comparing point-estimate signs alone is too crude: an effect that
            # is simply not resolved under the reference decoder will flip sign
            # by chance while remaining entirely consistent with ours.  What
            # matters is whether each arm rejects, and whether the intervals
            # are compatible.
            ours_rej, ref_rej = do[3] < 0.05, dr[3] < 0.05
            if ours_rej and ref_rej:
                verdict = ("replicated" if same_sign
                           else "CONTRADICTED: both reject, opposite signs")
            elif ours_rej and not ref_rej:
                verdict = ("not replicated: reference does not reject"
                           + ("" if overlap else ", CIs disjoint"))
            elif not ours_rej and ref_rej:
                verdict = "reference rejects where ours does not"
            else:
                verdict = "neither rejects"
            print("   %-5s %+7.2f [%+6.2f,%+6.2f]  %+7.2f [%+6.2f,%+6.2f]  %s"
                  % (arm, do[0], do[1], do[2], dr[0], dr[1], dr[2], verdict))
        print()
    if any_ref:
        print("A sign change in any arm would mean that effect is a property of")
        print("our decoder, not of the protocol, and the corresponding claim")
        print("would have to be withdrawn or rescoped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
