"""Random versus chronological division of the held-out subject, on BCI IV 2a.

Every result elsewhere in this study divides a subject's trials at random, so
the available and reserved halves are interleaved in time. That answers a
question about an offline calibration block and says nothing about a decoder
holding up on a later recording: trials minutes apart share drift, impedance
and attention state, and a random split hands some of that to training.

BCI IV 2a is recorded as two sessions of 288 trials, loaded in order, so the
calibration budget can be held at 288/288 while the partition moves from
arbitrary to chronological. If the effects shrink under that move, the random
split was flattering them.

Usage:
    python analysis/session_split.py
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

ARMS = ("EA", "SUP", "EASUP")
RANDOM_GLOB = "results/v4_2a/*.json"
SESSION_GLOB = "results/v4_sess_2a/*.json"


def effects(pattern):
    table, logs = load(pattern)
    if not logs or "SRC" not in table:
        return None, {}, 0
    seeds = {a: set(table[a]) for a in ("SRC",) + ARMS if a in table}
    common = set.intersection(*seeds.values()) if seeds else set()
    if any(set(s) != common for s in seeds.values()):
        return None, {"__incomplete__": {a: sorted(s)
                                         for a, s in seeds.items()}}, 0

    base = seed_mean_accuracy(table, "SRC")
    subs = sorted(base, key=lambda s: int(s))
    a = np.array([base[s] for s in subs])
    R = Resampler(len(subs), 20260912)

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

    # The contrast the paper's headline rests on.
    if {"EA", "SUP"} <= set(table):
        ea = seed_mean_accuracy(table, "EA")
        sup = seed_mean_accuracy(table, "SUP")
        e = np.array([ea[s] for s in subs])
        u = np.array([sup[s] for s in subs])
        d = paired_delta(e, u, R)
        out["SUP-EA"] = (100 * d["point"], 100 * d["lo"], 100 * d["hi"],
                         permutation_p(e, u, R))
    return float(a.mean()), out, len(common)


def main():
    print("BCI IV 2a: random versus session-aware division of the held-out")
    print("subject. Budget is 288/288 either way; only the partition moves.\n")

    acc_r, eff_r, n_r = effects(RANDOM_GLOB)
    acc_s, eff_s, n_s = effects(SESSION_GLOB)
    if acc_s is None:
        if "__incomplete__" in eff_s:
            print("session runs incomplete, not comparable:",
                  eff_s["__incomplete__"])
        else:
            print("session runs not present yet")
        return 0

    print("source-only accuracy: random %.4f (%d runs) | session %.4f (%d runs)"
          % (acc_r, n_r, acc_s, n_s))
    print("  %-7s %24s %24s" % ("", "random split", "session split"))
    for arm in ARMS + ("SUP-EA",):
        if arm not in eff_r or arm not in eff_s:
            continue
        dr, ds = eff_r[arm], eff_s[arm]
        print("  %-7s %+7.2f [%+6.2f,%+6.2f]  %+7.2f [%+6.2f,%+6.2f]  %s"
              % (arm, dr[0], dr[1], dr[2], ds[0], ds[1], ds[2],
                 "both resolve" if dr[3] < 0.05 and ds[3] < 0.05 else
                 "random only" if dr[3] < 0.05 else
                 "session only" if ds[3] < 0.05 else "neither"))
    print()
    print("An effect that shrinks or stops resolving under the chronological")
    print("split was partly an artefact of interleaving calibration and test")
    print("trials in time, and the corresponding claim must be scoped to the")
    print("offline setting it was measured in.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
