"""Statistics for the schema-v4 target-access experiments.

Four analyses, all resampled over subjects because the subject is the unit the
cross-subject protocol is designed to generalize over:

  1. ``deltas``       per-condition paired change against the source-only
                      reference, with the standard-deviation ratio.
  2. ``interaction``  whether alignment and target supervision compose, i.e.
                      ``(EA+SUP - EA) - (SUP - SRC)`` per subject.
  3. ``pairwise``     the direct within-subject EA-vs-SUP difference.  This is
                      exploratory: it was prompted by the ordering seen in an
                      earlier summary, not specified in advance.
  4. ``regroup``      a fixed-prediction negative control.  The same test-trial
                      predictions are aggregated by true subject and by random
                      groups of the same sizes.  Nothing about the model
                      changes, so any difference in spread is attributable to
                      the aggregation unit alone.  Run on the source-only and
                      target-supervised conditions.

Two properties the comparisons depend on:

  * every condition within a dataset is resampled with the *same* bootstrap and
    sign-flip indices, so the arms are compared on identical resamples rather
    than on independent draws that happen to share a seed;
  * the spread statistic is the *standard deviation* across subjects, and a
    "ratio" is a ratio of two such standard deviations -- never a variance
    ratio, whose value would be its square.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

N_BOOT = 10000
N_REGROUP = 2000
CONDITION_ORDER = ["SRC", "EA", "SUP", "EASUP", "SEL", "POOL"]
CONDITION_LABEL = {
    "SRC": "source-only",
    "EA": "alignment",
    "SUP": "target-supervised",
    "EASUP": "alignment + target-supervised",
    "SEL": "target-subject selection",
    "POOL": "target-inclusive scaler",
}


# ─── shared resampling plan ───────────────────────────────────────────────────

class Resampler:
    """One bootstrap and sign-flip plan, shared by every condition in a dataset.

    Drawing fresh indices per condition would compare arms on different
    resamples, which makes two intervals not directly comparable even when they
    are both correct marginally.  Building the plan once from the subject count
    removes that.
    """

    def __init__(self, n_subjects, seed, n_boot=N_BOOT):
        rng = np.random.default_rng(seed)
        self.n = int(n_subjects)
        self.n_boot = int(n_boot)
        self.boot = rng.integers(0, self.n, size=(self.n_boot, self.n))
        self.flips = rng.choice([-1.0, 1.0], size=(self.n_boot, self.n))
        self.rng = rng

    def check(self, *arrays):
        for a in arrays:
            if len(a) != self.n:
                raise ValueError(
                    "resampler built for %d subjects, got %d" % (self.n, len(a)))


def paired_delta(a, b, R):
    """Mean of ``b - a`` over subjects, with a percentile bootstrap interval.

    ``n_improved`` is reported alongside because a sign-flip test on a small
    cohort saturates: with 9 subjects the smallest attainable two-sided
    $p$ is $2/2^9 = 0.0039$, so every arm that helps all nine reports the same
    value regardless of how large its effect is.  The count says which arms are
    at that floor.
    """

    a, b = np.asarray(a, float), np.asarray(b, float)
    R.check(a, b)
    d = b - a
    draws = d[R.boot].mean(axis=1)
    return dict(point=float(d.mean()),
                lo=float(np.percentile(draws, 2.5)),
                hi=float(np.percentile(draws, 97.5)),
                n=int(len(d)),
                n_improved=int((d > 0).sum()),
                min_attainable_p=float(2.0 / (2.0 ** len(d))) if len(d) <= 30
                else 0.0)


def sd_ratio(a, b, R):
    """Ratio of between-subject standard deviations, ``sd(b) / sd(a)``.

    Subjects are resampled jointly, on the shared indices, so the interval
    reflects the paired design.
    """

    a, b = np.asarray(a, float), np.asarray(b, float)
    R.check(a, b)
    sa = a[R.boot].std(axis=1, ddof=1)
    sb = b[R.boot].std(axis=1, ddof=1)
    ok = sa > 0
    draws = sb[ok] / sa[ok]
    return dict(point=float(b.std(ddof=1) / a.std(ddof=1)),
                lo=float(np.percentile(draws, 2.5)),
                hi=float(np.percentile(draws, 97.5)),
                sd_ref=float(a.std(ddof=1)), sd_arm=float(b.std(ddof=1)),
                statistic="ratio_of_standard_deviations")


def permutation_p(a, b, R):
    """Two-sided paired sign-flip test on the within-subject differences."""

    a, b = np.asarray(a, float), np.asarray(b, float)
    R.check(a, b)
    d = b - a
    obs = abs(d.mean())
    null = np.abs((R.flips * d).mean(axis=1))
    return float((np.sum(null >= obs) + 1) / (R.n_boot + 1))


def paired_t(a, b):
    """Paired t-test, reported as a sensitivity check beside the permutation p."""

    d = np.asarray(b, float) - np.asarray(a, float)
    if len(d) < 2:
        return dict(t=float("nan"), p=float("nan"), df=0)
    sd = d.std(ddof=1)
    se = sd / np.sqrt(len(d))
    t = float(d.mean() / se) if se else float("nan")
    try:
        from scipy import stats
        p = float(2 * stats.t.sf(abs(t), df=len(d) - 1))
    except Exception:
        p = float("nan")
    return dict(t=t, p=p, df=int(len(d) - 1))


def holm(pvalues, labels):
    """Holm-Bonferroni adjusted p-values for a family of tests."""

    order = np.argsort(pvalues)
    m = len(pvalues)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvalues[i])
        adjusted[i] = min(1.0, running)
    return {labels[i]: float(adjusted[i]) for i in range(m)}


# ─── loading ──────────────────────────────────────────────────────────────────

def condition_of(log):
    """Recover the condition from a log's protocol block."""

    p = log["protocol"]
    if p.get("normalization_mode") == "pooled_all_subject_scaler":
        return "POOL"
    if p.get("test_subject_used_for_model_selection"):
        return "SEL"
    align = bool(p.get("euclidean_alignment"))
    sup = bool(p.get("target_supervised"))
    return {(False, False): "SRC", (True, False): "EA",
            (False, True): "SUP", (True, True): "EASUP"}[(align, sup)]


def load(pattern):
    """Return {condition: {seed: {subject: record}}} plus the log list."""

    logs = []
    for path in sorted(glob.glob(pattern)):
        if os.path.basename(path).startswith("_"):
            continue
        with open(path, encoding="utf-8") as fh:
            log = json.load(fh)
        if log.get("schema_version") != 4 or log.get("status") != "completed":
            continue
        logs.append(log)

    table = defaultdict(lambda: defaultdict(dict))
    for log in logs:
        cond = condition_of(log)
        for subj, rec in log["per_subject"].items():
            table[cond][log["seed"]][subj] = rec
    return table, logs


def seed_mean_accuracy(table, cond):
    """Average each subject's accuracy over runs, so one subject is one point."""

    per_subject = defaultdict(list)
    for _seed, subjects in table[cond].items():
        for subj, rec in subjects.items():
            per_subject[subj].append(rec["main_accuracy"])
    return {s: float(np.mean(v)) for s, v in per_subject.items()}


def per_seed_accuracy(table, cond):
    """{seed: {subject: accuracy}}, for the per-run sensitivity report."""

    return {seed: {s: float(r["main_accuracy"]) for s, r in subjects.items()}
            for seed, subjects in table[cond].items()}


def control_summary(logs):
    """What the fixed controls actually delivered, read back from the logs."""

    out = defaultdict(list)
    for log in logs:
        cond = condition_of(log)
        for rec in log["per_subject"].values():
            c = rec.get("controls")
            if not c:
                continue
            out[cond].append(c)
    summary = {}
    for cond, rows in out.items():
        summary[cond] = {
            key: float(np.mean([r[key] for r in rows]))
            for key in ("n_source_train", "n_source_val", "n_target_train",
                        "n_target_available", "n_test", "steps_per_epoch",
                        "epochs_run", "total_update_steps",
                        "declared_target_sample_weight",
                        "realized_target_fraction")
            if key in rows[0]
        }
    return summary


# ─── analyses ─────────────────────────────────────────────────────────────────

def analyse_deltas(table, R, subjects):
    """Per-condition paired change and SD ratio against the source-only arm."""

    base = seed_mean_accuracy(table, "SRC")
    a = np.array([base[s] for s in subjects])

    rows, pvals, labels = [], [], []
    for cond in CONDITION_ORDER[1:]:
        if cond not in table:
            continue
        arm = seed_mean_accuracy(table, cond)
        if set(arm) != set(base):
            continue
        b = np.array([arm[s] for s in subjects])

        per_seed = []
        seeds_ref = per_seed_accuracy(table, "SRC")
        seeds_arm = per_seed_accuracy(table, cond)
        for seed in sorted(set(seeds_ref) & set(seeds_arm)):
            shared = sorted(set(seeds_ref[seed]) & set(seeds_arm[seed]),
                            key=lambda s: int(s))
            if not shared:
                continue
            d = np.array([seeds_arm[seed][s] - seeds_ref[seed][s] for s in shared])
            per_seed.append(dict(seed=int(seed), delta=float(d.mean()),
                                 n=len(shared)))

        p = permutation_p(a, b, R)
        pvals.append(p)
        labels.append(cond)
        rows.append(dict(condition=cond, label=CONDITION_LABEL[cond],
                         mean_ref=float(a.mean()), mean_arm=float(b.mean()),
                         sd_arm=float(b.std(ddof=1)),
                         delta=paired_delta(a, b, R),
                         sd_ratio=sd_ratio(a, b, R),
                         p_raw=p, paired_t=paired_t(a, b),
                         per_seed=per_seed))
    adj = holm(pvals, labels) if pvals else {}
    for row in rows:
        row["p_holm"] = adj.get(row["condition"])
    return dict(reference_mean=float(a.mean()),
                reference_sd=float(a.std(ddof=1)),
                n_subjects=len(subjects), rows=rows)


def analyse_interaction(table, R, subjects):
    """Does target supervision buy the same thing with and without alignment?"""

    needed = ("SRC", "EA", "SUP", "EASUP")
    if any(c not in table for c in needed):
        return None
    arms = {c: seed_mean_accuracy(table, c) for c in needed}
    if any(set(arms[c]) != set(arms["SRC"]) for c in needed):
        return None
    v = {c: np.array([arms[c][s] for s in subjects]) for c in needed}

    inter = (v["EASUP"] - v["EA"]) - (v["SUP"] - v["SRC"])
    draws = inter[R.boot].mean(axis=1)
    zero = np.zeros_like(inter)

    # Per-run interaction.  The sign of this quantity is the paper's most
    # fragile claim, so it is reported run by run: a sign that flips between
    # runs of the same dataset is noise, not a property of the dataset.
    seeds = {c: per_seed_accuracy(table, c) for c in needed}
    per_seed = []
    for seed in sorted(set.intersection(*(set(seeds[c]) for c in needed))):
        shared = sorted(set.intersection(*(set(seeds[c][seed]) for c in needed)),
                        key=lambda s: int(s))
        if not shared:
            continue
        g = {c: np.array([seeds[c][seed][s] for s in shared]) for c in needed}
        d = (g["EASUP"] - g["EA"]) - (g["SUP"] - g["SRC"])
        per_seed.append(dict(seed=int(seed), interaction=float(d.mean()),
                             n=len(shared)))

    return dict(
        n_subjects=len(subjects),
        per_seed=per_seed,
        n_positive=int((inter > 0).sum()),
        simple_effect_no_alignment=paired_delta(v["SRC"], v["SUP"], R),
        simple_effect_with_alignment=paired_delta(v["EA"], v["EASUP"], R),
        interaction=dict(point=float(inter.mean()),
                         lo=float(np.percentile(draws, 2.5)),
                         hi=float(np.percentile(draws, 97.5))),
        interaction_p=permutation_p(zero, inter, R),
        interaction_paired_t=paired_t(zero, inter),
    )


def analyse_pairwise(table, R, subjects, pairs=(("EA", "SUP"),)):
    """Within-subject differences between two arms, computed directly.

    Exploratory: prompted by an ordering seen in an earlier summary rather than
    specified in advance, and reported as such.
    """

    out = []
    for left, right in pairs:
        if left not in table or right not in table:
            continue
        la, ra = seed_mean_accuracy(table, left), seed_mean_accuracy(table, right)
        if set(la) != set(ra):
            continue
        a = np.array([la[s] for s in subjects])
        b = np.array([ra[s] for s in subjects])

        per_seed = []
        sl, sr = per_seed_accuracy(table, left), per_seed_accuracy(table, right)
        for seed in sorted(set(sl) & set(sr)):
            shared = sorted(set(sl[seed]) & set(sr[seed]), key=lambda s: int(s))
            if not shared:
                continue
            d = np.array([sr[seed][s] - sl[seed][s] for s in shared])
            per_seed.append(dict(seed=int(seed), delta=float(d.mean()),
                                 n=len(shared)))

        out.append(dict(left=left, right=right, exploratory=True,
                        mean_left=float(a.mean()), mean_right=float(b.mean()),
                        delta=paired_delta(a, b, R),
                        sd_ratio=sd_ratio(a, b, R),
                        p_raw=permutation_p(a, b, R),
                        paired_t=paired_t(a, b),
                        per_seed=per_seed))
    return out


def analyse_regrouping(table, rng, conditions=("SRC", "SUP"),
                       n_repeat=N_REGROUP):
    """Fixed-prediction negative control.

    The predictions are held fixed and only the grouping changes: once by true
    subject, then repeatedly into random groups of the same sizes.  The
    trial-weighted accuracy is identical either way, so a spread that survives
    only under the true grouping is a property of the subjects, and one that
    appears under both is a property of how many trials each group holds.
    """

    out = {}
    for cond in conditions:
        if cond not in table:
            continue
        per_seed = []
        for seed in sorted(table[cond]):
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
            true_sd = float(np.std([c.mean() for c in correct], ddof=1))

            bounds = np.cumsum(sizes)[:-1]
            sds = np.empty(n_repeat)
            n = len(flat)
            for r in range(n_repeat):
                shuffled = flat[rng.permutation(n)]
                sds[r] = np.std([g.mean() for g in np.split(shuffled, bounds)],
                                ddof=1)
            # What the null *should* be if the only thing driving between-group
            # spread were how many trials each group holds: drawing m of M
            # correctness records without replacement gives a group-mean
            # variance of (1 - m/M) S^2 / m.  This is textbook finite-population
            # sampling, quoted to show the control behaves as designed rather
            # than as an EEG finding.
            m, M = float(sizes[0]), float(len(flat))
            s2 = float(flat.var(ddof=1))
            predicted = float(np.sqrt((1.0 - m / M) * s2 / m))

            per_seed.append(dict(
                seed=int(seed), n_groups=len(sizes), group_sizes=sizes,
                equal_group_sizes=bool(len(set(sizes)) == 1),
                pooled_accuracy=float(flat.mean()),
                sd_predicted_by_sampling=predicted,
                sd_by_true_subject=true_sd,
                sd_by_random_groups_mean=float(sds.mean()),
                sd_by_random_groups_lo=float(np.percentile(sds, 2.5)),
                sd_by_random_groups_hi=float(np.percentile(sds, 97.5)),
                inflation_factor=float(true_sd / sds.mean()) if sds.mean() else None,
                p_exceeds_random=float((np.sum(sds >= true_sd) + 1) / (n_repeat + 1)),
                n_repeat=int(n_repeat),
            ))
        if per_seed:
            out[cond] = per_seed
    return out


# ─── reporting ────────────────────────────────────────────────────────────────

def pct(x):
    return 100.0 * x


def report(name, table, logs, seed=20260909):
    if "SRC" not in table:
        print("[skip] %s: no source-only reference" % name)
        return None
    subjects = sorted(seed_mean_accuracy(table, "SRC"), key=lambda s: int(s))
    R = Resampler(len(subjects), seed)

    res = dict(dataset=name, n_subjects=len(subjects),
               subjects=subjects,
               conditions=sorted(table, key=lambda c: CONDITION_ORDER.index(c)),
               controls=control_summary(logs))
    res["deltas"] = analyse_deltas(table, R, subjects)
    res["interaction"] = analyse_interaction(table, R, subjects)
    res["pairwise"] = analyse_pairwise(table, R, subjects)
    res["regrouping"] = analyse_regrouping(
        table, np.random.default_rng(seed + 1))

    d = res["deltas"]
    print("=" * 78)
    print("%s   n=%d subjects   source-only mean=%.4f  sd=%.4f" % (
        name, d["n_subjects"], d["reference_mean"], d["reference_sd"]))
    print("=" * 78)
    print("%-7s %-30s %21s %16s" % ("cond", "label", "delta pp [95% CI]",
                                    "SD ratio"))
    for row in d["rows"]:
        dl, sr = row["delta"], row["sd_ratio"]
        print("%-7s %-30s %+6.2f [%+6.2f,%+6.2f] %5.2f [%.2f,%.2f]  "
              "p=%.4f (Holm %.4f, t p=%.4f)" % (
                  row["condition"], row["label"],
                  pct(dl["point"]), pct(dl["lo"]), pct(dl["hi"]),
                  sr["point"], sr["lo"], sr["hi"],
                  row["p_raw"], row["p_holm"], row["paired_t"]["p"]))
        print("          per run: %s   improved %d/%d subjects" % (
            "  ".join("s%d %+.2f" % (r["seed"], pct(r["delta"]))
                      for r in row["per_seed"]),
            dl["n_improved"], dl["n"]))

    it = res["interaction"]
    if it:
        print()
        print("interaction  (EA+SUP - EA) - (SUP - SRC)")
        print("  supervision without alignment: %+6.2f [%+6.2f,%+6.2f] pp" % (
            pct(it["simple_effect_no_alignment"]["point"]),
            pct(it["simple_effect_no_alignment"]["lo"]),
            pct(it["simple_effect_no_alignment"]["hi"])))
        print("  supervision with alignment   : %+6.2f [%+6.2f,%+6.2f] pp" % (
            pct(it["simple_effect_with_alignment"]["point"]),
            pct(it["simple_effect_with_alignment"]["lo"]),
            pct(it["simple_effect_with_alignment"]["hi"])))
        print("  interaction                  : %+6.2f [%+6.2f,%+6.2f] pp  "
              "p=%.4f" % (
                  pct(it["interaction"]["point"]), pct(it["interaction"]["lo"]),
                  pct(it["interaction"]["hi"]), it["interaction_p"]))
        print("    per run: %s   positive for %d/%d subjects" % (
            "  ".join("s%d %+.2f" % (r["seed"], pct(r["interaction"]))
                      for r in it.get("per_seed", [])),
            it.get("n_positive", 0), it["n_subjects"]))

    for pw in res["pairwise"]:
        print()
        print("direct %s - %s (exploratory): %+6.2f [%+6.2f,%+6.2f] pp  "
              "p=%.4f  SD ratio %.2f [%.2f,%.2f]" % (
                  pw["right"], pw["left"],
                  pct(pw["delta"]["point"]), pct(pw["delta"]["lo"]),
                  pct(pw["delta"]["hi"]), pw["p_raw"],
                  pw["sd_ratio"]["point"], pw["sd_ratio"]["lo"],
                  pw["sd_ratio"]["hi"]))
        print("          per run: %s" % "  ".join(
            "s%d %+.2f" % (r["seed"], pct(r["delta"])) for r in pw["per_seed"]))

    if res["regrouping"]:
        print()
        print("regrouping control (predictions fixed, only the grouping changes)")
        for cond, seeds in res["regrouping"].items():
            for s in seeds:
                print("  %-6s run %d: SD by true subject %.4f | by random "
                      "groups %.4f [%.4f,%.4f] (sampling theory %.4f) | "
                      "x%.2f | p=%.4f" % (
                          cond, s["seed"], s["sd_by_true_subject"],
                          s["sd_by_random_groups_mean"],
                          s["sd_by_random_groups_lo"],
                          s["sd_by_random_groups_hi"],
                          s["sd_predicted_by_sampling"],
                          s["inflation_factor"], s["p_exceeds_random"]))
    print()
    return res


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        print("usage: v4_stats.py <name>=<glob> [...] [--out results.json]")
        return 2
    out_path, specs, i = None, [], 1
    while i < len(argv):
        if argv[i] == "--out":
            out_path = argv[i + 1]
            i += 2
            continue
        specs.append(argv[i])
        i += 1

    results = []
    for spec in specs:
        name, _, pattern = spec.partition("=")
        table, logs = load(pattern)
        if not logs:
            print("[skip] %s: no completed schema-v4 logs at %s" % (name, pattern))
            continue
        print("[%s] %d log(s), conditions: %s" % (
            name, len(logs),
            sorted(table, key=lambda c: CONDITION_ORDER.index(c))))
        got = report(name, table, logs)
        if got:
            results.append(got)

    if out_path and results:
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=1)
        print("written:", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
