#!/usr/bin/env python3
"""Emit the paper's LaTeX tables from the schema-v4 statistics bundle.

Reads the JSON written by ``analysis/v4_stats.py --out`` and writes
``manuscript/tables/*.tex``.  Numbers are never typed into the paper by
hand; every cell here traces to a per-subject accuracy in a run log.

The main table carries the reference arm's absolute accuracy once, in its block
header, and gives the other conditions as effects with intervals.  An absolute
accuracy never appears in a cell headed with a delta, and the standard
deviation, the interval and the ratio are kept in separate columns so they
cannot be read as the same quantity.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "manuscript" / "tables"

BS = chr(92)                      # backslash, kept out of string literals
MAIN = ["EA", "SUP", "EASUP"]
# The table carries the 2x2 only.  POOL is null everywhere and SEL now supports
# a single one-directional claim; both are quoted in the text, which costs far
# less space than six more rows.
SECONDARY = []
SHORT = {
    "SRC": BS + "textsc{src}",
    "EA": BS + "textsc{ea}",
    "SUP": BS + "textsc{sup}",
    "EASUP": BS + "textsc{ea+sup}",
    "SEL": BS + "textsc{sel}",
    "POOL": BS + "textsc{pool}",
}
PROCEDURE = {
    "EA": "align",
    "SUP": "supervise",
    "EASUP": "align + supervise",
    "SEL": "pick checkpoint",
    "POOL": "fit scaler",
}


def pp(x):
    return 100.0 * x


def fmt_delta(d):
    return "$%+.2f$ [$%+.2f$, $%+.2f$]" % (pp(d["point"]), pp(d["lo"]), pp(d["hi"]))


def fmt_ratio(r):
    return "$%.2f$ [$%.2f$, $%.2f$]" % (r["point"], r["lo"], r["hi"])


def fmt_runs(per_seed):
    if not per_seed:
        return "---"
    return " ".join("$%+.1f$" % pp(r["delta"]) for r in per_seed)


def fmt_p(p):
    if p is None:
        return "---"
    if p < 1e-3:
        return "$<$0.001"
    return "%.3f" % p


def row_lookup(res, cond):
    for row in res["deltas"]["rows"]:
        if row["condition"] == cond:
            return row
    return None


def table_main(results):
    """Main results: one block per dataset, one row per condition."""

    lines = [
        BS + "begin{table*}[t]",
        BS + "centering",
        BS + "caption{Effect of each procedure against the source-only "
        "reference under the controls of Section~" + BS + "ref{sec:controls}. "
        "Differences are paired within subject and averaged over three runs; "
        "the per-run column gives the same difference within each run. "
        "Intervals are descriptive percentile bootstrap over subjects, not "
        "multiplicity-adjusted; $p$ is Holm-corrected within each dataset and "
        "is followed by the number of subjects helped, which distinguishes arms "
        "at the permutation floor. SD is the between-subject standard deviation "
        "of run-averaged per-subject accuracy---a different pooling order from "
        "Section~" + BS + "ref{sec:results}, which pools variances within runs. "
        + SHORT["SEL"] + " and " + SHORT["POOL"] + " sit off the grid and are "
        "quoted in the text.}",
        BS + "label{tab:main}",
        BS + "small",
        BS + "begin{tabular}{llccccc}",
        BS + "toprule",
        "Condition & Uses $C_s$ to & $" + BS + "Delta$ accuracy (pp) [95" + BS
        + "% CI] & per run (pp) & subject SD & SD ratio [95" + BS + "% CI] & "
        "$p$ (improved) " + BS + BS,
        BS + "midrule",
    ]

    for res in results:
        d = res["deltas"]
        lines.append(
            BS + "multicolumn{7}{l}{" + BS + "textit{%s} "
            "($n=%d$ subjects; source-only accuracy $%.3f$, "
            "between-subject SD $%.3f$)}" % (
                res["dataset"], d["n_subjects"], d["reference_mean"],
                d["reference_sd"]) + " " + BS + BS)
        for cond in MAIN + SECONDARY:
            row = row_lookup(res, cond)
            if row is None:
                continue
            if SECONDARY and cond == SECONDARY[0]:
                lines.append(BS + "addlinespace[2pt]")
            d = row["delta"]
            lines.append(
                "%s & %s & %s & %s & $%.3f$ & %s & %s (%d/%d) %s" % (
                    SHORT[cond], PROCEDURE[cond],
                    fmt_delta(d), fmt_runs(row.get("per_seed", [])),
                    row.get("sd_arm", float("nan")),
                    fmt_ratio(row["sd_ratio"]),
                    fmt_p(row["p_holm"]),
                    d.get("n_improved", 0), d.get("n", 0), BS + BS))
        lines.append(BS + "midrule")

    lines[-1] = BS + "bottomrule"
    lines += [BS + "end{tabular}", BS + "end{table*}", ""]
    return "\n".join(lines)


def table_interaction(results):
    """Do alignment and target supervision compose?"""

    lines = [
        BS + "begin{table}[t]",
        BS + "centering",
        BS + "caption{Does supervising on $C_s$ buy the same thing with and "
        "without alignment? The interaction is the within-subject difference "
        "between supervision's effect given alignment and its effect without "
        "(the latter is the " + SHORT["SUP"] + " row of Table~" + BS
        + "ref{tab:main}). Intervals are the shared subject bootstrap. The "
        "per-run column recomputes the same quantity inside each run and is "
        "the measure of how firm the pooled estimate is.}",
        BS + "label{tab:interaction}",
        BS + "scriptsize",
        # Three columns only.  The simple effects are either already in
        # Table I (without alignment) or quoted in the text (with alignment):
        # carrying both here made the tabular overflow the column and print
        # over the neighbouring paragraph.
        BS + "begin{tabular}{lcc}",
        BS + "toprule",
        "Dataset & interaction (pp) [95" + BS + "% CI] & per run " + BS + BS,
        BS + "midrule",
    ]
    for res in results:
        it = res.get("interaction")
        if not it:
            continue
        runs = " ".join("$%+.1f$" % pp(r["interaction"])
                        for r in it.get("per_seed", [])) or "---"
        lines.append(
            "%s & $%+.2f$ [$%+.2f$, $%+.2f$] & %s %s" % (
                res["dataset"],
                pp(it["interaction"]["point"]), pp(it["interaction"]["lo"]),
                pp(it["interaction"]["hi"]), runs, BS + BS))
    lines += [BS + "bottomrule", BS + "end{tabular}", BS + "end{table}", ""]
    return "\n".join(lines)


def table_regrouping(results):
    """The fixed-prediction negative control.

    Deliberately narrow: this is a single-column float, and an earlier version
    carried a bootstrap interval in every cell, which pushed the last two
    columns out of the column and over the neighbouring text.
    """

    # The resampled null should equal the finite-population prediction
    # (1 - m/M) S^2 / m; quoting the worst disagreement in the caption is
    # cheaper than a sixth column and says the same thing.
    gaps = [abs(s["sd_by_random_groups_mean"] - s["sd_predicted_by_sampling"])
            for res in results
            for seeds in (res.get("regrouping") or {}).values()
            for s in seeds
            if "sd_predicted_by_sampling" in s]
    agreement = ("" if not gaps else
                 " The resampled null agrees with the finite-population "
                 "prediction $(1-m/M)S^2/m$ to within $%.4f$ everywhere."
                 % max(gaps))

    lines = [
        # [!b] rather than [t]: at the top of a column this float left ~8 lines
        # of stretched whitespace above the next section heading and pushed the
        # tail of the discussion onto a fifth page.
        BS + "begin{table}[!b]",
        BS + "centering",
        BS + "caption{Fixed-prediction regrouping control: identical predictions "
        "in both SD columns, only the grouping changes, and trial-weighted "
        "accuracy is unchanged. Random groups keep the true group sizes, "
        "resampled 2000 times. Spreads are within-run SDs averaged over runs "
        "(what regrouping one run yields); Table~" + BS + "ref{tab:main} reports "
        "the SD of run-averaged accuracy instead. \"var.\\ share\" is the "
        "fraction of between-subject \\emph{variance} the null accounts for; "
        "\"corrected\" is what remains." + agreement + "}",
        BS + "label{tab:regroup}",
        BS + "scriptsize",
        BS + "begin{tabular}{llcccc}",
        BS + "toprule",
        " & & " + BS + "multicolumn{2}{c}{SD of accuracy} & & " + BS + BS,
        BS + "cmidrule(lr){3-4}",
        "Dataset & Cond. & subject & random & var.\\ share & corrected "
        + BS + BS,
        BS + "midrule",
    ]
    for res in results:
        rg = res.get("regrouping") or {}
        first = True
        for cond in ("SRC", "SUP"):
            seeds = rg.get(cond)
            if not seeds:
                continue
            n = len(seeds)
            true_sd = sum(s["sd_by_true_subject"] for s in seeds) / n
            rand_sd = sum(s["sd_by_random_groups_mean"] for s in seeds) / n
            # Share of the observed between-subject VARIANCE attributable to
            # finite trials, and the SD that remains once it is removed.  The
            # SD ratio alone reads as a much larger share than it is.
            share = (rand_sd ** 2) / (true_sd ** 2) if true_sd else float("nan")
            corrected = max(true_sd ** 2 - rand_sd ** 2, 0.0) ** 0.5
            # The literal LaTeX "\%" is concatenated, never formatted -- a "%"
            # inside a format string is a conversion specifier.
            lines.append(
                "%s & %s & $%.3f$ & $%.3f$ & $%.0f$" % (
                    res["dataset"] if first else "", SHORT[cond], true_sd,
                    rand_sd, 100 * share)
                + BS + "%"
                + " & $%.3f$ " % corrected
                + BS + BS)
            first = False
    lines += [BS + "bottomrule", BS + "end{tabular}", BS + "end{table}", ""]
    return "\n".join(lines)


def scan(text, name):
    """Refuse to emit a file containing a control character or a stray command.

    Building LaTeX from Python strings has bitten this project before: an
    escape such as ``chr(92) + "textsc"`` turns into a tab, the compile emits no
    warning, and the command silently disappears from the PDF.  Everything here
    is assembled from ``chr(92)``, and this check confirms it stayed that way.

    A fragment such as ``extsc{`` is legitimate only as the tail of a complete
    control word, so the letters immediately before it must run back to a
    backslash -- which is what tells a real ``cmidrule`` apart from a
    ``midrule`` whose backslash was eaten.
    """

    bad = [(i, repr(c)) for i, c in enumerate(text)
           if ord(c) < 32 and c not in "\n"]
    if bad:
        raise SystemExit("%s: control characters at %s" % (name, bad[:5]))

    for frag in ("extsc{", "extbf{", "extit{", "egin{", "nd{",
                 "oprule", "idrule", "ottomrule", "ulticolumn{"):
        at = text.find(frag)
        while at != -1:
            back = at
            while back > 0 and text[back - 1].isalpha():
                back -= 1
            if back == 0 or text[back - 1] != BS:
                raise SystemExit(
                    "%s: orphaned command fragment %r at offset %d "
                    "(context %r)" % (name, frag, at,
                                      text[max(0, at - 14):at + 14]))
            at = text.find(frag, at + 1)
    return text


def main(argv):
    if len(argv) < 2:
        print("usage: gen_v4_tables.py <v4_stats.json>")
        return 2
    results = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    for name, text in (("table_main.tex", table_main(results)),
                       ("table_interaction.tex", table_interaction(results)),
                       ("table_regrouping.tex", table_regrouping(results))):
        (OUT / name).write_text(scan(text, name), encoding="utf-8")
        print("written:", OUT / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
