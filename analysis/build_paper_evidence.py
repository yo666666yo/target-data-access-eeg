#!/usr/bin/env python3
"""Rebuild every paper artifact from the schema-v4 run logs, in one command.

    python analysis/build_paper_evidence.py [--results-root results]

Runs, in order:
  1. the statistics bundle          -> results/v4_stats.json
  2. the LaTeX tables               -> manuscript/tables/*.tex
  3. the effects figure             -> figures/fig_v4_effects.pdf
  4. the regrouping-control figure  -> figures/fig_regroup.pdf

Nothing here recomputes a model; every number comes from a run log, so the
whole thing is cheap and safe to re-run after any change to the analysis.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATASETS = [
    ("BCI IV 2a", "v4_2a"),
    ("BNCI2014-002", "v4_002"),
    ("PhysionetMI", "v4_physionet"),
]


def run(argv, label):
    print("\n" + "=" * 72)
    print("[%s] %s" % (label, " ".join(str(a) for a in argv)))
    print("=" * 72)
    result = subprocess.run([sys.executable, *[str(a) for a in argv]], cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit("[%s] failed with exit code %d" % (label, result.returncode))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", default="results")
    args = ap.parse_args()

    root = Path(args.results_root)
    present = [(name, root / d / "*.json") for name, d in DATASETS
               if (ROOT / root / d).is_dir()]
    if not present:
        raise SystemExit("no schema-v4 result directories under %s" % root)
    missing = [d for _, d in DATASETS if not (ROOT / root / d).is_dir()]
    if missing:
        print("[warn] proceeding without: %s" % ", ".join(missing))

    stats_path = root / "v4_stats.json"
    run(["analysis/v4_stats.py",
         *["%s=%s" % (name, pattern.as_posix()) for name, pattern in present],
         "--out", stats_path.as_posix()], "stats")
    run(["figures/gen_v4_tables.py", stats_path.as_posix()], "tables")
    run(["figures/gen_fig_v4_effects.py", stats_path.as_posix()], "effects figure")
    run(["figures/gen_fig_regroup.py",
         *["%s=%s" % (name, pattern.as_posix()) for name, pattern in present]],
        "regrouping figure")

    print("\nall artifacts rebuilt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
