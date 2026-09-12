"""Assert that the schema-v4 experimental controls actually hold on real data.

The v3 conditions were not matched.  The held-out-subject selection arm trained
on 25% more source trials than the clean arm, the target-supervised arm re-split
source train/validation after mixing target trials into the pool, and the
aligned arm fitted its whitener on trials that included the reserved test half.
Each of those is a difference between arms that the paper would otherwise have
attributed to the factor named in the arm's label.

This script replays the fold construction for all four v4 conditions on a real
dataset -- no training -- and fails loudly if any control is violated.  It is
meant to run before the experiments, not after.

Usage:
    python experiments/verify_controls.py --dataset BNCI2014_001 \
        --data-source official-gdf --official-gdf-root <path>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import LabelEncoder  # noqa: E402

from experiments import paper_runner as pr  # noqa: E402
from experiments.schema import target_access_mode  # noqa: E402


CONDITIONS = [
    ("SRC", False, False),
    ("EA", True, False),
    ("SUP", False, True),
    ("EA+SUP", True, True),
]


class ControlViolation(AssertionError):
    """Raised when a condition differs from the others in more than its factor."""


def _args_for(base, align, supervised):
    ns = argparse.Namespace(**vars(base))
    ns.euclidean_alignment = align
    ns.target_supervised = supervised
    return ns


def collect(base_args, n_folds):
    """Replay every condition's fold construction and record what it received."""

    config = pr.make_config(base_args)
    data_source = pr.build_data_source(config, base_args)
    subjects = list(data_source.subject_list)
    if getattr(base_args, "max_subjects", None):
        subjects = subjects[: base_args.max_subjects]

    seed = base_args.seed
    weight = float(base_args.target_sample_weight)
    cap = getattr(base_args, "available_trials", None)
    split = getattr(base_args, "target_split", "random") or "random"
    observed = {}

    # Raw per-subject data, used to check the alignment reference scope and to
    # recompute the reserved-test split independently of the runner.
    raw = pr._subject_blocks(data_source, subjects)

    for name, align, supervised in CONDITIONS:
        folds = []
        it = pr._iter_evaluation_folds(
            data_source, subjects, seed, align,
            available_trials=cap, target_split=split,
            dataset=base_args.dataset)
        for _ in range(n_folds):
            subj, X_outer, y_outer, X_avail, y_avail, X_te, y_te = next(it)
            le = LabelEncoder()
            y_outer_enc = le.fit_transform(y_outer)
            y_avail_enc = le.transform(y_avail)

            idx_src_tr, idx_src_val = pr._split_outer_train(y_outer_enc, seed)
            n_source_train = len(idx_src_tr)
            n_target_train = len(y_avail_enc) if supervised else 0
            w = pr._training_sample_weights(n_source_train, n_target_train, weight)
            realized = pr._measure_target_fraction(
                w, n_source_train, n_target_train, seed)

            # Independently recompute the reserved-test split from raw data.
            # This has to follow whichever partition the run declared: checking
            # a session-split run against a random split would report a
            # mismatch on every fold and say nothing about the controls.
            X_s, y_s = raw[subj]
            if split == "session":
                idx_avail_ref, idx_test_ref = pr._session_split(
                    len(y_s), base_args.dataset)
            else:
                idx_avail_ref, idx_test_ref = train_test_split(
                    np.arange(len(y_s)), test_size=pr.TARGET_TEST_FRACTION,
                    stratify=y_s, random_state=seed * 1000 + int(subj))
            if cap is not None and cap < len(idx_avail_ref):
                idx_avail_ref = pr._nested_available_subset(
                    idx_avail_ref, y_s[idx_avail_ref], int(cap),
                    seed * 1000 + int(subj))

            rec = {
                "subject": int(subj),
                "src_train_idx_hash": int(
                    np.frombuffer(idx_src_tr.astype("<i8").tobytes(),
                                  dtype="<i8").sum()),
                "src_train_idx": idx_src_tr,
                "src_val_idx": idx_src_val,
                "n_source_train": int(n_source_train),
                "n_source_val": int(len(idx_src_val)),
                "n_target_train": int(n_target_train),
                "n_target_available": int(len(y_avail_enc)),
                "n_test": int(len(y_te)),
                "samples_per_epoch": int(n_source_train),
                "realized_target_fraction": float(realized),
                "test_label_signature": np.asarray(y_te).tolist(),
                "reserved_test_idx": idx_test_ref.tolist(),
                "align": align,
                "supervised": supervised,
            }

            if align:
                # The whitener the runner used on the held-out subject must be
                # reproducible from the available half alone.
                W_avail = pr._ea_whitener(X_s[idx_avail_ref])
                expect_avail = pr._apply_whitener(W_avail, X_s[idx_avail_ref])
                expect_test = pr._apply_whitener(W_avail, X_s[idx_test_ref])
                rec["ea_avail_matches_available_only_reference"] = bool(
                    np.allclose(X_avail, expect_avail, atol=1e-4, rtol=1e-3))
                rec["ea_test_matches_available_only_reference"] = bool(
                    np.allclose(X_te, expect_test, atol=1e-4, rtol=1e-3))
                # And it must NOT match a reference that saw the whole subject.
                W_all = pr._ea_whitener(X_s)
                rec["ea_test_matches_full_subject_reference"] = bool(
                    np.allclose(X_te, pr._apply_whitener(W_all, X_s[idx_test_ref]),
                                atol=1e-4, rtol=1e-3))
            folds.append(rec)
        observed[name] = folds
    return observed, subjects


def check(observed, weight, tol=0.02):
    """Raise on the first control violation; return the audit table otherwise."""

    problems = []
    ref_name = CONDITIONS[0][0]
    ref = observed[ref_name]

    for name, _, supervised in CONDITIONS[1:]:
        got = observed[name]
        for a, b in zip(ref, got):
            if a["subject"] != b["subject"]:
                problems.append(
                    f"{name}: fold order differs from {ref_name} "
                    f"({a['subject']} vs {b['subject']})")
                continue
            s = a["subject"]
            if not np.array_equal(a["src_train_idx"], b["src_train_idx"]):
                problems.append(
                    f"{name} subj {s}: source training indices differ from {ref_name}")
            if not np.array_equal(a["src_val_idx"], b["src_val_idx"]):
                problems.append(
                    f"{name} subj {s}: source validation indices differ from {ref_name}")
            if a["n_source_train"] != b["n_source_train"]:
                problems.append(
                    f"{name} subj {s}: source training size "
                    f"{b['n_source_train']} != {a['n_source_train']}")
            if a["samples_per_epoch"] != b["samples_per_epoch"]:
                problems.append(
                    f"{name} subj {s}: samples drawn per epoch "
                    f"{b['samples_per_epoch']} != {a['samples_per_epoch']}")
            if a["reserved_test_idx"] != b["reserved_test_idx"]:
                problems.append(
                    f"{name} subj {s}: reserved test trials differ from {ref_name}")

    for name, align, supervised in CONDITIONS:
        for rec in observed[name]:
            s = rec["subject"]
            expected = weight if supervised else 0.0
            if abs(rec["realized_target_fraction"] - expected) > tol:
                problems.append(
                    f"{name} subj {s}: realized target fraction "
                    f"{rec['realized_target_fraction']:.4f} is not within {tol} "
                    f"of the declared {expected:.4f}")
            if not supervised and rec["n_target_train"] != 0:
                problems.append(
                    f"{name} subj {s}: {rec['n_target_train']} target trials "
                    "entered training in a source-only condition")
            if align:
                if not rec["ea_avail_matches_available_only_reference"]:
                    problems.append(
                        f"{name} subj {s}: aligned available trials do not match "
                        "an available-only reference")
                if not rec["ea_test_matches_available_only_reference"]:
                    problems.append(
                        f"{name} subj {s}: aligned test trials do not match the "
                        "frozen available-only reference")
                if rec["ea_test_matches_full_subject_reference"]:
                    problems.append(
                        f"{name} subj {s}: aligned test trials still match a "
                        "reference fitted on the whole subject")

    if problems:
        raise ControlViolation(
            "%d control violation(s):\n  - %s"
            % (len(problems), "\n  - ".join(problems)))
    return True


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="BNCI2014_001")
    p.add_argument("--data-source", dest="data_source",
                   choices=("moabb", "official-gdf"), default="moabb")
    p.add_argument("--official-gdf-root", dest="official_gdf_root", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--folds", type=int, default=3,
                   help="Number of held-out subjects to replay (default 3).")
    p.add_argument("--max_subjects", type=int, default=None)
    p.add_argument("--target_sample_weight", type=float,
                   default=pr.TARGET_SAMPLE_WEIGHT)
    p.add_argument("--available_trials", type=int, default=None,
                   help=("Audit the run under a capped calibration budget, as "
                         "--available_trials does for the runner."))
    p.add_argument("--target_split", default="random",
                   choices=["random", "session"],
                   help="Audit the chronological partition instead of the random one.")
    p.add_argument("--out", default=None, help="Optional JSON audit output path.")
    args = p.parse_args(argv)

    # Fields make_config/build_data_source expect but that this script fixes.
    args.variant = "eegnet"
    args.weighting = "equal"
    args.epochs = 50
    args.batch_size = 64
    args.lr = 1e-3
    args.wd = 1e-4
    args.patience = 10
    args.lr_schedule = "step"
    args.lr_step_size = 15
    args.lr_gamma = 0.5
    args.save_dir = str(ROOT / 'results' / '_verify_controls_unused')

    observed, subjects = collect(args, args.folds)
    check(observed, args.target_sample_weight)

    print("=" * 72)
    print(f"CONTROL AUDIT  dataset={args.dataset}  seed={args.seed}  "
          f"subjects={len(subjects)}  folds replayed={args.folds}")
    print("=" * 72)
    header = ("condition", "access", "src_train", "src_val", "tgt_train",
              "per_epoch", "tgt_frac", "test")
    print("%-8s %-18s %9s %8s %9s %9s %8s %6s" % header)
    for name, align, supervised in CONDITIONS:
        r = observed[name][0]
        print("%-8s %-18s %9d %8d %9d %9d %8.4f %6d" % (
            name, target_access_mode(align, supervised),
            r["n_source_train"], r["n_source_val"], r["n_target_train"],
            r["samples_per_epoch"], r["realized_target_fraction"], r["n_test"]))
    print()
    print("PASS: source partition, optimizer budget, reserved test trials and "
          "target sampling rate are identical or as declared across all four "
          "conditions.")

    if args.out:
        serial = {
            name: [{k: v for k, v in rec.items()
                    if k not in ("src_train_idx", "src_val_idx")}
                   for rec in folds]
            for name, folds in observed.items()
        }
        Path(args.out).write_text(json.dumps(serial, indent=1), encoding="utf-8")
        print("audit written:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
