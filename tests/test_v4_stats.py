"""Calibration checks for the schema-v4 statistics.

The paper's claims are differences and ratios of standard deviations with
bootstrap intervals, so the estimators are checked against cases whose answers
are known in advance -- particularly the regrouping control, whose whole
purpose is to say when an observed spread is *not* a subject effect, and which
is therefore only useful if it can return a negative result.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.v4_stats import (  # noqa: E402
    Resampler,
    analyse_deltas,
    analyse_regrouping,
    holm,
    paired_delta,
    paired_t,
    permutation_p,
    sd_ratio,
    seed_mean_accuracy,
)

ACCS = np.array([0.40, 0.45, 0.50, 0.55, 0.60, 0.35, 0.48, 0.52, 0.44])


def fake_table(cond_accuracies, seeds=(0, 1, 2)):
    """Build the {condition: {seed: {subject: record}}} shape ``load`` returns."""

    return {
        cond: {
            seed: {str(i + 1): {"main_accuracy": float(acc)}
                   for i, acc in enumerate(per_subject)}
            for seed in seeds
        }
        for cond, per_subject in cond_accuracies.items()
    }


def fake_prediction_table(per_subject_accuracy, n_trials, seeds=(0,), rng=None,
                          cond="SRC"):
    """Per-trial predictions whose per-subject accuracy is exactly as specified."""

    rng = rng or np.random.default_rng(0)
    table = {cond: {}}
    for seed in seeds:
        subjects = {}
        for i, acc in enumerate(per_subject_accuracy):
            correct = np.zeros(n_trials, dtype=int)
            correct[:int(round(acc * n_trials))] = 1
            rng.shuffle(correct)
            subjects[str(i + 1)] = {
                "main_accuracy": float(correct.mean()),
                "test_labels": np.zeros(n_trials, dtype=int).tolist(),
                "test_predictions": (1 - correct).tolist(),
            }
        table[cond][seed] = subjects
    return table


class ResamplerTests(unittest.TestCase):

    def test_indices_are_shared_across_conditions(self):
        # The point of building the plan once: two arms compared against the
        # same reference must be evaluated on identical resamples.
        R = Resampler(9, seed=3)
        a = ACCS
        b, c = a + 0.05, a + 0.09
        first = paired_delta(a, b, R)
        second = paired_delta(a, c, R)
        # Both are constant shifts on the same indices, so the interval widths
        # match exactly and the point estimates differ by exactly 0.04.
        self.assertAlmostEqual(second["point"] - first["point"], 0.04, places=9)
        self.assertAlmostEqual(first["hi"] - first["lo"],
                               second["hi"] - second["lo"], places=9)

    def test_wrong_subject_count_is_rejected(self):
        R = Resampler(9, seed=3)
        with self.assertRaises(ValueError):
            paired_delta(ACCS[:5], ACCS[:5] + 0.01, R)


class EstimatorTests(unittest.TestCase):

    def setUp(self):
        self.R = Resampler(len(ACCS), seed=11)

    def test_paired_delta_recovers_a_known_constant_shift(self):
        got = paired_delta(ACCS, ACCS + 0.07, self.R)
        self.assertAlmostEqual(got["point"], 0.07, places=9)
        # A constant shift has zero within-subject variance, so the interval
        # collapses onto the point estimate.
        self.assertAlmostEqual(got["lo"], 0.07, places=9)
        self.assertAlmostEqual(got["hi"], 0.07, places=9)

    def test_sd_ratio_recovers_a_known_scaling(self):
        b = ACCS.mean() + 0.5 * (ACCS - ACCS.mean())
        got = sd_ratio(ACCS, b, self.R)
        self.assertAlmostEqual(got["point"], 0.5, places=9)
        self.assertEqual(got["statistic"], "ratio_of_standard_deviations")

    def test_sd_ratio_is_one_when_nothing_changes(self):
        self.assertAlmostEqual(sd_ratio(ACCS, ACCS.copy(), self.R)["point"],
                               1.0, places=9)

    def test_sd_ratio_is_not_a_variance_ratio(self):
        # Halving the spread gives 0.50 as a standard-deviation ratio and 0.25
        # as a variance ratio; the reported statistic must be the former.
        b = ACCS.mean() + 0.5 * (ACCS - ACCS.mean())
        self.assertNotAlmostEqual(sd_ratio(ACCS, b, self.R)["point"], 0.25,
                                  places=3)

    def test_permutation_p_is_small_for_a_consistent_shift(self):
        self.assertLess(permutation_p(ACCS, ACCS + 0.07, self.R), 0.01)

    def test_permutation_p_is_large_for_no_effect(self):
        wobble = np.array([+.01, -.01, +.01, -.01, +.01, -.01, +.01, -.01, 0.0])
        self.assertGreater(permutation_p(ACCS, ACCS + wobble, self.R), 0.20)

    def test_paired_t_agrees_with_the_permutation_test_in_direction(self):
        self.assertLess(paired_t(ACCS, ACCS + 0.07)["p"], 0.01)
        wobble = np.array([+.01, -.01, +.01, -.01, +.01, -.01, +.01, -.01, 0.0])
        self.assertGreater(paired_t(ACCS, ACCS + wobble)["p"], 0.20)

    def test_holm_is_monotone_and_never_below_the_raw_value(self):
        raw = [0.001, 0.02, 0.30, 0.9]
        ordered = [holm(raw, ["a", "b", "c", "d"])[k] for k in "abcd"]
        for r, adj in zip(raw, ordered):
            self.assertGreaterEqual(adj, r)
        self.assertEqual(ordered, sorted(ordered))


class RegroupingControlTests(unittest.TestCase):
    """The negative control must be able to return a negative result."""

    def setUp(self):
        self.rng = np.random.default_rng(5)

    def test_no_subject_effect_gives_a_ratio_near_one(self):
        # Every subject has the same accuracy, so all of the observed spread
        # comes from finite trials.  Regrouping at random must reproduce it.
        table = fake_prediction_table([0.50] * 20, n_trials=100,
                                      rng=np.random.default_rng(3))
        row = analyse_regrouping(table, self.rng, conditions=("SRC",),
                                 n_repeat=400)["SRC"][0]
        self.assertLess(row["inflation_factor"], 1.6)
        self.assertGreater(row["p_exceeds_random"], 0.05)

    def test_a_real_subject_effect_is_detected(self):
        table = fake_prediction_table(np.linspace(0.25, 0.75, 20), n_trials=100,
                                      rng=np.random.default_rng(4))
        row = analyse_regrouping(table, self.rng, conditions=("SRC",),
                                 n_repeat=400)["SRC"][0]
        self.assertGreater(row["inflation_factor"], 2.0)
        self.assertLess(row["p_exceeds_random"], 0.01)

    def test_pooled_accuracy_is_unchanged_by_regrouping(self):
        table = fake_prediction_table([0.4, 0.6, 0.5, 0.56], n_trials=50,
                                      rng=np.random.default_rng(6))
        # Compare against what the fixture actually realized, not what it was
        # asked for: rounding a requested accuracy to whole trials can move it.
        realized = [r["main_accuracy"] for r in table["SRC"][0].values()]
        row = analyse_regrouping(table, self.rng, conditions=("SRC",),
                                 n_repeat=50)["SRC"][0]
        self.assertAlmostEqual(row["pooled_accuracy"], float(np.mean(realized)),
                               places=9)
        self.assertEqual(row["group_sizes"], [50] * 4)
        self.assertTrue(row["equal_group_sizes"])

    def test_the_null_matches_finite_population_sampling_theory(self):
        # Drawing m of M correctness records without replacement gives a
        # group-mean SD of sqrt((1 - m/M) S^2 / m).  The resampled null must
        # land on that, which is what makes the control interpretable rather
        # than merely empirical.
        table = fake_prediction_table([0.50] * 12, n_trials=120,
                                      rng=np.random.default_rng(9))
        row = analyse_regrouping(table, self.rng, conditions=("SRC",),
                                 n_repeat=800)["SRC"][0]
        self.assertAlmostEqual(row["sd_by_random_groups_mean"],
                               row["sd_predicted_by_sampling"], delta=0.006)

    def test_it_runs_on_the_supervised_arm_too(self):
        table = fake_prediction_table([0.4, 0.6, 0.5], n_trials=50, cond="SUP",
                                      rng=np.random.default_rng(7))
        out = analyse_regrouping(table, self.rng, n_repeat=50)
        self.assertIn("SUP", out)


class DeltaTableTests(unittest.TestCase):

    def test_per_run_values_are_reported_beside_the_average(self):
        table = fake_table({"SRC": ACCS, "EA": ACCS + 0.03})
        # Give one run a different arm value so the per-run column is not
        # trivially constant.
        for subj in table["EA"][1]:
            table["EA"][1][subj]["main_accuracy"] += 0.06
        subjects = sorted(seed_mean_accuracy(table, "SRC"), key=lambda s: int(s))
        out = analyse_deltas(table, Resampler(len(subjects), 2), subjects)
        row = out["rows"][0]
        self.assertEqual([r["seed"] for r in row["per_seed"]], [0, 1, 2])
        self.assertAlmostEqual(row["per_seed"][0]["delta"], 0.03, places=9)
        self.assertAlmostEqual(row["per_seed"][1]["delta"], 0.09, places=9)
        self.assertAlmostEqual(row["delta"]["point"], 0.05, places=9)


class SeedAveragingTests(unittest.TestCase):

    def test_one_subject_contributes_one_point_after_seed_averaging(self):
        table = fake_table({"SRC": [0.4, 0.5, 0.6]})
        for subj in table["SRC"][1]:
            table["SRC"][1][subj]["main_accuracy"] += 0.1
        got = seed_mean_accuracy(table, "SRC")
        self.assertEqual(len(got), 3)
        self.assertAlmostEqual(got["1"], (0.4 + 0.5 + 0.4) / 3, places=9)


if __name__ == "__main__":
    unittest.main()
