"""Tests for the schema-v4 experimental controls.

These cover the properties the paper's comparisons depend on: that the source
partition and the reserved test trials do not move between conditions, that the
target sampling rate is what was declared rather than what the cohort size
implies, and that the alignment reference for the held-out subject is estimated
from the available half alone.

The v3 design failed each of these, and none of the failures were visible in a
results table -- which is why they are asserted here rather than checked by
inspection.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.schema import (  # noqa: E402
    NORMALIZATION_MODES,
    TARGET_ACCESS_MODES,
    evaluation_unit,
    protocol_id,
    protocol_id_v4,
    selection_scope_v4,
    target_access_mode,
    target_reference_scope,
    validate_log,
)
# Aliased on import: pytest would otherwise collect a schema predicate whose
# name begins with test_ as a test case.
from experiments.schema import (  # noqa: E402
    test_subject_used_for_normalization as normalization_touches_target,
)

sys.path.insert(0, str(ROOT / "tests"))
from test_experiment_schema import (  # noqa: E402
    finalize_source_descriptor,
    make_log,
    set_subject_values,
)


class SyntheticSource:
    """A deterministic stand-in for a MOABB dataset.

    Each subject gets its own spatial mixing matrix, so Euclidean Alignment has
    something real to remove and the aligned and unaligned folds are genuinely
    different arrays.
    """

    def __init__(self, n_subjects=4, n_trials=40, n_chan=6, n_time=64,
                 n_classes=2, seed=7):
        self.subject_list = list(range(1, n_subjects + 1))
        self._blocks = {}
        rng = np.random.default_rng(seed)
        for s in self.subject_list:
            mixing = rng.normal(size=(n_chan, n_chan)) + 3.0 * np.eye(n_chan)
            latent = rng.normal(size=(n_trials, n_chan, n_time))
            X = np.einsum('cd,ndt->nct', mixing, latent).astype(np.float32)
            y = np.tile(np.arange(n_classes), n_trials // n_classes)
            self._blocks[s] = (X, y)

    def get_data(self, subjects):
        Xs = [self._blocks[s][0] for s in subjects]
        ys = [self._blocks[s][1] for s in subjects]
        return np.concatenate(Xs), np.concatenate(ys), None


def _runner():
    """Import the runner lazily; it pulls in torch, which the schema tests do not."""

    from experiments import paper_runner
    return paper_runner


def make_v4_log(*, align=False, supervised=False, weight=0.10,
                normalization="outer_train_scaler", legacy_loso=False,
                seed=0):
    """A synthetic schema-v4 run log that the gate should accept."""

    log = make_log(seed=seed, normalization=normalization,
                   legacy_loso=legacy_loso)
    split_level = "trial" if supervised else "subject"
    protocol = log["protocol"]
    protocol.update({
        "protocol_id": protocol_id_v4(align, supervised, normalization,
                                      legacy_loso),
        "split_level": split_level,
        "evaluation_unit": evaluation_unit(split_level),
        "test_subject_used_for_training": supervised,
        "euclidean_alignment": align,
        "test_subject_signals_used_for_alignment": align,
        "target_supervised": supervised,
        "target_access": target_access_mode(align, supervised),
        "target_reference_scope": target_reference_scope(align),
        "selection_scope": selection_scope_v4(legacy_loso),
        "target_sample_weight": weight if supervised else 0.0,
        "source_partition":
            "fixed_stratified_80_20_of_outer_pool_shared_by_all_conditions",
        "normalization_mode": normalization,
        "test_subject_used_for_normalization":
            normalization_touches_target(normalization),
    })
    log["run_id"] = "eegnet_equal_BNCI2014_001_s%d__%s" % (
        seed, protocol["protocol_id"])
    log["normalization"] = normalization
    log["split_level"] = split_level
    set_subject_values(log, 0.42)
    finalize_source_descriptor(log)
    return log


class DatasetMetadataTests(unittest.TestCase):
    """Dataset facts must come from one place, not be copied per call site."""

    def test_every_declared_dataset_gets_a_class_count(self):
        from experiments.schema import DATASET_MAIN_CLASSES, task_definition

        class Task:
            def __init__(self, n):
                self.name, self.n_classes, self.loss_weight = "main", n, 1.0

        for dataset, n in DATASET_MAIN_CLASSES.items():
            got = task_definition(dataset, [Task(n)])
            # A private copy of this mapping once went stale for two datasets,
            # which put a null here and made the gate reject every one of their
            # logs while leaving the accuracies untouched.
            self.assertEqual(got["n_original_classes"], n, msg=dataset)

    def test_an_undeclared_dataset_is_refused_rather_than_defaulted(self):
        from experiments.schema import task_definition

        class Task:
            name, n_classes, loss_weight = "main", 2, 1.0

        with self.assertRaises(ValueError):
            task_definition("NotADataset", [Task()])


class SchemaV4ValidationTests(unittest.TestCase):

    def test_all_four_conditions_validate(self):
        for align in (False, True):
            for sup in (False, True):
                log = make_v4_log(align=align, supervised=sup)
                self.assertEqual(validate_log(log), [],
                                 msg="align=%s supervised=%s" % (align, sup))

    def test_mislabelled_access_mode_is_rejected(self):
        log = make_v4_log(align=True, supervised=False)
        log["protocol"]["target_access"] = "none"
        self.assertTrue(any("target_access" in e for e in validate_log(log)))

    def test_supervised_run_without_a_sampling_weight_is_rejected(self):
        log = make_v4_log(supervised=True)
        log["protocol"]["target_sample_weight"] = 0.0
        self.assertTrue(any("target_sample_weight" in e
                            for e in validate_log(log)))

    def test_unsupervised_run_declaring_a_weight_is_rejected(self):
        log = make_v4_log(supervised=False)
        log["protocol"]["target_sample_weight"] = 0.10
        self.assertTrue(any("target_sample_weight" in e
                            for e in validate_log(log)))

    def test_training_flag_must_agree_with_supervision(self):
        log = make_v4_log(supervised=False)
        log["protocol"]["test_subject_used_for_training"] = True
        self.assertTrue(any("test_subject_used_for_training" in e
                            for e in validate_log(log)))

    def test_alignment_reference_scope_must_agree_with_the_flag(self):
        log = make_v4_log(align=True)
        log["protocol"]["target_reference_scope"] = "not_applicable"
        self.assertTrue(any("target_reference_scope" in e
                            for e in validate_log(log)))

    def test_a_v3_log_still_validates_against_the_v3_builder(self):
        log = make_log(normalization="outer_train_scaler")
        set_subject_values(log, 0.42)
        finalize_source_descriptor(log)
        self.assertNotIn("target_supervised", log["protocol"])
        self.assertEqual(validate_log(log), [])


class ProtocolIdentityTests(unittest.TestCase):

    def test_four_conditions_have_distinct_identities(self):
        ids = {
            protocol_id_v4(a, s, "outer_train_scaler")
            for a in (False, True) for s in (False, True)
        }
        self.assertEqual(len(ids), 4)

    def test_v4_identity_never_collides_with_a_v3_identity(self):
        v3 = {
            protocol_id(legacy, norm, split, align)
            for legacy in (False, True)
            for norm in NORMALIZATION_MODES
            for split in ("subject", "trial")
            for align in (False, True)
            if not (legacy and norm == "inner_train_scaler")
        }
        v4 = {
            protocol_id_v4(a, s, norm, legacy)
            for a in (False, True) for s in (False, True)
            for norm in NORMALIZATION_MODES for legacy in (False, True)
        }
        self.assertEqual(v3 & v4, set())

    def test_access_mode_covers_the_declared_vocabulary(self):
        seen = {target_access_mode(a, s)
                for a in (False, True) for s in (False, True)}
        self.assertEqual(seen, set(TARGET_ACCESS_MODES))

    def test_reference_scope_names_the_available_half(self):
        self.assertIn("available", target_reference_scope(True))
        self.assertEqual(target_reference_scope(False), "not_applicable")

    def test_selection_scope_distinguishes_the_secondary_arm(self):
        self.assertNotEqual(selection_scope_v4(True), selection_scope_v4(False))
        self.assertIn("source", selection_scope_v4(False))


class SamplingRateTests(unittest.TestCase):

    def test_source_only_weights_are_uniform(self):
        pr = _runner()
        w = pr._training_sample_weights(100, 0, 0.10)
        self.assertEqual(len(w), 100)
        self.assertTrue(np.allclose(w, w[0]))

    def test_declared_target_share_is_the_expected_share(self):
        pr = _runner()
        for n_src, n_tgt in ((3686, 288), (1664, 80), (7486, 45)):
            w = pr._training_sample_weights(n_src, n_tgt, 0.10)
            share = w[n_src:].sum() / w.sum()
            self.assertAlmostEqual(share, 0.10, places=9)

    def test_realized_share_matches_the_declaration_across_cohort_sizes(self):
        pr = _runner()
        # Under a plain shuffled loader these three would give 7.2%, 4.6% and
        # 0.6%; the point of the sampler is that they no longer do.
        for n_src, n_tgt in ((3686, 288), (1664, 80), (7486, 45)):
            w = pr._training_sample_weights(n_src, n_tgt, 0.10)
            got = pr._measure_target_fraction(w, n_src, n_tgt, seed=0)
            self.assertAlmostEqual(got, 0.10, delta=0.02)

    def test_no_target_trials_means_no_target_draws(self):
        pr = _runner()
        w = pr._training_sample_weights(500, 0, 0.10)
        self.assertEqual(pr._measure_target_fraction(w, 500, 0, seed=0), 0.0)


class FoldConstructionTests(unittest.TestCase):

    def setUp(self):
        self.pr = _runner()
        self.source = SyntheticSource()
        self.subjects = self.source.subject_list

    def _folds(self, align):
        return list(self.pr._iter_evaluation_folds(
            self.source, self.subjects, seed=0, align=align))

    def test_reserved_test_labels_do_not_move_with_alignment(self):
        plain = self._folds(False)
        aligned = self._folds(True)
        self.assertEqual(len(plain), len(aligned))
        for (s_a, *_, y_a), (s_b, *_, y_b) in zip(plain, aligned):
            self.assertEqual(s_a, s_b)
            np.testing.assert_array_equal(y_a, y_b)

    def test_source_partition_is_identical_across_conditions(self):
        plain = self._folds(False)
        aligned = self._folds(True)
        for (_, _, y_outer_a, *_), (_, _, y_outer_b, *_) in zip(plain, aligned):
            tr_a, va_a = self.pr._split_outer_train(y_outer_a, 0)
            tr_b, va_b = self.pr._split_outer_train(y_outer_b, 0)
            np.testing.assert_array_equal(tr_a, tr_b)
            np.testing.assert_array_equal(va_a, va_b)

    def test_alignment_reference_uses_the_available_half_only(self):
        from sklearn.model_selection import train_test_split

        for subj, _, _, X_avail, _, X_te, _ in self._folds(True):
            X_s, y_s = self.source._blocks[subj]
            idx_avail, idx_test = train_test_split(
                np.arange(len(y_s)), test_size=self.pr.TARGET_TEST_FRACTION,
                stratify=y_s, random_state=0 * 1000 + int(subj))

            expected = self.pr._ea_whitener(X_s[idx_avail])
            np.testing.assert_allclose(
                X_avail, self.pr._apply_whitener(expected, X_s[idx_avail]),
                rtol=1e-4, atol=1e-4)
            np.testing.assert_allclose(
                X_te, self.pr._apply_whitener(expected, X_s[idx_test]),
                rtol=1e-4, atol=1e-4)

            # And it is not the whole-subject reference the v3 runner used.
            whole = self.pr._apply_whitener(
                self.pr._ea_whitener(X_s), X_s[idx_test])
            self.assertFalse(np.allclose(X_te, whole, rtol=1e-4, atol=1e-4))

    def test_held_out_subject_never_appears_in_the_source_pool(self):
        for subj, X_outer, y_outer, *_ in self._folds(False):
            expected = sum(len(self.source._blocks[s][1])
                           for s in self.subjects if s != subj)
            self.assertEqual(len(y_outer), expected)
            self.assertEqual(len(X_outer), expected)

    def test_available_and_reserved_halves_are_disjoint_and_exhaustive(self):
        for subj, _, _, X_avail, y_avail, X_te, y_te in self._folds(False):
            X_s, _ = self.source._blocks[subj]
            self.assertEqual(len(y_avail) + len(y_te), len(X_s))
            rows = {tuple(r) for r in X_avail.reshape(len(X_avail), -1)}
            for r in X_te.reshape(len(X_te), -1):
                self.assertNotIn(tuple(r), rows)


class FactorizationAxisTests(unittest.TestCase):
    """The aggregator has to pick its 2x2 axes from the logs, not by assumption."""

    def test_v4_logs_select_the_target_access_axes(self):
        from experiments.aggregate import _factorization_axes

        logs = [make_v4_log(align=a, supervised=s)
                for a in (False, True) for s in (False, True)]
        axes = _factorization_axes(logs)
        self.assertTrue(axes["is_v4"])
        self.assertEqual(axes["columns"],
                         ("euclidean_alignment", "target_supervised"))
        cells = {axes["cell_of"](log) for log in logs if axes["eligible"](log)}
        self.assertEqual(cells, {(False, False), (False, True),
                                 (True, False), (True, True)})

    def test_secondary_arms_are_kept_off_the_grid(self):
        from experiments.aggregate import _factorization_axes

        grid = [make_v4_log(align=a, supervised=s)
                for a in (False, True) for s in (False, True)]
        pooled = make_v4_log(normalization="pooled_all_subject_scaler")
        selected = make_v4_log(legacy_loso=True)
        axes = _factorization_axes(grid + [pooled, selected])
        self.assertFalse(axes["eligible"](pooled))
        self.assertFalse(axes["eligible"](selected))
        self.assertTrue(all(axes["eligible"](log) for log in grid))

    def test_v3_logs_still_select_the_old_axes(self):
        from experiments.aggregate import _factorization_axes

        axes = _factorization_axes([make_log(normalization="outer_train_scaler")])
        self.assertFalse(axes["is_v4"])
        self.assertEqual(axes["columns"], ("legacy_loso", "legacy_norm"))


if __name__ == "__main__":
    unittest.main()
