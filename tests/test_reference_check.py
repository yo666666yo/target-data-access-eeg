"""The reference comparison must refuse an incomplete sweep.

While the braindecode runs were still landing, the comparison happily averaged
each arm over whatever seeds had finished and then paired the arms against each
other -- source-only from runs 1 and 2 against alignment from runs 0 and 2.
That produces a plausible-looking table from runs that were never comparable,
which is exactly the failure mode the paper is about, so it is guarded here.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.reference_check import effects  # noqa: E402


def write_log(directory, cond, seed, accuracies):
    """Emit the minimum a schema-v4 log needs for the comparison to load it."""

    align = cond in ("EA",)
    sup = cond in ("SUP",)
    legacy = cond == "SEL"
    protocol = {
        "normalization_mode": "outer_train_scaler",
        "test_subject_used_for_model_selection": legacy,
        "euclidean_alignment": align,
        "target_supervised": sup,
    }
    log = {
        "schema_version": 4,
        "status": "completed",
        "seed": seed,
        "dataset": "BNCI2014_001",
        "protocol": protocol,
        "per_subject": {str(i + 1): {"main_accuracy": float(a)}
                        for i, a in enumerate(accuracies)},
    }
    name = "%s_s%d.json" % (cond, seed)
    (Path(directory) / name).write_text(json.dumps(log), encoding="utf-8")


class SeedCoverageGuardTests(unittest.TestCase):

    BASE = [0.40, 0.45, 0.50, 0.55, 0.60]

    def test_matched_seeds_are_compared(self):
        with TemporaryDirectory() as d:
            for seed in (0, 1, 2):
                write_log(d, "SRC", seed, self.BASE)
                write_log(d, "EA", seed, [a + 0.05 for a in self.BASE])
            acc, eff = effects(str(Path(d) / "*.json"))
            self.assertIsNotNone(acc)
            self.assertIn("EA", eff)
            self.assertAlmostEqual(eff["EA"][0], 5.0, places=6)

    def test_mismatched_seeds_are_refused(self):
        with TemporaryDirectory() as d:
            # The exact shape that slipped through: no seed is shared by both
            # arms, yet each arm on its own looks complete enough to average.
            for seed in (1, 2):
                write_log(d, "SRC", seed, self.BASE)
            for seed in (0,):
                write_log(d, "EA", seed, [a + 0.05 for a in self.BASE])
            acc, eff = effects(str(Path(d) / "*.json"))
            self.assertIsNone(acc)
            self.assertIn("__incomplete__", eff)

    def test_partial_overlap_is_also_refused(self):
        with TemporaryDirectory() as d:
            for seed in (0, 1, 2):
                write_log(d, "SRC", seed, self.BASE)
            for seed in (0, 2):
                write_log(d, "EA", seed, [a + 0.05 for a in self.BASE])
            acc, eff = effects(str(Path(d) / "*.json"))
            self.assertIsNone(acc, "an arm missing a run must not be compared")
            self.assertIn("__incomplete__", eff)


if __name__ == "__main__":
    unittest.main()
