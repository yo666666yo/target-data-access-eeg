"""The chronological split must refuse datasets whose session order is unknown.

Splitting a subject at the midpoint is only a session split if the loader put
whole sessions there in a known order.  On a dataset that interleaves runs, or
concatenates them in an order nobody checked, the same code would silently
return an arbitrary partition wearing the word "session", which is worse than
having no such option: the paper would claim a chronological test it did not
run.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "EEGNets"))
sys.path.insert(0, str(ROOT / "configs"))

from experiments.paper_runner import (  # noqa: E402
    SESSION_SPLIT_DATASETS, _session_split,
)


class SessionSplitTests(unittest.TestCase):

    def test_2a_splits_at_the_session_boundary(self):
        avail, test = _session_split(576, "BNCI2014_001")
        self.assertEqual(len(avail), 288)
        self.assertEqual(len(test), 288)
        # The earlier session is the one a condition may use; the later one is
        # reserved.  Reversing this would test the past from the future.
        self.assertEqual(avail[0], 0)
        self.assertEqual(avail[-1], 287)
        self.assertEqual(test[0], 288)
        self.assertEqual(test[-1], 575)

    def test_halves_are_disjoint_and_cover_every_trial(self):
        avail, test = _session_split(576, "BNCI2014_001")
        self.assertEqual(set(avail) & set(test), set())
        self.assertEqual(sorted(np.concatenate([avail, test])),
                         list(range(576)))

    def test_unsupported_dataset_is_refused(self):
        for dataset in ("PhysionetMI", "BNCI2014_002", "NotADataset"):
            with self.assertRaises(ValueError, msg=dataset):
                _session_split(576, dataset)

    def test_odd_trial_count_is_refused(self):
        # An odd count means the two halves cannot both be whole sessions, so
        # the premise of the split has already failed.
        with self.assertRaises(ValueError):
            _session_split(575, "BNCI2014_001")

    def test_only_declared_datasets_are_supported(self):
        # Guards against someone adding a dataset to the map without checking
        # that its loader really concatenates sessions in order.
        self.assertEqual(set(SESSION_SPLIT_DATASETS), {"BNCI2014_001"})


if __name__ == "__main__":
    unittest.main()
