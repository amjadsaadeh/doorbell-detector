"""Tests for the quantize / evaluate-quantized split.

The parts that need TensorFlow to actually convert a graph are covered by
the pipeline run itself; what is worth unit-testing is the calibration
subset selection, since a mistake there is silent -- the model still
quantizes, it is just calibrated on the wrong chunks.
"""

import unittest
from pathlib import Path

import numpy as np
import yaml

from src.quantize_model import select_calibration_rows

PARAMS_PATH = Path(__file__).parent.parent / "params.yaml"


class TestSelectCalibrationRows(unittest.TestCase):

    def setUp(self):
        # an offset candidate range, to catch index-space mistakes
        self.train_idx = np.arange(200, 1000)

    def test_never_leaves_the_candidate_rows(self):
        rows = select_calibration_rows(self.train_idx, 500, seed=42)
        self.assertTrue(set(rows).issubset(set(self.train_idx.tolist())))

    def test_indices_are_dataset_global(self):
        """They have to join back to chunk_manifest.csv, so they must be the
        original row numbers, not offsets into a re-indexed subset."""
        rows = select_calibration_rows(self.train_idx, 10, seed=0)
        self.assertTrue((rows >= 200).all())

    def test_requested_count_is_honoured(self):
        self.assertEqual(len(select_calibration_rows(self.train_idx, 500, seed=1)), 500)

    def test_no_duplicates(self):
        rows = select_calibration_rows(self.train_idx, 500, seed=1)
        self.assertEqual(len(rows), len(set(rows.tolist())))

    def test_clamped_to_available_chunks(self):
        """Asking for more calibration chunks than the fold holds must not
        raise or silently sample with replacement."""
        rows = select_calibration_rows(np.arange(50), 500, seed=1)
        self.assertEqual(len(rows), 50)
        self.assertEqual(len(set(rows.tolist())), 50)

    def test_deterministic_for_a_given_seed(self):
        """Reproducing a quantized model means reproducing its calibration."""
        first = select_calibration_rows(self.train_idx, 100, seed=42)
        second = select_calibration_rows(self.train_idx, 100, seed=42)
        np.testing.assert_array_equal(first, second)

    def test_seed_changes_the_selection(self):
        first = select_calibration_rows(self.train_idx, 100, seed=42)
        other = select_calibration_rows(self.train_idx, 100, seed=7)
        self.assertFalse(np.array_equal(first, other))

    def test_sorted_for_stable_artifacts(self):
        rows = select_calibration_rows(self.train_idx, 100, seed=3)
        np.testing.assert_array_equal(rows, np.sort(rows))


class TestQuantizationParams(unittest.TestCase):

    def setUp(self):
        with open(PARAMS_PATH) as f:
            self.params = yaml.safe_load(f)

    def test_quantization_block_is_present(self):
        self.assertIn("quantization", self.params)
        for key in ("calibration_samples", "max_f1_drop"):
            self.assertIn(key, self.params["quantization"])

    def test_max_f1_drop_is_a_real_gate(self):
        """A limit of 1.0 would let a total collapse through."""
        self.assertGreater(self.params["quantization"]["max_f1_drop"], 0)
        self.assertLess(self.params["quantization"]["max_f1_drop"], 0.5)


if __name__ == "__main__":
    unittest.main()
