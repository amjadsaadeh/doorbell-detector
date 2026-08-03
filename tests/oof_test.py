"""Tests for the held-out prediction table (src/oof.py).

This table is what someone uses to decide a label is wrong and go change it
in Label Studio, so the two properties worth pinning down are that a row's
predictions belong to the chunk its provenance columns name, and that a
partially-covered dataset says so instead of looking complete.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.oof import CLASS_NAMES, OutOfFoldPredictions, outcome


def manifest_csv(directory: Path, n_chunks: int) -> Path:
    """A manifest shaped like select_chunks.py's, small enough to check by eye.

    Even rows are background, odd rows are front_doorbell, and every row's
    chunk_start encodes its own index -- so a mis-join shows up as a
    mismatch between row_index and the provenance columns.
    """
    path = directory / "chunk_manifest.csv"
    pd.DataFrame(
        {
            "annotation_id": [f"ann_{i}" for i in range(n_chunks)],
            "file_id": [f"file_{i}" for i in range(n_chunks)],
            "audio_file_name": [f"rec_{i // 4}.wav" for i in range(n_chunks)],
            "remote_audio_path": ["s3://bucket/raw/x.wav"] * n_chunks,
            "start": 0.0,
            "end": 9.0,
            "chunk_start": [i * 1000 for i in range(n_chunks)],
            "chunk_end": [i * 1000 + 2000 for i in range(n_chunks)],
            "label": [
                "background" if i % 2 == 0 else "front_doorbell"
                for i in range(n_chunks)
            ],
            "split_group": [f"rec_{i // 4}.wav" for i in range(n_chunks)],
        }
    ).to_csv(path, index=False)
    return path


class TestOutcome(unittest.TestCase):

    def test_all_four_cells(self):
        self.assertEqual(outcome(1, 1), "true_positive")
        self.assertEqual(outcome(0, 0), "true_negative")
        self.assertEqual(outcome(1, 0), "false_negative")
        self.assertEqual(outcome(0, 1), "false_positive")


class TestOutOfFoldPredictions(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.directory = Path(self._tmp.name)
        self.n_chunks = 8
        self.manifest = manifest_csv(self.directory, self.n_chunks)

    def tearDown(self):
        self._tmp.cleanup()

    def full_coverage(self) -> OutOfFoldPredictions:
        """Two disjoint folds covering every chunk, one error planted in each."""
        oof = OutOfFoldPredictions(self.n_chunks)
        # rows 0-3: background, bell, background, bell. The bell at row 3 is
        # scored 0.2 -> a false negative.
        oof.add(0, [0, 1, 2, 3], [0, 1, 0, 1], [0.05, 0.9, 0.1, 0.2])
        # rows 4-7: the background at row 6 is scored 0.8 -> a false positive.
        oof.add(1, [4, 5, 6, 7], [0, 1, 0, 1], [0.02, 0.99, 0.8, 0.95])
        return oof

    def test_predictions_land_on_the_chunk_they_came_from(self):
        table = self.full_coverage().to_frame(self.manifest)

        self.assertEqual(len(table), self.n_chunks)
        # Provenance is what makes a row actionable; a shifted join would
        # silently point at the neighbouring chunk.
        np.testing.assert_array_equal(table["row_index"], np.arange(self.n_chunks))
        self.assertEqual(list(table["annotation_id"]), [f"ann_{i}" for i in range(8)])
        np.testing.assert_array_equal(
            table["chunk_start"], np.arange(self.n_chunks) * 1000
        )
        self.assertEqual(table.loc[3, "audio_file_name"], "rec_0.wav")
        self.assertEqual(table.loc[4, "audio_file_name"], "rec_1.wav")

    def test_outcomes_and_margins(self):
        table = self.full_coverage().to_frame(self.manifest).set_index("row_index")

        self.assertEqual(table.loc[3, "outcome"], "false_negative")
        self.assertEqual(table.loc[6, "outcome"], "false_positive")
        self.assertEqual(table.loc[1, "outcome"], "true_positive")
        self.assertEqual(table.loc[0, "outcome"], "true_negative")
        self.assertEqual(int(table["correct"].sum()), 6)

        # The two errors differ in how badly the model was wrong, and margin
        # is what sorts one above the other.
        self.assertAlmostEqual(table.loc[3, "margin"], 0.3)
        self.assertAlmostEqual(table.loc[6, "margin"], 0.3)
        self.assertAlmostEqual(table.loc[1, "margin"], 0.4)

    def test_class_names_match_the_binary_labels(self):
        table = self.full_coverage().to_frame(self.manifest).set_index("row_index")

        self.assertEqual(CLASS_NAMES, ("background", "bell"))
        # Row 3 is a front_doorbell chunk the model called background.
        self.assertEqual(table.loc[3, "true_class"], "bell")
        self.assertEqual(table.loc[3, "predicted_class"], "background")
        self.assertEqual(table.loc[3, "label"], "front_doorbell")

    def test_sorted_by_row_index_regardless_of_fold_order(self):
        oof = OutOfFoldPredictions(self.n_chunks)
        oof.add(1, [4, 5, 6, 7], [0, 1, 0, 1], [0.1, 0.9, 0.1, 0.9])
        oof.add(0, [0, 1, 2, 3], [0, 1, 0, 1], [0.1, 0.9, 0.1, 0.9])
        table = oof.to_frame(self.manifest)

        np.testing.assert_array_equal(table["row_index"], np.arange(self.n_chunks))
        self.assertEqual(list(table["fold"]), [0, 0, 0, 0, 1, 1, 1, 1])

    def test_summary_counts_the_two_error_kinds_separately(self):
        summary = self.full_coverage().summary(self.manifest)

        self.assertEqual(summary["oof_coverage"], 1.0)
        self.assertEqual(summary["oof_chunks"], 8.0)
        self.assertEqual(summary["oof_errors"], 2.0)
        self.assertEqual(summary["oof_false_negatives"], 1.0)
        self.assertEqual(summary["oof_false_positives"], 1.0)

    def test_partial_coverage_is_reported_not_hidden(self):
        """n_eval_folds=1 leaves most chunks unscored; the table must not
        pretend otherwise, because a small error count on a fifth of the data
        reads very differently from the same count on all of it."""
        oof = OutOfFoldPredictions(self.n_chunks)
        oof.add(0, [0, 1, 2, 3], [0, 1, 0, 1], [0.05, 0.9, 0.1, 0.2])
        table = oof.to_frame(self.manifest)

        self.assertEqual(len(table), 4)
        self.assertEqual(oof.summary(self.manifest)["oof_coverage"], 0.5)

    def test_overlapping_folds_are_rejected(self):
        """CV folds are disjoint by construction; if two ever claim the same
        chunk the split is broken, and silently keeping both predictions
        would hide that."""
        oof = OutOfFoldPredictions(self.n_chunks)
        oof.add(0, [0, 1], [0, 1], [0.1, 0.9])
        oof.add(1, [1, 2], [1, 0], [0.9, 0.1])

        with self.assertRaises(ValueError):
            oof.to_frame(self.manifest)

    def test_ragged_fold_input_is_rejected(self):
        oof = OutOfFoldPredictions(self.n_chunks)
        with self.assertRaises(ValueError):
            oof.add(0, [0, 1, 2], [0, 1], [0.1, 0.9, 0.5])

    def test_stale_manifest_is_rejected(self):
        """balanced_data.h5 and the manifest are row-aligned by construction;
        if they disagree the row_index values point at other chunks."""
        oof = OutOfFoldPredictions(self.n_chunks + 5)
        oof.add(0, [0, 1], [0, 1], [0.1, 0.9])

        with self.assertRaises(SystemExit):
            oof.to_frame(self.manifest)

    def test_write_roundtrips_through_csv(self):
        path = self.directory / "predictions" / "oof_predictions.csv"
        written = self.full_coverage().write(path, self.manifest)

        self.assertEqual(written, path)
        reloaded = pd.read_csv(path).set_index("row_index")
        self.assertEqual(reloaded.loc[3, "outcome"], "false_negative")
        self.assertEqual(reloaded.loc[3, "annotation_id"], "ann_3")


if __name__ == "__main__":
    unittest.main()
