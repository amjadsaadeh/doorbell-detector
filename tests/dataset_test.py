"""Tests for the chunk dataset container (src/dataset.py).

The reason this file exists: the previous container stored per-row numpy
arrays in a pandas object column, which PyTables pickled into bytes that
differed between runs over identical data. DVC hashes files, so draw_data
always looked changed and train_model could never be skipped. Byte
reproducibility is therefore a tested property, not an incidental one.
"""

import hashlib
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.dataset import binarize_labels, read_features, write_dataset


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


class TestWriteReadRoundtrip(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "balanced_data.h5"
        self.X = np.random.default_rng(0).normal(size=(50, 40, 100)).astype(np.float32)

    def tearDown(self):
        self._tmp.cleanup()

    def test_roundtrip_preserves_values_and_type(self):
        write_dataset(self.X, "logmel", self.path)
        loaded, feature_type = read_features(self.path)
        np.testing.assert_array_equal(loaded, self.X)
        self.assertEqual(feature_type, "logmel")

    def test_stored_as_float32(self):
        write_dataset(self.X.astype(np.float64), "logmel", self.path)
        loaded, _ = read_features(self.path)
        self.assertEqual(loaded.dtype, np.float32)

    def test_one_dimensional_chunks_roundtrip(self):
        """yamnet pools the time axis away, so chunks are 1D vectors."""
        embeddings = np.random.default_rng(1).normal(size=(50, 1024)).astype(np.float32)
        write_dataset(embeddings, "yamnet", self.path)
        loaded, feature_type = read_features(self.path)
        self.assertEqual(loaded.shape, (50, 1024))
        self.assertEqual(feature_type, "yamnet")


class TestByteReproducibility(unittest.TestCase):
    """The bug this container replaced: identical content, different bytes,
    so DVC saw a change on every run and always retrained."""

    def test_identical_data_produces_identical_bytes(self):
        X = np.random.default_rng(2).normal(size=(100, 40, 100)).astype(np.float32)
        digests = []
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("a.h5", "b.h5"):
                path = Path(tmp) / name
                write_dataset(X, "logmel", path)
                digests.append(md5(path))
                # a wall-clock gap, so any embedded timestamp would show up
                time.sleep(1.1)
        self.assertEqual(digests[0], digests[1])

    def test_different_data_produces_different_bytes(self):
        rng = np.random.default_rng(3)
        with tempfile.TemporaryDirectory() as tmp:
            first, second = Path(tmp) / "a.h5", Path(tmp) / "b.h5"
            write_dataset(rng.normal(size=(20, 4, 5)).astype(np.float32), "logmel", first)
            write_dataset(rng.normal(size=(20, 4, 5)).astype(np.float32), "logmel", second)
            self.assertNotEqual(md5(first), md5(second))


class TestBinarizeLabels(unittest.TestCase):

    def test_everything_not_background_becomes_bell(self):
        y, encoder = binarize_labels(
            pd.Series(["background", "front_doorbell", "flat_doorbell"])
        )
        # background=0, bell=1 (alphabetical LabelEncoder order)
        np.testing.assert_array_equal(y, [0, 1, 1])
        self.assertEqual(list(encoder.classes_), ["background", "bell"])

    def test_single_class_input_does_not_crash(self):
        y, _ = binarize_labels(pd.Series(["background", "background"]))
        np.testing.assert_array_equal(y, [0, 0])


if __name__ == "__main__":
    unittest.main()
