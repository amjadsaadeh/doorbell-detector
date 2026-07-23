import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from src import draw_data
from src.draw_data import (
    external_background_rows,
    get_yamnet_features,
    split_background_draw,
)


class TestGetYamnetFeatures(unittest.TestCase):
    """Chunk slicing over per-file YAMNet frame embeddings (0.48s hop),
    mean-pooled to a fixed-size vector regardless of how many frames the
    file actually has (a 2s augmented clip has 3 frames, a 2s slice of a
    long file has 4 — see the MFCC frame-rate comment in draw_data.py)."""

    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self._orig_base = draw_data.YAMNET_FEATURES_FILE_BASE
        draw_data.YAMNET_FEATURES_FILE_BASE = Path(self._tmp_dir.name)

    def tearDown(self):
        draw_data.YAMNET_FEATURES_FILE_BASE = self._orig_base
        self._tmp_dir.cleanup()

    def _write_embeddings(self, name: str, n_frames: int, dim: int = 8) -> np.ndarray:
        # frame i is a constant vector of value i, so an expected mean over
        # any frame range is trivial to compute
        emb = np.tile(np.arange(n_frames, dtype=np.float32), (dim, 1))
        np.save(Path(self._tmp_dir.name) / name.replace(".wav", ".npy"), emb)
        return emb

    def test_full_chunk_of_long_file(self):
        self._write_embeddings("long.wav", n_frames=10)
        res = get_yamnet_features(0, 2000, "long.wav")
        # 2000ms // 480ms hop -> frames 0..3
        np.testing.assert_allclose(res, np.full(8, np.mean([0, 1, 2, 3])))
        self.assertEqual(res.shape, (8,))

    def test_chunk_with_offset(self):
        self._write_embeddings("long.wav", n_frames=10)
        res = get_yamnet_features(1000, 3000, "long.wav")
        # start frame 1000 // 480 = 2 -> frames 2..5
        np.testing.assert_allclose(res, np.full(8, np.mean([2, 3, 4, 5])))

    def test_exactly_chunk_sized_file_has_one_frame_less(self):
        # a 2s clip yields only 3 YAMNet frames (no final partial window);
        # pooling must still return the same shape as the 4-frame case
        self._write_embeddings("aug.wav", n_frames=3)
        res = get_yamnet_features(0, 2000, "aug.wav")
        np.testing.assert_allclose(res, np.full(8, np.mean([0, 1, 2])))
        self.assertEqual(res.shape, (8,))

    def test_empty_slice_falls_back_to_last_frame(self):
        self._write_embeddings("short.wav", n_frames=4)
        res = get_yamnet_features(2000, 4000, "short.wav")
        np.testing.assert_allclose(res, np.full(8, 3.0))


class TestSplitBackgroundDraw(unittest.TestCase):

    def test_exact_split(self):
        self.assertEqual(split_background_draw(100, 0.5, 1000, 1000), (50, 50))

    def test_all_real(self):
        self.assertEqual(split_background_draw(100, 0.0, 1000, 1000), (100, 0))

    def test_all_external(self):
        self.assertEqual(split_background_draw(100, 1.0, 1000, 1000), (0, 100))

    def test_real_shortfall_topped_up_by_external(self):
        # wants 80 real / 20 external, but only 30 real chunks exist
        self.assertEqual(split_background_draw(100, 0.2, 30, 1000), (30, 70))

    def test_external_shortfall_topped_up_by_real(self):
        # wants 20 real / 80 external, but only 10 external chunks exist
        self.assertEqual(split_background_draw(100, 0.8, 1000, 10), (90, 10))

    def test_both_short_returns_what_exists(self):
        self.assertEqual(split_background_draw(100, 0.5, 30, 40), (30, 40))


class TestExternalBackgroundRows(unittest.TestCase):

    def test_one_background_row_per_pool_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            pool_dir = Path(tmp_dir) / "esc50"
            pool_dir.mkdir()
            samples = np.zeros(16000, dtype=np.int16)
            sf.write(pool_dir / "a.wav", samples, 16000)
            sf.write(pool_dir / "b.wav", samples, 16000)

            rows = external_background_rows([str(pool_dir)])

            self.assertEqual(len(rows), 2)
            self.assertTrue((rows["label"] == "background").all())
            self.assertTrue((rows["noise_pool"] == "esc50").all())
            self.assertTrue((rows["start"] == 0.0).all())
            # no end time: the real file duration is filled in downstream,
            # like tag-only real background annotations
            self.assertTrue(rows["end"].isna().all())
            self.assertEqual(list(rows["audio_file_name"]), ["a.wav", "b.wav"])

    def test_empty_pool_list(self):
        self.assertEqual(len(external_background_rows([])), 0)


if __name__ == "__main__":
    unittest.main()
