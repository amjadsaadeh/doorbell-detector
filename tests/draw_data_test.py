import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from src.draw_data import external_background_rows, split_background_draw


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
