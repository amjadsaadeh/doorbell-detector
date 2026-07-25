"""Tests for chunk selection (src/select_chunks.py) and the audio-source
helpers it uses. Replaces draw_data_test.py, which covered these before the
sampling moved out of draw_data.py.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from src.audio_sources import external_background_rows
from src.select_chunks import balance_chunks, cut_into_chunks, split_background_draw


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


def _annotation(name, label, start, end, noise_pool=np.nan):
    return {
        "annotation_id": f"{name}-{label}",
        "file_id": name,
        "start": start,
        "end": end,
        "label": label,
        "audio_file_name": f"{name}.wav",
        "remote_audio_path": "",
        "noise_pool": noise_pool,
        "split_group": f"{name}.wav",
    }


class TestCutIntoChunks(unittest.TestCase):

    def test_sliding_window_respects_overlap(self):
        annotations = pd.DataFrame([_annotation("rec", "front_doorbell", 0.0, 10.0)])
        chunks = cut_into_chunks(annotations, chunk_size=2000, chunk_overlap=250)
        starts = list(chunks["chunk_start"])
        self.assertEqual(starts[:3], [0, 250, 500])
        # last window must end inside the annotation
        self.assertLess(chunks["chunk_end"].max(), 10_000)

    def test_annotation_shorter_than_chunk_yields_nothing(self):
        annotations = pd.DataFrame([_annotation("blip", "front_doorbell", 0.0, 1.5)])
        self.assertEqual(len(cut_into_chunks(annotations, 2000, 250)), 0)

    def test_chunk_end_is_chunk_size_past_start(self):
        annotations = pd.DataFrame([_annotation("rec", "background", 0.0, 6.0)])
        chunks = cut_into_chunks(annotations, chunk_size=2000, chunk_overlap=500)
        self.assertTrue(((chunks["chunk_end"] - chunks["chunk_start"]) == 2000).all())


class TestBalanceChunks(unittest.TestCase):

    def setUp(self):
        self.params = {"inbalance_ratio": 1.0, "external_background_ratio": 0.5}
        annotations = pd.DataFrame(
            [
                _annotation("bell", "front_doorbell", 0.0, 30.0),
                _annotation("room", "background", 0.0, 60.0),
                _annotation("esc", "background", 0.0, 60.0, noise_pool="esc50"),
            ]
        )
        self.chunks = cut_into_chunks(annotations, 2000, 250)

    def test_background_matches_positive_count(self):
        balanced = balance_chunks(self.chunks, self.params)
        counts = balanced.groupby("label").size()
        self.assertEqual(counts["background"], counts["front_doorbell"])

    def test_background_split_between_real_and_external(self):
        balanced = balance_chunks(self.chunks, self.params)
        background = balanced[balanced["label"] == "background"]
        n_external = background["noise_pool"].notna().sum()
        n_real = background["noise_pool"].isna().sum()
        self.assertEqual(n_real, n_external)

    def test_selection_is_deterministic(self):
        """Feature variants re-run this stage independently; identical input
        must give an identical manifest or the comparison is meaningless."""
        first = balance_chunks(self.chunks, self.params)
        second = balance_chunks(self.chunks, self.params)
        pd.testing.assert_frame_equal(first, second)

    def test_rows_are_shuffled_not_grouped_by_label(self):
        balanced = balance_chunks(self.chunks, self.params)
        labels = list(balanced["label"])
        transitions = sum(a != b for a, b in zip(labels, labels[1:]))
        self.assertGreater(transitions, len(labels) // 10)

    def test_inbalance_ratio_scales_background(self):
        balanced = balance_chunks(
            self.chunks, {**self.params, "inbalance_ratio": 0.5}
        )
        counts = balanced.groupby("label").size()
        self.assertEqual(counts["background"], counts["front_doorbell"] // 2)


class TestHeadCompatibility(unittest.TestCase):

    def test_cnn_rejects_time_pooled_features(self):
        from src.train_model import check_compatible

        with self.assertRaises(SystemExit) as ctx:
            check_compatible("cnn", "yamnet")
        self.assertIn("xgboost", str(ctx.exception))

    def test_xgboost_accepts_any_feature_type(self):
        from src.train_model import check_compatible

        for feature_type in ["mfcc", "logmel", "stft", "yamnet"]:
            with self.subTest(feature_type):
                check_compatible("xgboost", feature_type)

    def test_unknown_head_is_reported(self):
        from src.train_model import check_compatible

        with self.assertRaises(SystemExit) as ctx:
            check_compatible("randomforest", "logmel")
        self.assertIn("randomforest", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
