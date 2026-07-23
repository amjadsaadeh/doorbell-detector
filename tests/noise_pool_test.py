import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from src.augment_data import draw_noise_window, index_pool_windows
from src.fetch_noise_esc50 import select_clips
from src.noise_pool import write_16k_mono_wav


class TestSelectClips(unittest.TestCase):

    def setUp(self):
        self.meta = pd.DataFrame(
            {
                "filename": ["a.wav", "b.wav", "c.wav"],
                "category": ["church_bells", "vacuum_cleaner", "rain"],
            }
        )

    def test_excludes_categories(self):
        clips = select_clips(self.meta, ["church_bells"])
        self.assertEqual(list(clips["filename"]), ["b.wav", "c.wav"])

    def test_empty_exclude_keeps_all(self):
        clips = select_clips(self.meta, [])
        self.assertEqual(len(clips), 3)

    def test_unknown_category_raises(self):
        with self.assertRaises(ValueError):
            select_clips(self.meta, ["church_bell"])  # typo: singular


class TestWrite16kMonoWav(unittest.TestCase):

    def test_resamples_and_downmixes(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "stereo_44k.wav"
            target = Path(tmp_dir) / "out.wav"
            # 2s stereo sine at 44.1kHz, like an ESC-50 clip
            t = np.linspace(0, 2, 2 * 44100, endpoint=False)
            stereo = np.stack([np.sin(2 * np.pi * 440 * t)] * 2, axis=1) * 0.5
            sf.write(source, stereo, 44100)

            write_16k_mono_wav(source, target)

            info = sf.info(str(target))
            self.assertEqual(info.samplerate, 16000)
            self.assertEqual(info.channels, 1)
            self.assertEqual(info.subtype, "PCM_16")
            self.assertEqual(info.frames, 2 * 16000)


class TestPoolWindowIndexing(unittest.TestCase):

    def test_index_and_draw(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            pool_dir = Path(tmp_dir)
            # 5s of noise at 16kHz, like a prepared ESC-50 pool file
            rng = np.random.default_rng(0)
            samples = (rng.uniform(-0.1, 0.1, 5 * 16000) * 32767).astype(np.int16)
            sf.write(pool_dir / "clip.wav", samples, 16000)

            entries = index_pool_windows(pool_dir, chunk_size_ms=2000, chunk_overlap_ms=250)

            # same stride convention as extract_windows / draw_data.py
            self.assertEqual(len(entries), len(range(0, 5000 - 2000, 250)))

            window = draw_noise_window(entries[-1])
            self.assertEqual(len(window), 2 * 16000)
            # last window starts at 2750ms into the clip
            start_sample = int(16 * 2750)
            np.testing.assert_array_equal(
                window, samples[start_sample : start_sample + 2 * 16000].astype(np.float64)
            )

    def test_empty_pool_yields_no_entries(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.assertEqual(index_pool_windows(Path(tmp_dir), 2000, 250), [])


if __name__ == "__main__":
    unittest.main()
