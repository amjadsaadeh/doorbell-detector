"""Tests for detector.py — Phases 2 and 3.

Covers:
  - compute_score(): cross-correlation scoring behaviour
  - select_template_window(): energetic-window trim selection
  - parse_args(): --threshold, --cooldown-seconds, and save clip flags
  - save_clip(): WAV file writing correctness and error safety
"""

import os
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np

# Add data_collection/ to the import path so the module can be imported
# without installing it as a package.
DATA_COLLECTION = str(Path(__file__).parent.parent / "data_collection")
if DATA_COLLECTION not in sys.path:
    sys.path.insert(0, DATA_COLLECTION)


class TestComputeScore(unittest.TestCase):
    """Tests for compute_score(audio_bytes, template)."""

    def setUp(self):
        # 0.5 s of audio at 16 kHz — matches default chunk size
        self.template = np.ones(8000, dtype=np.float32)

    def test_silence_returns_zero(self):
        """All-zero audio against any non-zero template must return exactly 0.0."""
        from detector import compute_score  # noqa: PLC0415
        silence = np.zeros(8000, dtype=np.int16).tobytes()
        score = compute_score(silence, self.template)
        self.assertEqual(score, 0.0)

    def test_matching_signal_returns_high_score(self):
        """Audio identical to the template must return a score > 0.5."""
        from detector import compute_score  # noqa: PLC0415
        audio = (self.template * 16384).astype(np.int16).tobytes()
        score = compute_score(audio, self.template)
        self.assertGreater(score, 0.5)

    def test_return_type_is_float(self):
        """compute_score must return a Python float."""
        from detector import compute_score  # noqa: PLC0415
        audio = np.zeros(8000, dtype=np.int16).tobytes()
        result = compute_score(audio, self.template)
        self.assertIsInstance(result, float)

    def test_score_is_bounded_by_one(self):
        """Loud unrelated noise against a quiet template must never exceed 1.0.

        Regression test for the false-positive bug: the score was normalized
        only by template energy, so it scaled with chunk loudness and loud
        non-doorbell audio produced scores far above any usable threshold.
        """
        from detector import compute_score  # noqa: PLC0415
        rng = np.random.default_rng(42)
        quiet_template = (rng.uniform(-1, 1, 8000) * 0.02).astype(np.float32)
        loud_noise = (rng.uniform(-1, 1, 8000) * 30000).astype(np.int16).tobytes()
        score = compute_score(loud_noise, quiet_template)
        self.assertLessEqual(score, 1.0)

    def test_score_is_amplitude_invariant(self):
        """The same waveform at 20x the template amplitude must score ~1.0, not ~20."""
        from detector import compute_score  # noqa: PLC0415
        rng = np.random.default_rng(7)
        template = (rng.uniform(-1, 1, 8000) * 0.02).astype(np.float32)
        audio = (template * 20 * 32768).astype(np.int16).tobytes()
        score = compute_score(audio, template)
        self.assertGreater(score, 0.9)
        self.assertLessEqual(score, 1.01)

    def test_template_found_inside_longer_window(self):
        """A template embedded mid-way in a longer analysis window must score ~1.0."""
        from detector import compute_score  # noqa: PLC0415
        rng = np.random.default_rng(3)
        template = (rng.uniform(-1, 1, 8000) * 0.5).astype(np.float32)
        window = np.zeros(32000, dtype=np.float32)  # 2 s analysis window
        window[12000:20000] = template
        audio = (window * 32767).astype(np.int16).tobytes()
        score = compute_score(audio, template)
        self.assertGreater(score, 0.9)

    def test_unrelated_noise_scores_low(self):
        """Unrelated noise must score well below the ~0.5 match region."""
        from detector import compute_score  # noqa: PLC0415
        rng = np.random.default_rng(11)
        template = (rng.uniform(-1, 1, 8000) * 0.5).astype(np.float32)
        noise = (rng.uniform(-1, 1, 32000) * 30000).astype(np.int16).tobytes()
        score = compute_score(noise, template)
        self.assertLess(score, 0.3)


class TestSelectTemplateWindow(unittest.TestCase):
    """Tests for select_template_window(template, target_samples)."""

    def test_template_shorter_than_target_returned_unchanged(self):
        """A template already shorter than target_samples must pass through untouched."""
        from detector import select_template_window  # noqa: PLC0415
        template = np.ones(4000, dtype=np.float32)
        windowed, start, end = select_template_window(template, 8000)
        np.testing.assert_array_equal(windowed, template)
        self.assertEqual(start, 0)
        self.assertEqual(end, 4000)

    def test_template_equal_to_target_returned_unchanged(self):
        """A template exactly target_samples long must pass through untouched."""
        from detector import select_template_window  # noqa: PLC0415
        template = np.ones(8000, dtype=np.float32)
        windowed, start, end = select_template_window(template, 8000)
        np.testing.assert_array_equal(windowed, template)
        self.assertEqual(start, 0)
        self.assertEqual(end, 8000)

    def test_selects_highest_energy_window(self):
        """A loud burst inside a longer silent template must be the selected window."""
        from detector import select_template_window  # noqa: PLC0415
        template = np.zeros(24000, dtype=np.float32)  # 1.5 s of silence
        burst_start = 16000  # loud region starts at 1.0 s
        template[burst_start:burst_start + 4000] = 1.0  # 0.25 s loud burst
        windowed, start, end = select_template_window(template, 4000)
        self.assertEqual(start, burst_start)
        self.assertEqual(end, burst_start + 4000)
        self.assertEqual(len(windowed), 4000)

    def test_output_length_matches_target_when_trimmed(self):
        """The trimmed window must always be exactly target_samples long."""
        from detector import select_template_window  # noqa: PLC0415
        rng = np.random.default_rng(0)
        template = rng.uniform(-1, 1, 160000).astype(np.float32)
        windowed, start, end = select_template_window(template, 8000)
        self.assertEqual(len(windowed), 8000)
        self.assertEqual(end - start, 8000)

    def test_edges_faded_when_trimmed(self):
        """Trimmed windows must fade in/out at the edges to avoid hard-cut artifacts."""
        from detector import select_template_window  # noqa: PLC0415
        template = np.ones(160000, dtype=np.float32)
        windowed, _, _ = select_template_window(template, 8000)
        self.assertAlmostEqual(windowed[0], 0.0, places=5)
        self.assertLess(windowed[1], windowed[40])


class TestParseArgsNewFlags(unittest.TestCase):
    """Tests for --threshold and --cooldown-seconds CLI flags added in Phase 2."""

    def _parse(self, extra_argv=None):
        """Parse args with a minimal valid argv, optionally adding extra flags."""
        original = sys.argv[:]
        sys.argv = ["detector.py", "--template", "dummy.wav"] + (extra_argv or [])
        try:
            from detector import parse_args  # noqa: PLC0415
            return parse_args()
        finally:
            sys.argv = original

    def test_threshold_default(self):
        """--threshold must default to 0.7."""
        args = self._parse()
        self.assertAlmostEqual(args.threshold, 0.7)

    def test_cooldown_seconds_default(self):
        """--cooldown-seconds must default to 10.0."""
        args = self._parse()
        self.assertAlmostEqual(args.cooldown_seconds, 10.0)

    def test_threshold_override(self):
        """--threshold value passed on CLI must be reflected in args."""
        args = self._parse(["--threshold", "0.5"])
        self.assertAlmostEqual(args.threshold, 0.5)

    def test_cooldown_seconds_override(self):
        """--cooldown-seconds value passed on CLI must be reflected in args."""
        args = self._parse(["--cooldown-seconds", "5.0"])
        self.assertAlmostEqual(args.cooldown_seconds, 5.0)

    def test_analysis_window_seconds_default(self):
        """--analysis-window-seconds must default to 2.0."""
        args = self._parse()
        self.assertAlmostEqual(args.analysis_window_seconds, 2.0)

    def test_analysis_window_seconds_override(self):
        """--analysis-window-seconds value passed on CLI must be reflected in args."""
        args = self._parse(["--analysis-window-seconds", "1.0"])
        self.assertAlmostEqual(args.analysis_window_seconds, 1.0)


class TestSaveClip(unittest.TestCase):
    """Tests for save_clip(frames, filepath)."""

    def setUp(self):
        # mktemp gives us a path without holding an open file handle so wave.open can create it.
        self.tmp_path = tempfile.mktemp(suffix=".wav")

    def tearDown(self):
        import os
        if Path(self.tmp_path).exists():
            os.remove(self.tmp_path)

    def test_empty_frames_creates_valid_wav(self):
        """save_clip with no frames must create a valid WAV (0 audio bytes, correct headers)."""
        from detector import save_clip  # noqa: PLC0415
        save_clip([], self.tmp_path)
        with wave.open(self.tmp_path, "rb") as wf:
            self.assertEqual(wf.getnchannels(), 1)
            self.assertEqual(wf.getsampwidth(), 2)
            self.assertEqual(wf.getframerate(), 16000)

    def test_single_chunk_roundtrip(self):
        """save_clip with one 8000-sample chunk must write exactly 8000 frames."""
        from detector import save_clip  # noqa: PLC0415
        # 8000 int16 zero samples = 0.5 s at 16 kHz
        chunk_bytes = (b"\x00\x00" * 8000)
        save_clip([chunk_bytes], self.tmp_path)
        with wave.open(self.tmp_path, "rb") as wf:
            self.assertEqual(wf.getnframes(), 8000)

    def test_exception_logged_on_bad_path(self):
        """save_clip must not raise even when the path is unwritable — error is logged."""
        from detector import save_clip  # noqa: PLC0415
        # Should not raise; errors are caught internally
        save_clip([], "/no/such/dir/x.wav")


class TestSaveArgs(unittest.TestCase):
    """Tests for --save, --save-dir, --buffer-minutes, --post-trigger-seconds CLI flags."""

    def _parse(self, extra_argv=None):
        """Parse args with a minimal valid argv, optionally adding extra flags."""
        original = sys.argv[:]
        sys.argv = ["detector.py", "--template", "dummy.wav"] + (extra_argv or [])
        try:
            from detector import parse_args  # noqa: PLC0415
            return parse_args()
        finally:
            sys.argv = original

    def test_save_default_false(self):
        """--save must default to False."""
        args = self._parse()
        self.assertFalse(args.save)

    def test_save_dir_default(self):
        """--save-dir must default to 'recordings'."""
        args = self._parse()
        self.assertEqual(args.save_dir, "recordings")

    def test_buffer_minutes_default(self):
        """--buffer-minutes must default to BUFFER_SECONDS env var (seconds) / 60."""
        with mock.patch.dict(os.environ, {"BUFFER_SECONDS": "3"}):
            args = self._parse()
        self.assertAlmostEqual(args.buffer_minutes, 3 / 60)

    def test_post_trigger_seconds_default(self):
        """--post-trigger-seconds must default to POST_TRIGGER_MINUTES env var (minutes) * 60."""
        with mock.patch.dict(os.environ, {"POST_TRIGGER_MINUTES": "0.1"}):
            args = self._parse()
        self.assertAlmostEqual(args.post_trigger_seconds, 0.1 * 60)

    def test_save_flag_sets_true(self):
        """--save flag must set args.save to True."""
        args = self._parse(["--save"])
        self.assertTrue(args.save)

    def test_save_dir_override(self):
        """--save-dir override must be reflected in args.save_dir."""
        args = self._parse(["--save-dir", "/tmp/clips"])
        self.assertEqual(args.save_dir, "/tmp/clips")

    def test_buffer_minutes_override(self):
        """--buffer-minutes override must be reflected in args.buffer_minutes."""
        args = self._parse(["--buffer-minutes", "1.5"])
        self.assertAlmostEqual(args.buffer_minutes, 1.5)

    def test_post_trigger_seconds_override(self):
        """--post-trigger-seconds override must be reflected in args.post_trigger_seconds."""
        args = self._parse(["--post-trigger-seconds", "5.0"])
        self.assertAlmostEqual(args.post_trigger_seconds, 5.0)


if __name__ == "__main__":
    unittest.main()
