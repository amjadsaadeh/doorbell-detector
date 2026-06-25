"""Tests for detector.py — Phase 2 Plan 01.

Covers:
  - compute_score(): cross-correlation scoring behaviour
  - parse_args(): new --threshold and --cooldown-seconds flags
"""

import sys
import unittest
from pathlib import Path
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


if __name__ == "__main__":
    unittest.main()
