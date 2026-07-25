"""Tests for the unified feature registry (src/features.py).

Replaces feature_extraction_test.py and logmel_extraction_test.py, which
each covered one branch's extractor script.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from src.features import (
    REGISTRY,
    extractor_config,
    fixed_rate_slice,
    get_spec,
    verify_feature_config,
    write_feature_config,
    yamnet_slice,
)

BASE_DATA_PATH = Path(__file__).parent / "data"
PARAMS_PATH = Path(__file__).parent.parent / "params.yaml"
TEST_AUDIO = BASE_DATA_PATH / "audio" / "test_audio.wav"

# Geometry the fixed-rate extractors are exercised at. Independent of
# params.yaml so a sweep of the live config can't quietly weaken the test.
MFCC_CFG = {"n_mfcc": 40, "n_fft": 512, "hop_length": 320}
LOGMEL_CFG = {
    "n_mels": 40,
    "n_fft": 512,
    "hop_length": 320,
    "fmin": 50,
    "fmax": 8000,
    "log_offset": 1e-6,
}
STFT_CFG = {"n_fft": 256, "hop_length": 128}


class TestExtractors(unittest.TestCase):
    """Every fixed-rate extractor must return (n_bins, n_frames) float32."""

    def test_mfcc_bin_count(self):
        out = REGISTRY["mfcc"].extract(TEST_AUDIO, **MFCC_CFG)
        self.assertEqual(out.shape[0], MFCC_CFG["n_mfcc"])
        self.assertEqual(out.dtype, np.float32)

    def test_logmel_bin_count_and_finiteness(self):
        out = REGISTRY["logmel"].extract(TEST_AUDIO, **LOGMEL_CFG)
        self.assertEqual(out.shape[0], LOGMEL_CFG["n_mels"])
        self.assertEqual(out.dtype, np.float32)
        # log_offset is what keeps digital-silence frames from becoming -inf
        self.assertTrue(np.isfinite(out).all())

    def test_stft_bin_count(self):
        out = REGISTRY["stft"].extract(TEST_AUDIO, **STFT_CFG)
        self.assertEqual(out.shape[0], STFT_CFG["n_fft"] // 2 + 1)
        self.assertEqual(out.dtype, np.float32)

    def test_hop_length_drives_frame_rate(self):
        """Halving the hop must double the frames — the property the fixed
        rate slicer assumes when it converts ms to frame indices."""
        coarse = REGISTRY["logmel"].extract(TEST_AUDIO, **{**LOGMEL_CFG, "hop_length": 640})
        fine = REGISTRY["logmel"].extract(TEST_AUDIO, **{**LOGMEL_CFG, "hop_length": 320})
        self.assertAlmostEqual(fine.shape[1] / coarse.shape[1], 2.0, delta=0.05)

    def test_missing_audio_raises(self):
        for name, cfg in [("mfcc", MFCC_CFG), ("logmel", LOGMEL_CFG), ("stft", STFT_CFG)]:
            with self.subTest(name), self.assertRaises(FileNotFoundError):
                REGISTRY[name].extract(BASE_DATA_PATH / "audio" / "nope.wav", **cfg)

    def test_invalid_n_fft_raises(self):
        with self.assertRaises(ValueError):
            REGISTRY["mfcc"].extract(TEST_AUDIO, **{**MFCC_CFG, "n_fft": -1})


class TestFixedRateSlice(unittest.TestCase):

    def test_width_depends_only_on_duration(self):
        """Every 2000 ms chunk must come out the same width regardless of
        where it starts — this is what np.vstack downstream relies on."""
        array = np.arange(40 * 500, dtype=np.float32).reshape(40, 500)
        cfg = {"hop_length": 320}
        widths = {
            fixed_rate_slice(array, start, start + 2000, cfg).shape[1]
            for start in range(0, 4000, 250)
        }
        self.assertEqual(widths, {100})

    def test_uses_nominal_rate_not_per_file_average(self):
        """A file with one extra frame (librosa's constant +1 offset) must
        not shift the slice — the bug that broke augmented-vs-real chunks."""
        cfg = {"hop_length": 320}
        short = np.arange(40 * 100, dtype=np.float32).reshape(40, 100)
        long = np.arange(40 * 101, dtype=np.float32).reshape(40, 101)
        self.assertEqual(fixed_rate_slice(short, 0, 2000, cfg).shape, (40, 100))
        self.assertEqual(fixed_rate_slice(long, 0, 2000, cfg).shape, (40, 100))

    def test_start_offset_maps_to_frame_index(self):
        array = np.tile(np.arange(500, dtype=np.float32), (40, 1))
        cfg = {"hop_length": 320}  # 0.05 frames per ms
        sliced = fixed_rate_slice(array, 2000, 4000, cfg)
        self.assertEqual(sliced[0, 0], 100.0)


class TestYamnetSlice(unittest.TestCase):

    def test_pools_time_axis_to_fixed_length(self):
        """3-frame and 4-frame windows must both yield one 1024-vector, the
        reason this type pools instead of slicing at a fixed rate."""
        embeddings = np.random.default_rng(0).normal(size=(1024, 30))
        pooled = yamnet_slice(embeddings, 0, 2000, {})
        self.assertEqual(pooled.shape, (1024,))

    def test_past_last_frame_falls_back_to_final_frame(self):
        embeddings = np.random.default_rng(0).normal(size=(1024, 2))
        pooled = yamnet_slice(embeddings, 10_000, 12_000, {})
        self.assertEqual(pooled.shape, (1024,))
        np.testing.assert_allclose(pooled, embeddings[:, -1])

    def test_declared_as_time_pooling(self):
        # train_model.py rejects the cnn head on this basis
        self.assertFalse(REGISTRY["yamnet"].keeps_time_axis)
        self.assertTrue(REGISTRY["logmel"].keeps_time_axis)


class TestExtractorConfig(unittest.TestCase):

    def test_excludes_features_dir(self):
        """Where arrays are written doesn't change what's in them, so moving
        the directory must not invalidate a cached extraction."""
        config = extractor_config(
            {"type": "logmel", "features_dir": "./data/anywhere", **LOGMEL_CFG}
        )
        self.assertNotIn("features_dir", config)
        self.assertEqual(config["type"], "logmel")
        self.assertEqual(set(config) - {"type"}, set(LOGMEL_CFG))

    def test_ignores_params_belonging_to_other_types(self):
        config = extractor_config(
            {"type": "stft", "features_dir": "x", **STFT_CFG, "n_mels": 40}
        )
        self.assertNotIn("n_mels", config)

    def test_missing_param_is_reported(self):
        with self.assertRaises(SystemExit) as ctx:
            extractor_config({"type": "logmel", "n_mels": 40})
        self.assertIn("n_fft", str(ctx.exception))

    def test_unknown_type_is_reported(self):
        with self.assertRaises(SystemExit) as ctx:
            get_spec("cepstrum")
        self.assertIn("cepstrum", str(ctx.exception))


class TestFeatureConfigGuard(unittest.TestCase):
    """The guard against data/logmel_data.esp32bak: two configs writing
    different geometry into one directory name."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.features_dir = Path(self._tmp.name)
        self.params = {"type": "logmel", "features_dir": str(self.features_dir), **LOGMEL_CFG}

    def tearDown(self):
        self._tmp.cleanup()

    def test_roundtrip_passes(self):
        write_feature_config(self.features_dir, self.params)
        self.assertEqual(
            verify_feature_config(self.features_dir, self.params),
            extractor_config(self.params),
        )

    def test_changed_geometry_is_rejected(self):
        write_feature_config(self.features_dir, self.params)
        with self.assertRaises(SystemExit) as ctx:
            verify_feature_config(
                self.features_dir, {**self.params, "n_mels": 13, "hop_length": 512}
            )
        message = str(ctx.exception)
        self.assertIn("n_mels", message)
        self.assertIn("hop_length", message)

    def test_changed_type_is_rejected(self):
        write_feature_config(self.features_dir, self.params)
        with self.assertRaises(SystemExit):
            verify_feature_config(
                self.features_dir, {"type": "stft", "features_dir": "x", **STFT_CFG}
            )

    def test_moving_features_dir_alone_is_accepted(self):
        write_feature_config(self.features_dir, self.params)
        verify_feature_config(
            self.features_dir, {**self.params, "features_dir": "./somewhere/else"}
        )

    def test_missing_sidecar_is_rejected(self):
        with self.assertRaises(SystemExit) as ctx:
            verify_feature_config(self.features_dir, self.params)
        self.assertIn("extract_features", str(ctx.exception))

    def test_sidecar_is_valid_json(self):
        write_feature_config(self.features_dir, self.params)
        written = json.loads((self.features_dir / "feature_config.json").read_text())
        self.assertEqual(written["n_mels"], 40)


class TestLiveParams(unittest.TestCase):
    """params.yaml must describe a variant the registry can actually run."""

    def setUp(self):
        with open(PARAMS_PATH) as f:
            self.params = yaml.safe_load(f)

    def test_configured_type_is_fully_specified(self):
        config = extractor_config(self.params["feature_extraction"])
        self.assertEqual(config["type"], self.params["feature_extraction"]["type"])

    def test_chunk_size_yields_whole_frames(self):
        feature_params = self.params["feature_extraction"]
        if not REGISTRY[feature_params["type"]].slice_chunk is fixed_rate_slice:
            self.skipTest("configured type does not slice at a fixed rate")
        frames = 16000 / feature_params["hop_length"] / 1000 * self.params["chunk_size"]
        self.assertEqual(frames, int(frames), "chunk_size must be a whole frame count")

    def test_head_and_feature_type_are_compatible(self):
        from src.train_model import check_compatible

        check_compatible(
            self.params["training"]["head"], self.params["feature_extraction"]["type"]
        )


if __name__ == "__main__":
    unittest.main()


class TestLogCompression(unittest.TestCase):
    """Whether the trainer must log-compress is a property of the
    representation, not a params flag someone has to flip per branch."""

    def test_only_stft_needs_log_compression(self):
        needing = {n for n, s in REGISTRY.items() if s.needs_log_compression}
        self.assertEqual(needing, {"stft"})

    def test_log_domain_extractors_do_not_double_log(self):
        # logmel already applies the log, so re-logging would be wrong
        self.assertFalse(REGISTRY["logmel"].needs_log_compression)
        self.assertFalse(REGISTRY["mfcc"].needs_log_compression)

    def test_yamnet_is_not_log_compressed(self):
        """YAMNet embeddings go negative; np.log would produce NaN."""
        self.assertFalse(REGISTRY["yamnet"].needs_log_compression)


class TestPerHeadParams(unittest.TestCase):
    """Regression: a single flat `model:` block fed both heads, so XGBoost
    got the CNN's learning_rate and trained to val F1 0.0."""

    def setUp(self):
        with open(PARAMS_PATH) as f:
            self.params = yaml.safe_load(f)

    def test_each_head_has_its_own_block(self):
        for head in ("cnn", "xgboost"):
            self.assertIn(head, self.params["model"], f"missing model.{head}")

    def test_blocks_do_not_leak_foreign_hyperparameters(self):
        cnn, xgboost = self.params["model"]["cnn"], self.params["model"]["xgboost"]
        for key in ("dropout", "epochs", "batch_size", "early_stopping_patience"):
            self.assertNotIn(key, xgboost, f"{key} is not an XGBoost parameter")
        for key in ("n_estimators", "max_depth", "eval_metric"):
            self.assertNotIn(key, cnn, f"{key} is not a Keras parameter")

    def test_xgboost_learning_rate_is_tree_scaled(self):
        """0.001 (the CNN's) against xgboost's few estimators does not learn."""
        self.assertGreaterEqual(self.params["model"]["xgboost"]["learning_rate"], 0.01)

    def test_random_state_is_shared_not_per_head(self):
        self.assertIn("random_state", self.params["training"])
        for head in ("cnn", "xgboost"):
            self.assertNotIn("random_state", self.params["model"][head])
