import unittest
from pathlib import Path

import numpy as np
import yaml

from src.extract_logmel_features import extract_logmel_features, process_audio_data


BASE_DATA_PATH = Path(__file__).parent / 'data'
PARAMS_PATH = Path(__file__).parent.parent / 'params.yaml'

# The whole point of this branch: the geometry has to stay MCU-sized. A
# regression here (more bins, smaller hop) silently blows the ESP32-S3
# activation budget, so pin it in a test rather than only in params.yaml.
EXPECTED_MEL_BINS = 40
EXPECTED_FRAMES_PER_2S_CHUNK = 100
MAX_PEAK_ACTIVATION_BYTES = 256 * 1024  # first conv layer, int8, 32 filters


class TestExtractLogMelFeatures(unittest.TestCase):

    def setUp(self):
        self.input_path = BASE_DATA_PATH / 'audio'
        self.output_path = BASE_DATA_PATH / 'tmp_logmel_data'
        self.output_path.mkdir(exist_ok=True)
        with open(PARAMS_PATH) as f:
            self.params = yaml.safe_load(f)

    def tearDown(self):
        if self.output_path.exists():
            for file in self.output_path.glob('*'):
                file.unlink()
            self.output_path.rmdir()

    def test_process_audio_data(self):
        process_audio_data(self.params, self.input_path, self.output_path)

        self.assertEqual(
            len(list(self.output_path.glob('*.npy'))),
            len(list(self.input_path.glob('*.wav'))),
            "Number of output files does not match the number of input files",
        )

    def test_extract_logmel_features(self):
        logmel = extract_logmel_features(self.input_path / 'test_audio.wav')
        self.assertIsInstance(logmel, np.ndarray, "Features are not returned as a numpy array")
        self.assertEqual(logmel.shape[0], EXPECTED_MEL_BINS, "Number of mel bins does not match")
        self.assertEqual(logmel.dtype, np.float32, "Features must be float32 for the tflite path")
        self.assertTrue(np.isfinite(logmel).all(), "log_offset must keep silent frames finite")

    def test_chunk_geometry_fits_mcu_budget(self):
        """2000 ms must land on exactly 100 frames, and the resulting first
        conv activation must fit the ESP32-S3 internal SRAM budget."""
        feature_params = self.params['feature_extraction']
        frames_per_ms = 16000 / feature_params['hop_length'] / 1000
        frames = int(frames_per_ms * self.params['chunk_size'])

        self.assertEqual(frames, EXPECTED_FRAMES_PER_2S_CHUNK)

        peak_activation = feature_params['n_mels'] * frames * 32  # int8, 32 filters
        self.assertLess(peak_activation, MAX_PEAK_ACTIVATION_BYTES)

    def test_extracted_file_covers_the_chunk_slice(self):
        """draw_data.py slices [0:100] off each file at the fixed frame rate;
        an off-by-one in the extractor would silently produce short chunks."""
        logmel = extract_logmel_features(self.input_path / 'test_audio.wav')
        self.assertGreaterEqual(logmel.shape[1], EXPECTED_FRAMES_PER_2S_CHUNK)

    def test_invalid_audio_path(self):
        with self.assertRaises(FileNotFoundError):
            extract_logmel_features(self.input_path / 'invalid_audio.wav')


if __name__ == '__main__':
    unittest.main()
