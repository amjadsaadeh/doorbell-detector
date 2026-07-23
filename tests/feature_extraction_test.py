import unittest
from pathlib import Path
import pandas as pd
import numpy as np
from src.extract_mfcc_features import process_audio_data, extract_mfcc_features
from src.extract_yamnet_features import MIN_SAMPLES, ensure_min_length


class TestEnsureMinLength(unittest.TestCase):

    def test_short_waveform_is_zero_padded(self):
        waveform = np.ones(100, dtype=np.float32)
        padded = ensure_min_length(waveform)
        self.assertEqual(len(padded), MIN_SAMPLES)
        np.testing.assert_array_equal(padded[:100], waveform)
        self.assertEqual(padded[100:].sum(), 0)

    def test_long_waveform_is_untouched(self):
        waveform = np.ones(MIN_SAMPLES + 1, dtype=np.float32)
        self.assertIs(ensure_min_length(waveform), waveform)


BASE_DATA_PATH = Path(__file__).parent / 'data'


class TestExtractMFCCFeatures(unittest.TestCase):

    def setUp(self):
        self.params = {
            'feature_extraction': {
                'n_mfcc': 13,
                'n_fft': 512
            }
        }
        self.input_path = BASE_DATA_PATH / 'audio'
        self.output_path = BASE_DATA_PATH / 'tmp_mfcc_data'
        self.output_path.mkdir(exist_ok=True)

    def tearDown(self):
        if self.output_path.exists():
            for file in self.output_path.glob('*'):
                file.unlink()
            self.output_path.rmdir()

    def test_process_audio_data(self):

        # Check if the output file is created
        self.assertTrue(self.output_path.exists(), "Output file was not created")

        process_audio_data(self.params, self.input_path, self.output_path)

        self.assertEqual(len(list(self.output_path.glob('*.npy'))), len(list(self.input_path.glob('*.wav'))), "Number of output files does not match the number of input files")

    def test_extract_mfcc_features(self):
        mfccs = extract_mfcc_features(self.input_path / 'test_audio.wav', 13, 512)
        self.assertIsInstance(mfccs, np.ndarray, "MFCC features are not returned as a numpy array")
        self.assertEqual(mfccs.shape[0], 13, "Number of MFCC features does not match the expected value")

    def test_invalid_audio_path(self):
        with self.assertRaises(FileNotFoundError):
            extract_mfcc_features(self.input_path / 'invalid_audio.wav', 13, 512)

    def test_invalid_params(self):
        with self.assertRaises(ValueError):
            extract_mfcc_features(self.input_path / 'test_audio.wav', 13, -1)


if __name__ == '__main__':
    unittest.main()
