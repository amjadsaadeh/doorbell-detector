import unittest
from pathlib import Path
import numpy as np
from src.extract_stft_features import process_audio_data, extract_stft_features


BASE_DATA_PATH = Path(__file__).parent / 'data'

# scipy.signal.stft default nperseg=256 -> nperseg // 2 + 1 frequency bins
EXPECTED_FREQ_BINS = 129


class TestExtractSTFTFeatures(unittest.TestCase):

    def setUp(self):
        self.input_path = BASE_DATA_PATH / 'audio'
        self.output_path = BASE_DATA_PATH / 'tmp_stft_data'
        self.output_path.mkdir(exist_ok=True)

    def tearDown(self):
        if self.output_path.exists():
            for file in self.output_path.glob('*'):
                file.unlink()
            self.output_path.rmdir()

    def test_process_audio_data(self):

        # Check if the output file is created
        self.assertTrue(self.output_path.exists(), "Output file was not created")

        process_audio_data(self.input_path, self.output_path)

        self.assertEqual(len(list(self.output_path.glob('*.npy'))), len(list(self.input_path.glob('*.wav'))), "Number of output files does not match the number of input files")

    def test_extract_stft_features(self):
        spectrogram = extract_stft_features(self.input_path / 'test_audio.wav')
        self.assertIsInstance(spectrogram, np.ndarray, "STFT features are not returned as a numpy array")
        self.assertEqual(spectrogram.shape[0], EXPECTED_FREQ_BINS, "Number of frequency bins does not match the expected value")
        self.assertFalse(np.iscomplexobj(spectrogram), "Spectrogram should be a real-valued magnitude, not complex")

    def test_invalid_audio_path(self):
        with self.assertRaises(FileNotFoundError):
            extract_stft_features(self.input_path / 'invalid_audio.wav')


if __name__ == '__main__':
    unittest.main()
