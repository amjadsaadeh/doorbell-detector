import unittest

import numpy as np

from src.train_cnn import build_model, normalization_stats


class TestNormalizationStats(unittest.TestCase):

    def test_per_bin_stats_broadcast_shape(self):
        X = np.random.default_rng(0).normal(5.0, 2.0, size=(10, 13, 62, 1)).astype(np.float32)
        mean, std = normalization_stats(X)
        self.assertEqual(mean.shape, (13, 1, 1))
        self.assertEqual(std.shape, (13, 1, 1))
        normalized = (X - mean) / std
        self.assertAlmostEqual(float(normalized.mean()), 0.0, places=4)
        self.assertAlmostEqual(float(normalized.std()), 1.0, places=3)

    def test_zero_variance_bin_does_not_divide_by_zero(self):
        X = np.ones((4, 3, 5, 1), dtype=np.float32)
        _, std = normalization_stats(X)
        self.assertTrue((std > 0).all())


class TestBuildModel(unittest.TestCase):

    def test_binary_output_and_param_budget(self):
        for input_shape in [(13, 62, 1), (129, 250, 1)]:
            model = build_model(input_shape, dropout=0.3)
            # GlobalAveragePooling makes the parameter count independent of
            # the input size — same small model for MFCC and spectrogram
            self.assertLess(model.count_params(), 200_000)
            out = model.predict(np.zeros((2, *input_shape), dtype=np.float32), verbose=0)
            self.assertEqual(out.shape, (2, 1))
            self.assertTrue(((out >= 0) & (out <= 1)).all())


if __name__ == "__main__":
    unittest.main()
