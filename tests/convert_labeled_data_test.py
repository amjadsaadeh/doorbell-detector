import unittest
from pathlib import Path
import pandas as pd
from convert_labeled_data import (
    annotation_to_sample_per_row,
    derive_audio_file_name,
    normalize_audio_uri,
)
from download_audio import parse_s3_uri


BASE_DATA_PATH = Path(__file__).parent / 'data'


class TestConvertLabeledData(unittest.TestCase):

    def test_annotation_to_sample_per_row(self):
        df = pd.read_csv(BASE_DATA_PATH / 'labeled_data.csv')
        gt_df = pd.read_csv(BASE_DATA_PATH / 'annotation_per_row_data.csv')  # Data to test against
        converted_df = annotation_to_sample_per_row(df)

        for series_name, series in converted_df.items():
            self.assertIn(series_name, gt_df.columns, f'Column {series_name} not found in reference DataFrame')
            pd.testing.assert_series_equal(series, gt_df[series_name], f'Column {series_name} is not equal')


class TestNormalizeAudioUri(unittest.TestCase):

    def test_s3_uri_passthrough(self):
        uri = 's3://bell-detector-data/raw_data/background_1.wav'
        self.assertEqual(normalize_audio_uri(uri), uri)

    def test_presigned_path_style_url(self):
        uri = ('https://minio.example.com/bell-detector-data/raw_data/background_1.wav'
               '?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Expires=3600&X-Amz-Signature=abc')
        self.assertEqual(
            normalize_audio_uri(uri),
            's3://bell-detector-data/raw_data/background_1.wav',
        )

    def test_presigned_virtual_hosted_url(self):
        uri = ('https://bell-detector-data.s3.eu-central-1.amazonaws.com/raw_data/background_1.wav'
               '?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=abc')
        self.assertEqual(
            normalize_audio_uri(uri),
            's3://bell-detector-data/raw_data/background_1.wav',
        )

    def test_label_studio_resolver_path(self):
        uri = '/data/s3/?fileuri=s3://bell-detector-data/raw_data/background_1.wav'
        self.assertEqual(
            normalize_audio_uri(uri),
            's3://bell-detector-data/raw_data/background_1.wav',
        )

    def test_unrecognized_reference_raises(self):
        with self.assertRaises(ValueError):
            normalize_audio_uri('/data/local-files/?d=bell_detector_data/raw_data/background_1.wav')

    def test_incomplete_s3_uri_raises(self):
        with self.assertRaises(ValueError):
            normalize_audio_uri('s3://bucket-without-key')

    def test_derive_audio_file_name(self):
        self.assertEqual(
            derive_audio_file_name('s3://bell-detector-data/raw_data/background_1.wav'),
            'background_1.wav',
        )


class TestParseS3Uri(unittest.TestCase):

    def test_bucket_and_key(self):
        self.assertEqual(
            parse_s3_uri('s3://bell-detector-data/raw_data/background_1.wav'),
            ('bell-detector-data', 'raw_data/background_1.wav'),
        )

    def test_non_s3_uri_raises(self):
        with self.assertRaises(ValueError):
            parse_s3_uri('https://example.com/bucket/key.wav')


if __name__ == '__main__':
    unittest.main()
    