# Testing Patterns

**Analysis Date:** 2026-05-26

## Test Framework

**Runner:**
- `unittest` (Python stdlib) — the only configured test framework
- `pytest==8.3.3` is present in `requirements.txt` but is NOT enabled in VS Code settings (`.vscode/settings.json` sets `"python.testing.pytestEnabled": false`)
- Config: `.vscode/settings.json` (VS Code runner), no `pytest.ini` or `conftest.py`

**Assertion Library:**
- `unittest.TestCase` assertion methods: `assertEqual`, `assertIsInstance`, `assertIn`, `assertRaises`, `assertTrue`
- `pandas.testing.assert_series_equal` for DataFrame column comparison

**Run Commands:**
```bash
# From project root, run all tests matching *_test.py in ./tests/
python -m unittest discover -v -s ./tests -p "*_test.py"

# Run a single test file directly
python -m pytest tests/feature_extraction_test.py        # if pytest preferred
python -m unittest tests/convert_labeled_data_test.py

# VS Code "Run Tests" uses the unittest discover command above
```

Note: `feature_extraction_test.py` imports from `src.extract_mfcc_features` using the package prefix, so it must be run from the project root. `convert_labeled_data_test.py` imports `convert_labeled_data` directly (no `src.` prefix), which requires `src/` to be on `PYTHONPATH`. The `.vscode/settings.json` sources `.env` for the environment file, which presumably sets this up.

## Test File Organization

**Location:**
- All tests are in a top-level `tests/` directory — separate from source, not co-located with modules

**Naming:**
- Pattern: `<module_name>_test.py` (suffix convention, not prefix)
- `tests/convert_labeled_data_test.py` — tests `src/convert_labeled_data.py`
- `tests/feature_extraction_test.py` — tests `src/extract_mfcc_features.py`

**Structure:**
```
tests/
├── convert_labeled_data_test.py
├── feature_extraction_test.py
└── data/
    ├── labeled_data.csv              # input fixture for convert_labeled_data tests
    ├── annotation_per_row_data.csv   # ground-truth fixture for convert_labeled_data tests
    ├── audio/
    │   ├── test_audio.wav
    │   ├── test_audio2.wav
    │   └── test_audio3.wav
    └── tmp_mfcc_data/                # temporary output dir created/deleted by tearDown
        ├── test_audio2.npy
        └── test_audio3.npy
```

## Test Structure

**Suite Organization:**
```python
import unittest
from pathlib import Path
import pandas as pd
from convert_labeled_data import annotation_to_sample_per_row

BASE_DATA_PATH = Path(__file__).parent / 'data'

class TestConvertLabeledData(unittest.TestCase):

    def test_annotation_to_sample_per_row(self):
        df = pd.read_csv(BASE_DATA_PATH / 'labeled_data.csv')
        gt_df = pd.read_csv(BASE_DATA_PATH / 'annotation_per_row_data.csv')
        converted_df = annotation_to_sample_per_row(df)

        for series_name, series in converted_df.items():
            self.assertIn(series_name, gt_df.columns, ...)
            pd.testing.assert_series_equal(series, gt_df[series_name], ...)

if __name__ == '__main__':
    unittest.main()
```

**Patterns:**
- `Path(__file__).parent / 'data'` used to resolve fixture paths relative to the test file — portable across environments
- `setUp` / `tearDown` used in `TestExtractMFCCFeatures` to create and clean up the temporary output directory:
  ```python
  def setUp(self):
      self.params = {'feature_extraction': {'n_mfcc': 13, 'n_fft': 512}}
      self.input_path = BASE_DATA_PATH / 'audio'
      self.output_path = BASE_DATA_PATH / 'tmp_mfcc_data'
      self.output_path.mkdir(exist_ok=True)

  def tearDown(self):
      if self.output_path.exists():
          for file in self.output_path.glob('*'):
              file.unlink()
          self.output_path.rmdir()
  ```
- Each `TestCase` class maps to one source module

## Mocking

**Framework:** None — no `unittest.mock`, `pytest-mock`, or other mocking library is used in the existing tests.

**What is tested:** Only pure data-transformation functions that take files/DataFrames as inputs and produce outputs. Functions with side effects (audio I/O, MQTT, GPIO, model inference) are not mocked and not tested.

## Fixtures and Test Data

**Test Data location:** `tests/data/`

**Fixtures used:**

| File | Used by | Purpose |
|------|---------|---------|
| `tests/data/labeled_data.csv` | `convert_labeled_data_test.py` | Input: Label Studio CSV export |
| `tests/data/annotation_per_row_data.csv` | `convert_labeled_data_test.py` | Ground-truth: expected output after conversion |
| `tests/data/audio/test_audio.wav` | `feature_extraction_test.py` | Real WAV file for MFCC extraction |
| `tests/data/audio/test_audio2.wav` | `feature_extraction_test.py` | Additional WAV for batch processing test |
| `tests/data/audio/test_audio3.wav` | `feature_extraction_test.py` | Additional WAV for batch processing test |
| `tests/data/tmp_mfcc_data/` | `feature_extraction_test.py` | Temporary output; created in setUp, deleted in tearDown |

Fixtures are real data files (not generated inline). Ground-truth CSV files serve as regression anchors.

## Coverage

**Requirements:** None enforced — no coverage configuration or threshold exists.

**Coverage tool:** Not configured (no `.coveragerc`, no `pytest-cov` invocation).

**Approximate functional coverage:**
- `src/convert_labeled_data.py`: `annotation_to_sample_per_row()` is tested — the `if __name__` block is not
- `src/extract_mfcc_features.py`: `extract_mfcc_features()` and `process_audio_data()` are tested — the `if __name__` block is not
- `src/dataqualityutils.py`: **not tested**
- `src/train_xgboost.py`: **not tested**
- `src/draw_data.py`: **not tested**
- `src/download_audio.py`: **not tested**
- `src/extract_data_quality.py`: **not tested**
- `data_collection/data_collector.py`: **not tested** (entire runtime daemon is untested)

## Test Types

**Unit Tests:**
- `tests/convert_labeled_data_test.py` — tests a pure data-transformation function with fixture CSV files
- `tests/feature_extraction_test.py` — tests MFCC feature extraction with real WAV files; also validates error cases (invalid file path, invalid parameter values)

**Integration Tests:** None present.

**E2E Tests:** None present.

**ML/Model Tests:** None present (no tests for training correctness, model output shape, or inference behavior).

## Notable Gaps in Testing

1. **`data_collector.py` is entirely untested.** The entire runtime daemon — ring buffer management, MQTT trigger handling, GPIO interrupt handling, ML inference loop, audio stream recovery — has zero test coverage. This is the most critical runtime component.

2. **`src/train_xgboost.py` is untested.** No tests verify that the training pipeline produces a valid model, that metrics are within expected ranges, or that the `prepare_data()` function handles edge cases correctly.

3. **`src/draw_data.py` is untested.** The data balancing logic (`draw_data.py`) is complex (chunking, random sampling, MFCC feature slicing) and has no tests.

4. **`src/dataqualityutils.py` is untested.** Despite being a shared utility imported by two pipeline scripts, `get_data_quality_metrics()` has no tests.

5. **No mocking for external dependencies.** Tests that exercise audio processing (`feature_extraction_test.py`) require real WAV files and spawn multiprocessing pools. If the test environment lacks audio codecs, tests would fail. MQTT and GPIO paths cannot be tested without hardware mocking.

6. **No CI pipeline.** No `.github/workflows/`, `Makefile`, or other CI configuration exists. Tests are only run manually or via VS Code.

7. **Import path inconsistency between test files.** `feature_extraction_test.py` uses `from src.extract_mfcc_features import ...` while `convert_labeled_data_test.py` uses `from convert_labeled_data import ...` — requiring `src/` on `PYTHONPATH` for the latter. This makes test invocation fragile without the `.env` file setting up the path.

---

*Testing analysis: 2026-05-26*
