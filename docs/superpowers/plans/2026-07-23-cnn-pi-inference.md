# CNN-on-Pi Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the `cnn-spectrogram` model (12.4k-param depthwise-separable CNN, val F1 0.9826) on the Pi Zero W as a second-stage verifier behind the existing cross-correlation detector — via exported weights and a pure-numpy forward pass, since no TF/tflite runtime exists for ARMv6.

**Architecture:** A new Pi-side module `data_collection/cnn_inference.py` (numpy + scipy only) implements preprocessing (scipy STFT → log → per-bin normalize) and the CNN forward pass from a single exported `.npz` (BatchNorm folded into conv weights at export time). A dev-side export script produces that `.npz` from `models/cnn_model.keras` + `models/cnn_normalization.npz`, with a Keras-parity test guarding numerical correctness. An on-Pi benchmark measures real latency and gates the final step: wiring the verifier into `detector.py` so cross-correlation triggers are confirmed or suppressed by the CNN.

**Tech Stack:** numpy (`sliding_window_view`, BLAS matmul), `scipy.signal.stft`; TensorFlow/Keras only on the dev machine (export + parity tests); ssh/rsync to `pi@mic` for deployment.

## Global Constraints

- Work on branch **`cnn-spectrogram`** (the model being deployed lives here).
- `data_collection/` code must import only stdlib + numpy + scipy (+ pyaudio/soundfile/paho already used there). **Never tensorflow, librosa, or xgboost** — none install on ARMv6l.
- Audio contract: 16 kHz, mono, int16 (`CHANNELS=1`, `RATE=16000` in `detector.py`) — do not change.
- **Train/serve parity:** the training pipeline feeds `scipy.signal.stft` with *raw int16 values as float32* (see `src/extract_stft_features.py`: `np.array(audio.get_array_of_samples(), dtype=np.float32)`) — NOT normalized to [-1, 1]. Inference must do the same (`.astype(np.float32)`, no `/32768`).
- STFT parameters are scipy defaults at fs=16000: `nperseg=256, noverlap=128` (hop 128 samples). One 2 s window (32000 samples) → 251 frames; the pipeline uses the first **250** (`FRAMES_PER_MS = 0.125` in `src/draw_data.py`). Spectrogram chunk shape: **(129, 250)**.
- Preprocessing after STFT magnitude, in order: `log(x + 1e-6)`, then `(x - mean) / std` with the per-bin stats from `models/cnn_normalization.npz` (`params.yaml` has `model.log_compress: true` on this branch).
- Tests are named `tests/<name>_test.py` and run with `PYTHONPATH=./src:. uv run pytest tests/`.
- Dev-machine commands use `uv run`. Pi commands use `/home/pi/doorbell-detector/data_collection/.venv/bin/python` over `ssh pi@mic` (passwordless sudo available; deployed env file at `/etc/doorbell-detector.env` uses the typo'd var name `TEMPLATE_SCORE_THREHOLD` — keep that typo when editing the *deployed* file).
- Commit after every task; commit messages end with the project's Co-Authored-By/Claude-Session trailer used on this branch.

## Model architecture being reimplemented (from `src/train_cnn.py::build_model`)

Input `(129, 250, 1)` →
`Conv2D(32, 3×3, same, no bias)` → BN → ReLU → `MaxPool 2×2 (valid)` →
`SeparableConv2D(64, 3×3, same, no bias)` (depthwise 3×3 then pointwise 1×1, no activation between) → BN → ReLU → `MaxPool 2×2` →
`SeparableConv2D(128, 3×3, same, no bias)` → BN → ReLU →
`GlobalAveragePooling2D` → Dropout (identity at inference) → `Dense(1, sigmoid)`.

Spatial sizes: (129,250) → pool → (64,125) → pool → (32,62). Keras BN epsilon is `1e-3` (read `layer.epsilon`, don't hardcode).

**BN folding** (BN directly after a bias-free conv, per output channel):
`scale = gamma / sqrt(var + eps)`; `w' = w * scale` (broadcast over the *output-channel* axis, which is the last kernel axis); `b' = beta - mean * scale`. For SeparableConv2D the BN follows the pointwise kernel — fold into the pointwise weights; the depthwise kernel `(3,3,C,1)` is squeezed to `(3,3,C)` unchanged.

**Exported npz keys** (single file `models/cnn_weights_pi.npz`):
`conv1_w (3,3,1,32)`, `conv1_b (32,)`, `dw2_w (3,3,32)`, `pw2_w (32,64)`, `pw2_b (64,)`, `dw3_w (3,3,64)`, `pw3_w (64,128)`, `pw3_b (128,)`, `dense_w (128,1)`, `dense_b (1,)`, `norm_mean (129,1)`, `norm_std (129,1)`.

---

### Task 1: Pure-numpy CNN layers + forward pass

**Files:**
- Create: `data_collection/cnn_inference.py`
- Test: `tests/cnn_inference_test.py`

**Interfaces:**
- Produces: `conv2d_same(x, w, b) -> np.ndarray` (x `(H,W,Cin)` float32, w `(3,3,Cin,Cout)`, b `(Cout,)` → `(H,W,Cout)`); `depthwise_conv2d_same(x, w) -> np.ndarray` (w `(3,3,C)` → `(H,W,C)`); `maxpool2(x) -> np.ndarray` (floor-halved spatial dims); `forward(spec, weights) -> float` (spec `(129,250)` already preprocessed; weights = dict with the npz keys above; returns sigmoid probability). Module constants `N_BINS = 129`, `N_FRAMES = 250`, `WINDOW_SAMPLES = 32000`.
- Consumes: nothing (first task).

- [ ] **Step 1: Write the failing tests**

Check how `tests/detector_test.py` imports the detector module and use the same style for `data_collection` imports (the code below assumes `from data_collection.cnn_inference import ...`, which works as a namespace package with `.` on `PYTHONPATH`).

```python
# tests/cnn_inference_test.py
import unittest

import numpy as np

from data_collection.cnn_inference import (
    conv2d_same,
    depthwise_conv2d_same,
    forward,
    maxpool2,
)


def identity_kernel(c_in, c_out):
    """3x3 kernel that copies channel i of the input to output channel i."""
    w = np.zeros((3, 3, c_in, c_out), dtype=np.float32)
    for c in range(min(c_in, c_out)):
        w[1, 1, c, c] = 1.0
    return w


class TestConv2dSame(unittest.TestCase):

    def test_identity_kernel_reproduces_input(self):
        x = np.random.default_rng(0).normal(size=(5, 7, 2)).astype(np.float32)
        out = conv2d_same(x, identity_kernel(2, 2), np.zeros(2, dtype=np.float32))
        np.testing.assert_allclose(out, x, atol=1e-6)

    def test_averaging_kernel_zero_padding(self):
        # all-ones 3x3 kernel on all-ones 3x3 input: center sees 9 ones,
        # corners see 4 (zero padding), edges see 6
        x = np.ones((3, 3, 1), dtype=np.float32)
        w = np.ones((3, 3, 1, 1), dtype=np.float32)
        out = conv2d_same(x, w, np.zeros(1, dtype=np.float32))[:, :, 0]
        expected = np.array([[4, 6, 4], [6, 9, 6], [4, 6, 4]], dtype=np.float32)
        np.testing.assert_allclose(out, expected)

    def test_bias_is_added(self):
        x = np.zeros((2, 2, 1), dtype=np.float32)
        w = np.zeros((3, 3, 1, 4), dtype=np.float32)
        out = conv2d_same(x, w, np.array([1, 2, 3, 4], dtype=np.float32))
        np.testing.assert_allclose(out[0, 0], [1, 2, 3, 4])


class TestDepthwiseConv2dSame(unittest.TestCase):

    def test_channels_stay_independent(self):
        x = np.random.default_rng(1).normal(size=(4, 4, 2)).astype(np.float32)
        w = np.zeros((3, 3, 2), dtype=np.float32)
        w[1, 1, 0] = 2.0  # channel 0: doubled; channel 1: zeroed
        out = depthwise_conv2d_same(x, w)
        np.testing.assert_allclose(out[:, :, 0], 2.0 * x[:, :, 0], atol=1e-6)
        np.testing.assert_allclose(out[:, :, 1], 0.0)


class TestMaxpool2(unittest.TestCase):

    def test_odd_dims_floor_like_keras_valid_padding(self):
        x = np.arange(5 * 7, dtype=np.float32).reshape(5, 7, 1)
        out = maxpool2(x)
        self.assertEqual(out.shape, (2, 3, 1))
        # window (0:2, 0:2) max = row1 col1 = 8
        self.assertEqual(out[0, 0, 0], 8.0)


class TestForward(unittest.TestCase):

    def _tiny_weights(self):
        rng = np.random.default_rng(42)
        return {
            "conv1_w": rng.normal(0, 0.1, (3, 3, 1, 32)).astype(np.float32),
            "conv1_b": np.zeros(32, dtype=np.float32),
            "dw2_w": rng.normal(0, 0.1, (3, 3, 32)).astype(np.float32),
            "pw2_w": rng.normal(0, 0.1, (32, 64)).astype(np.float32),
            "pw2_b": np.zeros(64, dtype=np.float32),
            "dw3_w": rng.normal(0, 0.1, (3, 3, 64)).astype(np.float32),
            "pw3_w": rng.normal(0, 0.1, (64, 128)).astype(np.float32),
            "pw3_b": np.zeros(128, dtype=np.float32),
            "dense_w": rng.normal(0, 0.1, (128, 1)).astype(np.float32),
            "dense_b": np.zeros(1, dtype=np.float32),
        }

    def test_returns_probability(self):
        spec = np.random.default_rng(2).normal(size=(129, 250)).astype(np.float32)
        p = forward(spec, self._tiny_weights())
        self.assertIsInstance(p, float)
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)

    def test_deterministic(self):
        spec = np.random.default_rng(3).normal(size=(129, 250)).astype(np.float32)
        w = self._tiny_weights()
        self.assertEqual(forward(spec, w), forward(spec, w))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=./src:. uv run pytest tests/cnn_inference_test.py -v`
Expected: FAIL/ERROR with `ModuleNotFoundError: No module named 'data_collection.cnn_inference'`

- [ ] **Step 3: Implement the module**

```python
# data_collection/cnn_inference.py
"""Pure-numpy inference for the cnn-spectrogram doorbell model (Pi Zero W).

The Pi Zero W is ARMv6l: no TensorFlow, no tflite-runtime, no librosa.
This module reimplements the exported Keras model (see
src/export_cnn_weights.py, which folds BatchNorm into the conv weights)
using only numpy + scipy so it runs in the existing detector venv.

Train/serve parity notes (must match src/extract_stft_features.py and
src/draw_data.py exactly):
- STFT input is raw int16 sample values as float32 — NOT scaled to [-1, 1].
- scipy.signal.stft defaults at fs=16000 (nperseg=256, noverlap=128); a 2 s
  window yields 251 frames, of which the first N_FRAMES=250 are used.
- Then log(x + 1e-6) and per-frequency-bin (x - mean) / std with the stats
  exported from training.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import stft

SAMPLE_RATE = 16000
WINDOW_SAMPLES = 2 * SAMPLE_RATE  # the model scores exactly 2 s of audio
N_BINS = 129
N_FRAMES = 250
LOG_EPS = 1e-6


def conv2d_same(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """3x3 stride-1 'same' conv. x: (H, W, Cin), w: (3, 3, Cin, Cout).

    im2col + matmul so the heavy lifting happens in BLAS — the Pi has no
    SIMD-accelerated numpy, plain loops would be an order of magnitude
    slower.
    """
    h, wd = x.shape[:2]
    xp = np.pad(x, ((1, 1), (1, 1), (0, 0)))
    # (H, W, Cin, 3, 3) — window axes appended after the sliced axes
    win = sliding_window_view(xp, (3, 3), axis=(0, 1))
    cols = win.reshape(h * wd, -1)  # per pixel: Cin*3*3, ordered (Cin, 3, 3)
    wmat = w.transpose(2, 0, 1, 3).reshape(-1, w.shape[3])  # same ordering
    return (cols @ wmat).reshape(h, wd, -1) + b


def depthwise_conv2d_same(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """3x3 stride-1 'same' depthwise conv. x: (H, W, C), w: (3, 3, C)."""
    xp = np.pad(x, ((1, 1), (1, 1), (0, 0)))
    win = sliding_window_view(xp, (3, 3), axis=(0, 1))  # (H, W, C, 3, 3)
    return np.einsum("hwcij,ijc->hwc", win, w)


def maxpool2(x: np.ndarray) -> np.ndarray:
    """2x2 stride-2 max pool, 'valid' padding (odd trailing row/col dropped,
    matching keras.layers.MaxPooling2D defaults)."""
    h, wd = x.shape[0] // 2, x.shape[1] // 2
    return x[: h * 2, : wd * 2].reshape(h, 2, wd, 2, -1).max(axis=(1, 3))


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def forward(spec: np.ndarray, weights: dict) -> float:
    """Forward pass over a preprocessed (N_BINS, N_FRAMES) spectrogram.

    Mirrors src/train_cnn.py::build_model with BatchNorm pre-folded into
    the conv/pointwise weights. Pointwise convs are plain matmuls over the
    channel axis. Returns the sigmoid probability of 'bell'.
    """
    x = spec[..., np.newaxis].astype(np.float32)
    x = _relu(conv2d_same(x, weights["conv1_w"], weights["conv1_b"]))
    x = maxpool2(x)
    x = depthwise_conv2d_same(x, weights["dw2_w"])
    x = _relu(x @ weights["pw2_w"] + weights["pw2_b"])
    x = maxpool2(x)
    x = depthwise_conv2d_same(x, weights["dw3_w"])
    x = _relu(x @ weights["pw3_w"] + weights["pw3_b"])
    x = x.mean(axis=(0, 1))  # GlobalAveragePooling2D
    z = float(x @ weights["dense_w"] + weights["dense_b"])
    return float(1.0 / (1.0 + np.exp(-z)))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=./src:. uv run pytest tests/cnn_inference_test.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add data_collection/cnn_inference.py tests/cnn_inference_test.py
git commit -m "pure-numpy CNN layers + forward pass for Pi inference"
```

---

### Task 2: Preprocessing with train/serve parity

**Files:**
- Modify: `data_collection/cnn_inference.py` (append)
- Test: `tests/cnn_inference_test.py` (append)

**Interfaces:**
- Produces: `preprocess(samples_int16, weights) -> np.ndarray` (raw int16 array, ≥ `WINDOW_SAMPLES`; uses the **last** `WINDOW_SAMPLES`; returns normalized `(129, 250)` float32) and `CnnVerifier` class: `CnnVerifier(weights_path)` / `.probability(window_bytes: bytes) -> float` (raises `ValueError` if the buffer holds fewer than `WINDOW_SAMPLES` int16 samples).
- Consumes: `forward`, constants from Task 1; `src/extract_stft_features.py::extract_stft_features` (test only, as parity reference).

- [ ] **Step 1: Write the failing tests**

Append to `tests/cnn_inference_test.py` (add `import soundfile as sf`, `import tempfile`, `from pathlib import Path` at the top, plus the new imports below):

```python
from data_collection.cnn_inference import WINDOW_SAMPLES, CnnVerifier, preprocess
from src.extract_stft_features import extract_stft_features


def _fake_norm_weights():
    rng = np.random.default_rng(7)
    return {
        "norm_mean": rng.normal(0, 1, (129, 1)).astype(np.float32),
        "norm_std": rng.uniform(0.5, 2.0, (129, 1)).astype(np.float32),
    }


class TestPreprocess(unittest.TestCase):

    def test_matches_training_extraction_path(self):
        # The training pipeline: pydub reads the wav, raw int16 values as
        # float32 -> scipy stft -> magnitude; draw_data slices the first 250
        # frames; train_cnn applies log then per-bin normalization.
        # preprocess() must reproduce that end to end on the same samples.
        rng = np.random.default_rng(4)
        samples = (rng.normal(0, 3000, WINDOW_SAMPLES)).astype(np.int16)
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / "chunk.wav"
            sf.write(wav_path, samples, 16000, subtype="PCM_16")
            reference = extract_stft_features(wav_path)[:, :250]
        w = _fake_norm_weights()
        expected = (np.log(reference + 1e-6) - w["norm_mean"]) / w["norm_std"]

        result = preprocess(samples, w)

        self.assertEqual(result.shape, (129, 250))
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_uses_last_two_seconds(self):
        rng = np.random.default_rng(5)
        tail = rng.normal(0, 3000, WINDOW_SAMPLES).astype(np.int16)
        longer = np.concatenate([rng.normal(0, 3000, 8000).astype(np.int16), tail])
        w = _fake_norm_weights()
        np.testing.assert_allclose(preprocess(longer, w), preprocess(tail, w))


class TestCnnVerifier(unittest.TestCase):

    def _weights_file(self, tmp_dir):
        rng = np.random.default_rng(42)
        path = Path(tmp_dir) / "w.npz"
        np.savez(
            path,
            conv1_w=rng.normal(0, 0.1, (3, 3, 1, 32)).astype(np.float32),
            conv1_b=np.zeros(32, dtype=np.float32),
            dw2_w=rng.normal(0, 0.1, (3, 3, 32)).astype(np.float32),
            pw2_w=rng.normal(0, 0.1, (32, 64)).astype(np.float32),
            pw2_b=np.zeros(64, dtype=np.float32),
            dw3_w=rng.normal(0, 0.1, (3, 3, 64)).astype(np.float32),
            pw3_w=rng.normal(0, 0.1, (64, 128)).astype(np.float32),
            pw3_b=np.zeros(128, dtype=np.float32),
            dense_w=rng.normal(0, 0.1, (128, 1)).astype(np.float32),
            dense_b=np.zeros(1, dtype=np.float32),
            norm_mean=np.zeros((129, 1), dtype=np.float32),
            norm_std=np.ones((129, 1), dtype=np.float32),
        )
        return path

    def test_probability_from_bytes(self):
        rng = np.random.default_rng(6)
        window = rng.normal(0, 3000, WINDOW_SAMPLES).astype(np.int16).tobytes()
        with tempfile.TemporaryDirectory() as tmp:
            verifier = CnnVerifier(self._weights_file(tmp))
            p = verifier.probability(window)
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)

    def test_short_window_raises(self):
        window = np.zeros(WINDOW_SAMPLES - 1, dtype=np.int16).tobytes()
        with tempfile.TemporaryDirectory() as tmp:
            verifier = CnnVerifier(self._weights_file(tmp))
            with self.assertRaises(ValueError):
                verifier.probability(window)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=./src:. uv run pytest tests/cnn_inference_test.py -v`
Expected: new tests FAIL with `ImportError: cannot import name 'preprocess'`

- [ ] **Step 3: Implement preprocess + CnnVerifier**

Append to `data_collection/cnn_inference.py`:

```python
def preprocess(samples_int16: np.ndarray, weights: dict) -> np.ndarray:
    """Raw int16 samples -> normalized (N_BINS, N_FRAMES) model input.

    Uses the most recent WINDOW_SAMPLES. Raw sample values go into the STFT
    unscaled (float32 of the int16 values) — this matches the training
    extraction exactly; do not divide by 32768.
    """
    if len(samples_int16) < WINDOW_SAMPLES:
        raise ValueError(
            f"need >= {WINDOW_SAMPLES} samples, got {len(samples_int16)}"
        )
    x = samples_int16[-WINDOW_SAMPLES:].astype(np.float32)
    _, _, zxx = stft(x, fs=SAMPLE_RATE)
    spec = np.abs(zxx)[:, :N_FRAMES]
    spec = np.log(spec + LOG_EPS)
    return (spec - weights["norm_mean"]) / weights["norm_std"]


class CnnVerifier:
    """Loads the exported weight npz once; scores 2 s PCM windows."""

    def __init__(self, weights_path):
        self.weights = dict(np.load(weights_path))

    def probability(self, window_bytes: bytes) -> float:
        samples = np.frombuffer(window_bytes, dtype=np.int16)
        return forward(preprocess(samples, self.weights), self.weights)
```

- [ ] **Step 4: Run the full suite**

Run: `PYTHONPATH=./src:. uv run pytest tests/ -v`
Expected: all pass (existing suite + 11 cnn_inference tests)

- [ ] **Step 5: Commit**

```bash
git add data_collection/cnn_inference.py tests/cnn_inference_test.py
git commit -m "CNN preprocessing with train/serve parity + CnnVerifier"
```

---

### Task 3: Weight export with BatchNorm folding + Keras parity

**Files:**
- Create: `src/export_cnn_weights.py`
- Test: `tests/export_cnn_weights_test.py`

**Interfaces:**
- Consumes: `models/cnn_model.keras`, `models/cnn_normalization.npz` (both produced by `dvc repro` on this branch); `forward` from `data_collection/cnn_inference.py`.
- Produces: `export_weights(model, norm_mean, norm_std) -> dict` (the npz-key dict from Global Constraints) and, when run as a script, `models/cnn_weights_pi.npz` plus `models/cnn_golden.npz` (keys: `spec` — one normalized `(129,250)` float32 input, `expected` — the Keras probability for it). Task 4 ships both files to the Pi.

- [ ] **Step 1: Write the failing test**

```python
# tests/export_cnn_weights_test.py
import unittest
from pathlib import Path

import numpy as np

from data_collection.cnn_inference import forward
from src.export_cnn_weights import export_weights

MODEL_PATH = Path("models/cnn_model.keras")
NORM_PATH = Path("models/cnn_normalization.npz")

EXPECTED_KEYS = {
    "conv1_w": (3, 3, 1, 32), "conv1_b": (32,),
    "dw2_w": (3, 3, 32), "pw2_w": (32, 64), "pw2_b": (64,),
    "dw3_w": (3, 3, 64), "pw3_w": (64, 128), "pw3_b": (128,),
    "dense_w": (128, 1), "dense_b": (1,),
    "norm_mean": (129, 1), "norm_std": (129, 1),
}


@unittest.skipUnless(
    MODEL_PATH.exists() and NORM_PATH.exists(),
    "trained model not present — run `uv run dvc repro` first",
)
class TestExportWeights(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from tensorflow import keras

        cls.model = keras.models.load_model(MODEL_PATH)
        stats = np.load(NORM_PATH)
        cls.weights = export_weights(cls.model, stats["mean"], stats["std"])

    def test_keys_and_shapes(self):
        self.assertEqual(set(self.weights), set(EXPECTED_KEYS))
        for key, shape in EXPECTED_KEYS.items():
            self.assertEqual(self.weights[key].shape, shape, key)

    def test_numpy_forward_matches_keras(self):
        # BN folding + the numpy reimplementation must agree with Keras on
        # arbitrary inputs; 1e-4 covers float32 accumulation-order noise.
        rng = np.random.default_rng(0)
        for _ in range(5):
            spec = rng.normal(size=(129, 250)).astype(np.float32)
            keras_p = float(self.model.predict(spec[np.newaxis, ..., np.newaxis], verbose=0)[0, 0])
            numpy_p = forward(spec, self.weights)
            self.assertAlmostEqual(numpy_p, keras_p, delta=1e-4)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=./src:. uv run pytest tests/export_cnn_weights_test.py -v`
Expected: ERROR with `ModuleNotFoundError: No module named 'src.export_cnn_weights'` (if it reports SKIPPED instead, run `set -a; source .env; set +a; uv run dvc repro` to materialize the model, then re-run)

- [ ] **Step 3: Implement the export script**

```python
# src/export_cnn_weights.py
"""Exports the trained Keras CNN to a single npz for pure-numpy inference
on the Pi (data_collection/cnn_inference.py).

BatchNorm layers are folded into the preceding (bias-free) conv weights:
    scale = gamma / sqrt(var + eps)
    w'    = w * scale        (over the output-channel axis)
    b'    = beta - mean * scale
For SeparableConv2D the BN follows the pointwise kernel, so the fold goes
there; the depthwise kernel passes through unchanged (squeezed to 3D).

Also writes models/cnn_golden.npz — one input with its Keras output — so
the Pi deployment can verify numerical parity on-device without TF.
"""

from pathlib import Path

import numpy as np

MODEL_PATH = Path("./models/cnn_model.keras")
NORM_PATH = Path("./models/cnn_normalization.npz")
WEIGHTS_OUT = Path("./models/cnn_weights_pi.npz")
GOLDEN_OUT = Path("./models/cnn_golden.npz")


def _fold_bn(kernel: np.ndarray, bn) -> tuple[np.ndarray, np.ndarray]:
    gamma, beta, mean, var = (w.astype(np.float32) for w in bn.get_weights())
    scale = gamma / np.sqrt(var + bn.epsilon)
    return kernel * scale, beta - mean * scale


def export_weights(model, norm_mean: np.ndarray, norm_std: np.ndarray) -> dict:
    from tensorflow.keras import layers

    convs = [
        l for l in model.layers
        if isinstance(l, (layers.Conv2D, layers.SeparableConv2D, layers.Dense))
        and not isinstance(l, layers.DepthwiseConv2D)
    ]
    bns = [l for l in model.layers if isinstance(l, layers.BatchNormalization)]
    conv1, sep2, sep3, dense = convs
    bn1, bn2, bn3 = bns

    conv1_w, conv1_b = _fold_bn(conv1.get_weights()[0], bn1)

    dw2, pw2 = sep2.get_weights()  # (3,3,32,1), (1,1,32,64)
    pw2_w, pw2_b = _fold_bn(pw2, bn2)
    dw3, pw3 = sep3.get_weights()
    pw3_w, pw3_b = _fold_bn(pw3, bn3)

    dense_w, dense_b = dense.get_weights()

    return {
        "conv1_w": conv1_w.astype(np.float32),
        "conv1_b": conv1_b.astype(np.float32),
        "dw2_w": dw2[:, :, :, 0].astype(np.float32),
        "pw2_w": pw2_w[0, 0].astype(np.float32),
        "pw2_b": pw2_b.astype(np.float32),
        "dw3_w": dw3[:, :, :, 0].astype(np.float32),
        "pw3_w": pw3_w[0, 0].astype(np.float32),
        "pw3_b": pw3_b.astype(np.float32),
        "dense_w": dense_w.astype(np.float32),
        "dense_b": dense_b.astype(np.float32),
        # stats were saved as (129, 1, 1); the numpy path broadcasts
        # against (129, 250) so squeeze to (129, 1)
        "norm_mean": norm_mean.reshape(129, 1).astype(np.float32),
        "norm_std": norm_std.reshape(129, 1).astype(np.float32),
    }


if __name__ == "__main__":
    from tensorflow import keras

    model = keras.models.load_model(MODEL_PATH)
    stats = np.load(NORM_PATH)
    weights = export_weights(model, stats["mean"], stats["std"])
    np.savez(WEIGHTS_OUT, **weights)
    print(f"wrote {WEIGHTS_OUT} ({sum(w.size for w in weights.values())} floats)")

    rng = np.random.default_rng(0)
    spec = rng.normal(size=(129, 250)).astype(np.float32)
    expected = float(model.predict(spec[np.newaxis, ..., np.newaxis], verbose=0)[0, 0])
    np.savez(GOLDEN_OUT, spec=spec, expected=np.float32(expected))
    print(f"wrote {GOLDEN_OUT} (expected={expected:.6f})")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=./src:. uv run pytest tests/export_cnn_weights_test.py -v`
Expected: 2 passed (this is the load-bearing parity check for the whole plan)

- [ ] **Step 5: Generate the artifacts**

Run: `set -a; source .env; set +a; PYTHONPATH=./src:. uv run python src/export_cnn_weights.py`
Expected: both `wrote models/cnn_weights_pi.npz ...` and `wrote models/cnn_golden.npz ...` lines. Verify size: `ls -la models/cnn_weights_pi.npz` (~60–80 KB).

- [ ] **Step 6: Commit**

```bash
git add src/export_cnn_weights.py tests/export_cnn_weights_test.py
git commit -m "Keras-to-npz weight export with BN folding + parity test"
```

(The npz artifacts are generated files; do not git-add them.)

---

### Task 4: Pi deployment + benchmark (decision gate)

**Files:**
- Create: `data_collection/benchmark_cnn.py`
- Remote: `pi@mic:/home/pi/doorbell-detector/`

**Interfaces:**
- Consumes: `CnnVerifier`-style pieces (`preprocess`, `forward`) from Task 2; `models/cnn_weights_pi.npz` + `models/cnn_golden.npz` from Task 3.
- Produces: measured on-Pi latency numbers and a **go/no-go decision** for Task 5. Gate: mean end-to-end (preprocess + forward) ≤ **2.5 s** → proceed; otherwise stop, report the numbers, and propose trimming options (crop frequency bins above ~4 kHz and retrain, or int8 quantization) instead of continuing.

- [ ] **Step 1: Write the benchmark script**

No unit test for this one — it *is* the measurement instrument; its correctness check is the golden-parity assertion it performs itself.

```python
# data_collection/benchmark_cnn.py
"""On-Pi benchmark for the numpy CNN. Verifies golden parity (the exported
weights produce the same probability the Keras model did on the dev
machine), then times preprocessing and forward pass separately.

Usage (on the Pi):
    .venv/bin/python benchmark_cnn.py --weights ../models/cnn_weights_pi.npz \
        --golden ../models/cnn_golden.npz --iters 10
"""

import argparse
import time

import numpy as np

from cnn_inference import WINDOW_SAMPLES, forward, preprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--golden", required=True)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    weights = dict(np.load(args.weights))
    golden = np.load(args.golden)

    p = forward(golden["spec"].astype(np.float32), weights)
    delta = abs(p - float(golden["expected"]))
    print(f"golden parity: got {p:.6f}, expected {float(golden['expected']):.6f}, "
          f"delta {delta:.2e} -> {'OK' if delta < 1e-4 else 'FAIL'}")
    if delta >= 1e-4:
        raise SystemExit(1)

    rng = np.random.default_rng(0)
    samples = rng.normal(0, 3000, WINDOW_SAMPLES).astype(np.int16)

    pre_times, fwd_times = [], []
    for _ in range(args.iters):
        t0 = time.monotonic()
        spec = preprocess(samples, weights)
        t1 = time.monotonic()
        forward(spec, weights)
        t2 = time.monotonic()
        pre_times.append(t1 - t0)
        fwd_times.append(t2 - t1)

    total = [a + b for a, b in zip(pre_times, fwd_times)]
    for name, ts in [("preprocess", pre_times), ("forward", fwd_times), ("total", total)]:
        print(f"{name:10s} mean {np.mean(ts):6.3f}s  min {np.min(ts):6.3f}s  "
              f"max {np.max(ts):6.3f}s  (n={len(ts)})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Sanity-run it on the dev machine**

Run: `cd data_collection && PYTHONPATH=. uv run --project .. python benchmark_cnn.py --weights ../models/cnn_weights_pi.npz --golden ../models/cnn_golden.npz --iters 3; cd ..`
Expected: `golden parity: ... OK` and sub-100 ms timings (dev machine is fast; this run validates the script, not the Pi).

- [ ] **Step 3: Check the Pi environment**

```bash
ssh pi@mic 'ls /home/pi/doorbell-detector/ /home/pi/doorbell-detector/data_collection/ | head -30'
ssh pi@mic '/home/pi/doorbell-detector/data_collection/.venv/bin/python -c "import numpy, scipy; print(numpy.__version__, scipy.__version__)"'
```
Expected: repo layout as in memory (venv under `data_collection/`); numpy ≥ 1.20 (needed for `sliding_window_view`). If numpy is older: `ssh pi@mic '/home/pi/doorbell-detector/data_collection/.venv/bin/pip install --upgrade numpy'` (piwheels has armv6l wheels) and re-check that `import scipy` still works afterward before proceeding.

- [ ] **Step 4: Deploy and run the benchmark**

```bash
rsync -av data_collection/cnn_inference.py data_collection/benchmark_cnn.py pi@mic:/home/pi/doorbell-detector/data_collection/
ssh pi@mic 'mkdir -p /home/pi/doorbell-detector/models'
rsync -av models/cnn_weights_pi.npz models/cnn_golden.npz pi@mic:/home/pi/doorbell-detector/models/
ssh pi@mic 'cd /home/pi/doorbell-detector/data_collection && .venv/bin/python benchmark_cnn.py --weights ../models/cnn_weights_pi.npz --golden ../models/cnn_golden.npz --iters 10'
```
Expected: `golden parity: ... OK`, then the three timing lines. Record the numbers.

- [ ] **Step 5: Decision gate**

- `total mean ≤ 2.5 s` → continue to Task 5.
- `total mean > 2.5 s` → STOP. Commit what exists, report the measured numbers to the user, and present the trimming options (bin cropping + retrain, int8 quantization). Do not implement them unprompted.

- [ ] **Step 6: Commit**

```bash
git add data_collection/benchmark_cnn.py
git commit -m "on-Pi CNN benchmark with golden parity check"
```

Record the measured timings in the commit message body.

---

### Task 5: detector.py second-stage verifier (only after Task 4 gate passes)

**Files:**
- Modify: `data_collection/detector.py` (args at `parse_args`, verifier init in `main`, detection block at the `score >= args.threshold` branch)
- Modify: `data_collection/systemd/doorbell-detector.service`, `data_collection/systemd/doorbell-detector.env`
- Test: `tests/cnn_inference_test.py` (append)

**Interfaces:**
- Consumes: `CnnVerifier` from Task 2 (`CnnVerifier(path).probability(bytes) -> float`).
- Produces: `--cnn-model PATH` / `--cnn-threshold FLOAT` CLI args on `detector.py`; empty/omitted `--cnn-model` keeps today's behavior exactly.

**Design decisions (already made — implement as stated):**
- The CNN runs **only after** a cross-correlation trigger passes threshold + cooldown (it is far too slow for every chunk).
- `last_detection_time` is set **before** the CNN verdict: a rejected event would otherwise re-trigger on every subsequent chunk, and each CNN pass costs seconds of dropped audio (this loop already has a documented audio-drop history, commit `c8c2788`).
- With `--save`, rejected detections are still saved, with a `doorbell_rejected_` filename prefix — they are exactly the hard negatives the training set lacks.

- [ ] **Step 1: Write the failing test**

Append to `tests/cnn_inference_test.py`:

```python
class TestDetectorCnnArgs(unittest.TestCase):

    def test_detector_exposes_cnn_args_with_safe_defaults(self):
        import sys
        from unittest import mock

        with mock.patch.object(sys, "argv", ["detector.py", "--template", "x.wav"]):
            from data_collection.detector import parse_args
            args = parse_args()
        self.assertIsNone(args.cnn_model)
        self.assertEqual(args.cnn_threshold, 0.5)
```

(If `tests/detector_test.py` already stubs `pyaudio` for imports, mirror its mechanism here; otherwise this import works because pyaudio is installed on the dev machine.)

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=./src:. uv run pytest tests/cnn_inference_test.py::TestDetectorCnnArgs -v`
Expected: FAIL with `AttributeError: ... 'cnn_model'`

- [ ] **Step 3: Add the CLI args**

In `data_collection/detector.py::parse_args`, after the `--cooldown-seconds` argument:

```python
    parser.add_argument(
        "--cnn-model",
        type=str,
        default=None,
        help="Path to the exported CNN weight npz (cnn_weights_pi.npz). If set, "
        "every cross-correlation trigger is verified by the CNN before a "
        "detection fires. Empty string disables (systemd env-var friendly).",
    )
    parser.add_argument(
        "--cnn-threshold",
        type=float,
        default=0.5,
        help="Minimum CNN probability to confirm a detection (default: 0.5).",
    )
```

- [ ] **Step 4: Wire the verifier into main()**

In `main()`, after the template-loading block and before "Save mode setup":

```python
    # -- CNN second-stage verifier ----------------------------------------
    cnn_verifier = None
    if args.cnn_model:  # None and "" (blank env var) both disable
        from cnn_inference import CnnVerifier
        cnn_verifier = CnnVerifier(args.cnn_model)
        log.info(
            "CNN verifier enabled: %s (threshold %.2f)",
            args.cnn_model, args.cnn_threshold,
        )
```

Replace the detection block (currently starting `if score >= args.threshold and (now - last_detection_time) >= args.cooldown_seconds:` through the end of its `log.info("Saving clip %s (detection) ...` call) with:

```python
                if score >= args.threshold and (now - last_detection_time) >= args.cooldown_seconds:
                    # Cooldown starts even if the CNN rejects below: the same
                    # acoustic event would re-trigger every chunk, and each CNN
                    # pass costs seconds of blocked capture (audio-drop history:
                    # see c8c2788).
                    last_detection_time = now
                    verified = True
                    cnn_prob = None
                    if cnn_verifier is not None:
                        t0 = time.monotonic()
                        cnn_prob = cnn_verifier.probability(b"".join(analysis_buf))
                        log.info(
                            "CNN verifier p(bell)=%.3f (%.2f s)",
                            cnn_prob, time.monotonic() - t0,
                        )
                        verified = cnn_prob >= args.cnn_threshold

                    if verified:
                        log.info("Doorbell detected! score=%.4f", score)
                        if mqtt_client is not None:
                            mqtt_client.publish(args.mqtt_detect_topic, datetime.datetime.now().isoformat())
                    else:
                        log.info(
                            "Detection suppressed by CNN verifier (score=%.4f, p=%.3f)",
                            score, cnn_prob,
                        )

                    if args.save and ring_buf is not None:
                        # Rejected triggers are saved too — labeled hard
                        # negatives for the training set.
                        prefix = "doorbell" if verified else "doorbell_rejected"
                        ts = datetime.datetime.now().strftime(f"{prefix}_%Y%m%d_%H%M%S.wav")
                        filepath = str(save_dir / ts)
                        pre_frames = list(ring_buf)
                        post_chunks = max(1, int(args.post_trigger_seconds / chunk_size_s))
                        post_frames = []
                        for _ in range(post_chunks):
                            try:
                                post_frames.append(stream.read(chunk, exception_on_overflow=False))
                            except OSError:
                                log.warning("Buffer overflow during post-trigger collection")
                                break
                        threading.Thread(
                            target=save_clip,
                            args=(pre_frames + post_frames, filepath),
                            daemon=True,
                        ).start()
                        total_seconds = (len(pre_frames) + len(post_frames)) * chunk_size_s
                        log.info(
                            "Saving clip %s (detection) (%d pre + %d post chunks, %.2f s)",
                            filepath, len(pre_frames), len(post_frames), total_seconds,
                        )
```

Note: `CnnVerifier.probability` raises `ValueError` if the analysis buffer holds under 2 s of audio. That can only happen in the first seconds after startup (the deque fills at 4 × 500 ms chunks); guard it by extending the condition: `if cnn_verifier is not None and len(analysis_buf) == analysis_buf.maxlen:` — with an `else` branch that logs `log.warning("Analysis window not yet full — skipping CNN verification")` and leaves `verified = True` (fail-open, same behavior as no verifier).

- [ ] **Step 5: Run the full suite**

Run: `PYTHONPATH=./src:. uv run pytest tests/ -v`
Expected: all pass, including the existing `detector_test.py` (nothing about the default path changed).

- [ ] **Step 6: Update the systemd unit + env sample**

In `data_collection/systemd/doorbell-detector.service`, extend `ExecStart` (after the `--threshold` line):

```
    --threshold      ${TEMPLATE_SCORE_THRESHOLD} \
    --cnn-model      ${CNN_MODEL_PATH} \
    --cnn-threshold  ${CNN_THRESHOLD}
```

In `data_collection/systemd/doorbell-detector.env`, append:

```
# ---- CNN second-stage verifier (optional) ------------------------------------
# Path to the exported CNN weight file (models/cnn_weights_pi.npz). When set,
# every cross-correlation trigger is confirmed by the CNN before the MQTT
# detection fires; rejected triggers are logged (and saved with a
# doorbell_rejected_ prefix when --save is active). Leave blank to disable.
CNN_MODEL_PATH=/home/pi/doorbell-detector/models/cnn_weights_pi.npz
# Minimum CNN probability in [0, 1] to confirm a detection.
CNN_THRESHOLD=0.5
```

- [ ] **Step 7: Deploy to the Pi and smoke-test**

```bash
rsync -av data_collection/detector.py data_collection/cnn_inference.py pi@mic:/home/pi/doorbell-detector/data_collection/
```

Add the two new variables to the **deployed** env file (remember it uses the typo'd `TEMPLATE_SCORE_THREHOLD`; do not "fix" that name, and do not touch existing lines):

```bash
ssh pi@mic "sudo sh -c 'printf \"\nCNN_MODEL_PATH=/home/pi/doorbell-detector/models/cnn_weights_pi.npz\nCNN_THRESHOLD=0.5\n\" >> /etc/doorbell-detector.env'"
```

Update the deployed unit's ExecStart the same way as the repo copy (the deployed unit may differ from the repo sample — read it first with `ssh pi@mic 'systemctl cat doorbell-detector'`, then edit via `sudo systemctl edit --full doorbell-detector` semantics or by copying the repo unit if they match). Then:

```bash
ssh pi@mic 'sudo systemctl daemon-reload && sudo systemctl restart doorbell-detector'
ssh pi@mic 'journalctl -u doorbell-detector -n 20 --no-pager'
```
Expected in the journal: `CNN verifier enabled: /home/pi/doorbell-detector/models/cnn_weights_pi.npz (threshold 0.50)` and the normal `Capture loop started` line, no tracebacks. If the service crash-loops, `journalctl -u doorbell-detector -n 50` for the traceback, and roll back by blanking `CNN_MODEL_PATH=` in the env file + restart.

- [ ] **Step 8: Live verification**

Trigger the doorbell (or play the template WAV near the mic) once and watch:

```bash
ssh pi@mic 'journalctl -u doorbell-detector -f'
```
Expected: `CNN verifier p(bell)=... (N.NN s)` followed by either `Doorbell detected!` or `Detection suppressed by CNN verifier`. Report the observed probability and latency to the user — the CNN threshold and the template threshold (0.4, still uncalibrated) may both need tuning against this.

- [ ] **Step 9: Commit**

```bash
git add data_collection/detector.py data_collection/systemd/doorbell-detector.service data_collection/systemd/doorbell-detector.env tests/cnn_inference_test.py
git commit -m "CNN second-stage verifier gates cross-correlation detections"
```

---

## Out of scope (deliberately)

- Retraining with cropped frequency bins / quantization — only if the Task 4 gate fails, and only after reporting to the user.
- Calibrating `TEMPLATE_SCORE_THRESHOLD` / `CNN_THRESHOLD` against real rings — no labeled real-ring recording exists yet (open item since 2026-07-04).
- Merging `cnn-spectrogram` into `main` — the branch comparison (YAMNet et al.) is a separate decision.
- Any change to the MFCC/YAMNet branches.
