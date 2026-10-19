"""Golden-vector comparisons run the actual firmware C++ on the host."""
from pathlib import Path
import subprocess

import librosa
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
FW = ROOT / "firmware/esp32-sense"


@pytest.fixture(scope="module")
def frontend_runner(tmp_path_factory):
    exe = tmp_path_factory.mktemp("frontend") / "run"
    subprocess.run(["c++", "-std=c++17", "-O2", "-I", str(FW / "lib/doorbell_frontend"),
                    str(FW / "test/host/frontend_runner.cpp"),
                    str(FW / "lib/doorbell_frontend/frontend.cpp"), "-o", str(exe)], check=True)
    return exe


@pytest.mark.parametrize("kind", ["silence", "noise", "tone", "impulse", "clipping"])
def test_streaming_matches_librosa(frontend_runner, kind):
    rng = np.random.default_rng(42)
    pcm = rng.integers(-16000, 16000, 80000, dtype=np.int16)
    if kind == "silence":
        pcm[:] = 0
    elif kind == "tone":
        pcm = (16000 * np.sin(2 * np.pi * 1000 * np.arange(len(pcm)) / 16000)).astype(np.int16)
    elif kind == "impulse":
        pcm[:] = 0
        pcm[[0, 1000, 31700, 79999]] = 32767
    elif kind == "clipping":
        pcm = np.where(pcm > 0, 32767, -32768).astype(np.int16)
    raw = subprocess.run([str(frontend_runner)], input=pcm.astype("<i2").tobytes(),
                         capture_output=True, check=True).stdout
    windows = np.frombuffer(raw, dtype=np.float32).reshape(-1, 40, 100)
    mel = librosa.feature.melspectrogram(y=pcm.astype(np.float32), sr=16000,
                n_fft=512, hop_length=320, n_mels=40, fmin=50, fmax=8000)
    expected = np.log(mel + 1e-6)
    assert len(windows) == 7
    for i, window in enumerate(windows):
        np.testing.assert_allclose(window, expected[:, i * 25:i * 25 + 100], atol=1e-3, rtol=0)


def test_generated_tables_are_current():
    from prepare_esp32_model import frontend_tables
    assert (FW / "lib/doorbell_frontend/frontend_tables.h").read_text() == frontend_tables()


def test_model_package_rejects_changed_source(tmp_path):
    import hashlib
    import importlib.util
    import json
    spec = importlib.util.spec_from_file_location("model_package", FW / "tools/model_package.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    firmware = tmp_path / "firmware/esp32-sense"
    package = firmware / "lib/doorbell_model"
    package.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="prepare_esp32_model"):
        module.verify_package(firmware)
    source = tmp_path / "model.tflite"
    source.write_bytes(b"evaluated bytes")
    (package / "model_manifest.json").write_text(json.dumps({"source_sha256": {
        "model.tflite": hashlib.sha256(source.read_bytes()).hexdigest()}}))
    module.verify_package(firmware)
    source.write_bytes(b"different bytes")
    with pytest.raises(RuntimeError, match="Stale model package"):
        module.verify_package(firmware)
    source.unlink()
    with pytest.raises(RuntimeError, match="Stale model package"):
        module.verify_package(firmware)


@pytest.mark.parametrize("section,key,value", [
    ("training", "head", "xgboost"),
    ("feature_extraction", "n_mels", 13),
    ("feature_extraction", "hop_length", 160),
    ("feature_extraction", "log_offset", 1e-4),
])
def test_preparation_rejects_incompatible_configuration(section, key, value):
    from prepare_esp32_model import EXPECTED, validate_configuration
    params = {"training": {"head": "cnn"}, "chunk_size": 2000,
              "feature_extraction": dict(EXPECTED)}
    validate_configuration(params)
    params[section][key] = value
    with pytest.raises(ValueError, match="40x100"):
        validate_configuration(params)
