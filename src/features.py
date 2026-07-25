"""One registry for every feature representation the pipeline can build.

Previously each representation was its own script on its own branch
(extract_mfcc_features.py, extract_stft_features.py, ...), which were ~90%
identical boilerplate around a ~10-line transform, and each shipped a
matching hand-edited slicer inside draw_data.py. Both halves live here
instead, keyed by `feature_extraction.type`, so a variant is a parameter
rather than a branch.

Each entry owns three things:

* `params` -- which keys it reads out of the `feature_extraction` block,
  so the sidecar config (and therefore the DVC param hash) contains exactly
  what affects the output and nothing else.
* `extract` -- file -> (n_bins, n_frames) float array.
* `slice_chunk` -- how draw_data.py cuts a [start_ms, end_ms) window out of
  that array. This is per-type because it is not always a fixed frame rate
  (see yamnet below).
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from audio_sources import SAMPLING_RATE

FEATURE_CONFIG_FILENAME = "feature_config.json"


def _load_mono_int16_scale(file_path: Path | str) -> tuple[np.ndarray, int]:
    """Audio as float32 at int16 amplitude scale, downmixed to mono.

    Mono matters: stereo files would double the sample count (interleaved)
    and break the frames-per-ms assumption the slicers rely on. int16 scale
    (rather than [-1, 1]) is kept because every extractor here has always
    used it, and normalization happens globally at training time.
    """
    from pydub import AudioSegment
    from pydub.utils import mediainfo

    audio = AudioSegment.from_wav(file_path)
    info = mediainfo(file_path)
    audio = audio.set_channels(1)
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    return samples, int(info["sample_rate"])


def extract_mfcc(file_path, n_mfcc: int, n_fft: int, hop_length: int) -> np.ndarray:
    import librosa

    samples, sample_rate = _load_mono_int16_scale(file_path)
    mfccs = librosa.feature.mfcc(
        y=samples, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
    )
    return mfccs.astype(np.float32)


def extract_logmel(
    file_path,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    fmin: int,
    fmax: int | None,
    log_offset: float,
) -> np.ndarray:
    """Log-mel filterbank energies -- the MFCC path minus the final DCT.

    MFCCs are log-mel energies run through a DCT, which decorrelates the
    bins so a tree sees compact, roughly independent features. A CNN does
    not want that: it wants the locally correlated structure along the
    frequency axis that the DCT destroys.

    The log is applied here rather than in the trainer, so params.yaml sets
    model.log_compress: false for this type and the .npy files are exactly
    what an on-device C frontend has to reproduce.
    """
    import librosa

    samples, sample_rate = _load_mono_int16_scale(file_path)
    mel = librosa.feature.melspectrogram(
        y=samples,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )
    # log_offset keeps digital-silence frames finite; librosa's power_to_db
    # would additionally clip to top_db below the per-file peak, which is a
    # per-file operation we don't want ahead of a global normalization.
    return np.log(mel + log_offset).astype(np.float32)


def extract_stft(file_path, n_fft: int, hop_length: int) -> np.ndarray:
    """Raw STFT magnitude. Not log-compressed here -- params.yaml sets
    model.log_compress: true for this type so the trainer does it."""
    from scipy.signal import stft

    samples, sample_rate = _load_mono_int16_scale(file_path)
    _, _, zxx = stft(
        samples, fs=sample_rate, nperseg=n_fft, noverlap=n_fft - hop_length
    )
    return np.abs(zxx).astype(np.float32)


def extract_yamnet(file_path, model=None) -> np.ndarray:
    """YAMNet (MobileNet-v1 on AudioSet) frame embeddings, (1024, n_frames).

    One 1024-dim embedding per 0.48s hop over 0.96s windows of 16kHz mono
    audio. `model` is injected by the caller so the TF Hub load happens once
    per process rather than once per file.
    """
    import librosa

    if model is None:
        model = load_yamnet()
    waveform, _ = librosa.load(file_path, sr=SAMPLING_RATE, mono=True)
    # zero-pad below one full analysis window, else YAMNet emits no frames
    min_samples = int(0.96 * SAMPLING_RATE)
    if len(waveform) < min_samples:
        waveform = np.pad(waveform, (0, min_samples - len(waveform)))
    _, embeddings, _ = model(waveform)
    # (n_frames, 1024) -> (1024, n_frames), matching the (bins, time)
    # orientation every other type produces
    return embeddings.numpy().T.astype(np.float32)


YAMNET_HUB_URL = "https://tfhub.dev/google/yamnet/1"
TFHUB_CACHE_DIR = Path("./data/downloads/tfhub")


def load_yamnet():
    import os

    # TFHUB_CACHE_DIR must be set before tensorflow_hub is imported
    TFHUB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TFHUB_CACHE_DIR", str(TFHUB_CACHE_DIR))
    import tensorflow_hub as hub

    return hub.load(YAMNET_HUB_URL)


def fixed_rate_slice(array: np.ndarray, start_ms: int, end_ms: int, cfg: dict):
    """Cut [start_ms, end_ms) using the extractor's nominal frame rate.

    Deliberately NOT a per-file average rate (array.shape[1] / duration):
    the frame count carries a constant +1 offset, which is negligible for
    long real files (average rate ~= true rate) but dominant for exactly
    chunk_size-long augmented clips (average rate skews high). Per-file
    rates produced inconsistent chunk widths and broke np.vstack downstream.
    """
    frames_per_ms = SAMPLING_RATE / cfg["hop_length"] / 1000
    start_frame = int(frames_per_ms * start_ms)
    # width from the duration, not from end_frame - start_frame, so every
    # chunk of a given length comes out the same width
    width = int(frames_per_ms * (end_ms - start_ms))
    return array[:, start_frame : start_frame + width]


YAMNET_HOP_MS = 480


def yamnet_slice(array: np.ndarray, start_ms: int, end_ms: int, cfg: dict):
    """Cut the covering frames and mean-pool them to one fixed-size vector.

    Pooling is what makes the result shape-independent of the frame count:
    a 2s augmented clip yields 3 frames (no final partial window) while a 2s
    slice of a long file yields 4, and both have to produce the same feature
    length for the downstream stack.
    """
    start_frame = int(start_ms // YAMNET_HOP_MS)
    n_frames = max(1, int((end_ms - start_ms) // YAMNET_HOP_MS))
    frames = array[:, start_frame : start_frame + n_frames]
    if frames.shape[1] == 0:
        # chunk starts past the last emitted frame (file-end edge case)
        frames = array[:, -1:]
    return frames.mean(axis=1)


@dataclass(frozen=True)
class FeatureSpec:
    params: tuple[str, ...]
    extract: Callable
    slice_chunk: Callable
    # YAMNet is a TF graph: not fork-safe, and each Pool worker would
    # re-load the model. Single-process on CPU is far faster than real time.
    parallel: bool = True
    # Optional per-process setup whose result is splatted into every
    # extract() call -- how the YAMNet model gets loaded once, not per file.
    context: Callable[[], dict] | None = None
    # A CNN head needs the (bins, frames) structure; yamnet_slice pools the
    # time axis away and returns a 1D embedding, which only the flattening
    # tree head can consume.
    keeps_time_axis: bool = True


REGISTRY: dict[str, FeatureSpec] = {
    "mfcc": FeatureSpec(
        params=("n_mfcc", "n_fft", "hop_length"),
        extract=extract_mfcc,
        slice_chunk=fixed_rate_slice,
    ),
    "logmel": FeatureSpec(
        params=("n_mels", "n_fft", "hop_length", "fmin", "fmax", "log_offset"),
        extract=extract_logmel,
        slice_chunk=fixed_rate_slice,
    ),
    "stft": FeatureSpec(
        params=("n_fft", "hop_length"),
        extract=extract_stft,
        slice_chunk=fixed_rate_slice,
    ),
    "yamnet": FeatureSpec(
        params=(),
        extract=extract_yamnet,
        slice_chunk=yamnet_slice,
        parallel=False,
        keeps_time_axis=False,
        context=lambda: {"model": load_yamnet()},
    ),
}


def get_spec(feature_type: str) -> FeatureSpec:
    try:
        return REGISTRY[feature_type]
    except KeyError:
        raise SystemExit(
            f"unknown feature_extraction.type {feature_type!r}; "
            f"known types: {', '.join(sorted(REGISTRY))}"
        )


def extractor_config(feature_params: dict) -> dict:
    """The subset of feature_extraction that actually changes the output.

    `features_dir` is deliberately excluded: where the arrays are written
    does not affect what is in them, so moving the directory must not
    invalidate a cached extraction.
    """
    spec = get_spec(feature_params["type"])
    missing = [k for k in spec.params if k not in feature_params]
    if missing:
        raise SystemExit(
            f"feature_extraction.type {feature_params['type']!r} needs "
            f"missing param(s): {', '.join(missing)}"
        )
    config = {k: feature_params[k] for k in spec.params}
    config["type"] = feature_params["type"]
    return config


def write_feature_config(features_dir: Path, feature_params: dict) -> None:
    config = extractor_config(feature_params)
    (features_dir / FEATURE_CONFIG_FILENAME).write_text(json.dumps(config, indent=4))


def verify_feature_config(features_dir: Path, feature_params: dict) -> dict:
    """Fail loudly when the .npy files on disk were built by a different
    configuration than the one currently in params.yaml.

    This is the guard for the failure that produced data/logmel_data.esp32bak:
    two branches wrote different geometry (13 bands @ hop 512 vs 40 @ hop 320)
    into the same directory name, and nothing downstream noticed until the
    shapes disagreed several stages later -- or worse, did not disagree.
    """
    expected = extractor_config(feature_params)
    config_path = features_dir / FEATURE_CONFIG_FILENAME

    if not config_path.exists():
        raise SystemExit(
            f"{config_path} is missing -- {features_dir} was not written by "
            f"extract_features.py. Re-run the extract_features stage."
        )

    actual = json.loads(config_path.read_text())
    if actual != expected:
        differing = sorted(
            set(actual) | set(expected), key=lambda k: (actual.get(k) == expected.get(k), k)
        )
        detail = "\n".join(
            f"  {k}: on disk {actual.get(k)!r} != params.yaml {expected.get(k)!r}"
            for k in differing
            if actual.get(k) != expected.get(k)
        )
        raise SystemExit(
            f"stale features in {features_dir}:\n{detail}\n"
            f"Re-run the extract_features stage, or point "
            f"feature_extraction.features_dir at a different directory."
        )
    return actual
