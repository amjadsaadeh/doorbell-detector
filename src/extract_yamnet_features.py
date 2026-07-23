"""
Per-file YAMNet frame embeddings for every audio source the pipeline uses
(real recordings, augmented mixes, external noise pools), written as
(1024, n_frames) .npy files into data/yamnet_data/ — same orientation and
naming convention as the previous MFCC path, so draw_data.py slices them
by chunk the same way.

YAMNet (MobileNet-v1 pretrained on AudioSet) emits one 1024-dim embedding
per 0.48s hop over 0.96s windows of 16kHz mono audio. The model is fetched
from TF Hub once and cached under data/downloads/tfhub (git-ignored, next
to the noise-pool archive cache), so only the first run needs the network.

Files are processed serially: TF is not fork-safe and each worker of a
multiprocessing.Pool would re-load the model; single-process YAMNet on CPU
is far faster than real time, which is enough here.
"""

import os
from pathlib import Path

import numpy as np
import yaml

SAMPLE_RATE = 16000  # project-wide convention, see CLAUDE.md
# YAMNet needs at least one full 0.96s analysis window
MIN_SAMPLES = int(0.96 * SAMPLE_RATE)

YAMNET_HUB_URL = "https://tfhub.dev/google/yamnet/1"
TFHUB_CACHE_DIR = Path("./data/downloads/tfhub")

AUDIO_DATA_PATH = Path("./data/audio")
AUGMENTED_AUDIO_DATA_PATH = Path("./data/augmented_audio")
OUTPUT_PATH = Path("./data/yamnet_data")


def ensure_min_length(waveform: np.ndarray, min_samples: int = MIN_SAMPLES) -> np.ndarray:
    """Zero-pads waveforms shorter than one YAMNet analysis window so the
    model emits at least one frame instead of an empty embedding array."""
    if len(waveform) >= min_samples:
        return waveform
    return np.pad(waveform, (0, min_samples - len(waveform)))


def load_yamnet():
    # TFHUB_CACHE_DIR must be set before tensorflow_hub is imported
    TFHUB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TFHUB_CACHE_DIR", str(TFHUB_CACHE_DIR))
    import tensorflow_hub as hub

    return hub.load(YAMNET_HUB_URL)


def process_audio_data(model, input_path: Path, output_path: Path):
    import librosa
    from tqdm import tqdm

    output_path.mkdir(parents=True, exist_ok=True)

    audio_files = sorted(input_path.glob("*.wav"))
    for audio_file in tqdm(audio_files, desc=f"YAMNet embeddings: {input_path}"):
        # resample + downmix defensively; noise pools are already
        # 16kHz mono, uploaded recordings may not be
        waveform, _ = librosa.load(audio_file, sr=SAMPLE_RATE, mono=True)
        waveform = ensure_min_length(waveform)
        _, embeddings, _ = model(waveform)
        # (n_frames, 1024) -> (1024, n_frames), mirroring the (n_mfcc, T)
        # orientation draw_data.py slices on its second axis
        np.save(output_path / (audio_file.stem + ".npy"), embeddings.numpy().T)


if __name__ == "__main__":
    with open("params.yaml", "r") as file:
        params = yaml.safe_load(file)

    model = load_yamnet()

    process_audio_data(model, AUDIO_DATA_PATH, OUTPUT_PATH)
    # Augmented (mixed) chunks land in the same output dir; filenames are
    # unique across both sources so there's no collision.
    process_audio_data(model, AUGMENTED_AUDIO_DATA_PATH, OUTPUT_PATH)
    # External noise pool files too (used as extra background chunks in
    # draw_data.py). Also flat: pool naming schemes (ESC-50 fold-id clips,
    # DEMAND <ENV>_ch01) don't collide with recordings or aug_* files.
    for pool_path in params["augmentation"]["external_noise_pools"]:
        process_audio_data(model, Path(pool_path), OUTPUT_PATH)
