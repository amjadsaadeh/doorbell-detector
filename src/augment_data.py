"""
Grows the front_doorbell class by mixing real doorbell audio (signal) with
real background audio (noise) via simple addition, at a range of target SNRs.
This creates new synthetic front_doorbell samples instead of just resampling
the existing ones.
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pydub import AudioSegment
from pydub.utils import mediainfo
from tqdm import tqdm

AUDIO_FILE_BASE = Path("./data/audio")
OUTPUT_AUDIO_BASE = Path("./data/augmented_audio")
OUTPUT_ANNOTATIONS_FILE = Path("./data/augmented_annotations.csv")

SIGNAL_LABEL = "front_doorbell"
NOISE_LABEL = "background"
SAMPLE_RATE = 16000  # project-wide convention, see CLAUDE.md
SAMPLE_WIDTH = 2  # 16-bit PCM


def load_mono_samples(audio_file_name: str) -> tuple[np.ndarray, int]:
    audio = AudioSegment.from_wav(AUDIO_FILE_BASE / audio_file_name).set_channels(1)
    samples = np.array(audio.get_array_of_samples(), dtype=np.float64)
    return samples, audio.frame_rate


def extract_windows(df: pd.DataFrame, chunk_size_ms: int, chunk_overlap_ms: int) -> list[np.ndarray]:
    """Slides a chunk_size_ms window (stepping by chunk_overlap_ms) over every
    labeled span in df and returns the raw sample arrays. Mirrors the
    windowing convention in draw_data.py so augmented chunks line up with
    real ones.
    """
    windows = []
    for audio_file_name, group in df.groupby("audio_file_name"):
        samples, frame_rate = load_mono_samples(audio_file_name)
        samples_per_ms = frame_rate / 1000
        chunk_len_samples = int(samples_per_ms * chunk_size_ms)

        for _, row in group.iterrows():
            end = row["end"]
            if pd.isna(end):
                end = float(mediainfo(AUDIO_FILE_BASE / audio_file_name)["duration"])
            start_ms = int(row["start"] * 1000)
            end_ms = int(end * 1000)

            for chunk_start_ms in range(start_ms, end_ms - chunk_size_ms, chunk_overlap_ms):
                start_sample = int(samples_per_ms * chunk_start_ms)
                window = samples[start_sample : start_sample + chunk_len_samples]
                if len(window) == chunk_len_samples:
                    windows.append(window)

    return windows


def mix_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Scales noise to hit the target SNR and adds it to signal (doorbell is
    always the signal, everything else is noise).
    """
    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)

    if noise_power == 0:
        return signal

    scale = np.sqrt(signal_power / (noise_power * 10 ** (snr_db / 10)))
    mixed = signal + scale * noise

    # Renormalize only if we'd clip int16 range; scaling both terms equally
    # preserves the realized SNR.
    peak = np.max(np.abs(mixed))
    if peak > 32767:
        mixed = mixed * (32767 / peak)

    return mixed


if __name__ == "__main__":
    with open("params.yaml", "r") as file:
        params = yaml.safe_load(file)

    chunk_size = params["chunk_size"]
    chunk_overlap = params["chunk_overlap"]
    snrs_db = params["augmentation"]["snrs_db"]
    pairs_per_signal_chunk = params["augmentation"]["pairs_per_signal_chunk"]

    annotated_data = pd.read_csv("./data/annotation_per_row_data.csv")

    signal_rows = annotated_data[annotated_data["label"] == SIGNAL_LABEL]
    noise_rows = annotated_data[annotated_data["label"] == NOISE_LABEL]

    signal_windows = extract_windows(signal_rows, chunk_size, chunk_overlap)
    noise_windows = extract_windows(noise_rows, chunk_size, chunk_overlap)

    OUTPUT_AUDIO_BASE.mkdir(parents=True, exist_ok=True)
    # Prune stale augmented files from previous runs (e.g. old SNR sweep)
    for existing_file in OUTPUT_AUDIO_BASE.glob("*.wav"):
        existing_file.unlink()

    rng = random.Random(42)
    rows = []

    for snr_db in snrs_db:
        for signal_idx, signal in enumerate(
            tqdm(signal_windows, desc=f"Mixing at {snr_db}dB SNR")
        ):
            for pair_idx in range(pairs_per_signal_chunk):
                noise = rng.choice(noise_windows)
                mixed = mix_at_snr(signal, noise, snr_db)

                sample_id = f"aug_snr{snr_db}_{signal_idx}_{pair_idx}"
                file_name = f"{sample_id}.wav"

                mixed_audio = AudioSegment(
                    mixed.astype(np.int16).tobytes(),
                    frame_rate=SAMPLE_RATE,
                    sample_width=SAMPLE_WIDTH,
                    channels=1,
                )
                mixed_audio.export(OUTPUT_AUDIO_BASE / file_name, format="wav")

                rows.append(
                    {
                        "annotation_id": sample_id,
                        "file_id": sample_id,
                        "start": 0.0,
                        # +1ms beyond the real chunk length forces draw_data.py's
                        # sliding-window loop to emit exactly one chunk_start=0
                        # window, reusing its existing chunking path unmodified.
                        "end": (chunk_size + 1) / 1000,
                        "label": SIGNAL_LABEL,
                        "audio_file_name": file_name,
                        "remote_audio_path": "",
                        "snr_db": snr_db,
                    }
                )

    pd.DataFrame(rows).to_csv(OUTPUT_ANNOTATIONS_FILE, index=False)
