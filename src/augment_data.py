"""
Grows the front_doorbell class by mixing real doorbell audio (signal) with
background audio (noise) via simple addition, at a range of target SNRs.
Noise comes from the labeled background recordings and from external noise
pools (data/noise/<pool>, prepared by the fetch_noise_* stages). This creates
new synthetic front_doorbell samples instead of just resampling the existing
ones.
"""

import random
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
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


def load_mono_samples(audio_path: Path) -> tuple[np.ndarray, int]:
    audio = AudioSegment.from_wav(audio_path).set_channels(1)
    samples = np.array(audio.get_array_of_samples(), dtype=np.float64)
    return samples, audio.frame_rate


def extract_windows(df: pd.DataFrame, chunk_size_ms: int, chunk_overlap_ms: int) -> list[tuple[np.ndarray, str]]:
    """Slides a chunk_size_ms window (stepping by chunk_overlap_ms) over every
    labeled span in df and returns (raw sample array, source file name) pairs.
    Mirrors the windowing convention in draw_data.py so augmented chunks line
    up with real ones. The source file name becomes the split_group of the
    mixed sample, so augmented variants never straddle a group-aware
    train/val split of their source recording.
    """
    windows = []
    for audio_file_name, group in df.groupby("audio_file_name"):
        samples, frame_rate = load_mono_samples(AUDIO_FILE_BASE / audio_file_name)
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
                    windows.append((window, audio_file_name))

    return windows


def _index_span_windows(
    wav_path: Path, start_ms: int, end_ms: int, chunk_size_ms: int, chunk_overlap_ms: int
) -> list[tuple[Path, int, int]]:
    """Window index entries (path, start_sample, chunk_len_samples) for one
    labeled span, using the same size/stride convention as extract_windows.
    Noise windows are indexed instead of materialized: the external pools
    hold tens of thousands of candidate windows, and cutting them all up
    front would cost gigabytes of memory for windows that are mostly never
    drawn.
    """
    info = sf.info(str(wav_path))
    samples_per_ms = info.samplerate / 1000
    chunk_len_samples = int(samples_per_ms * chunk_size_ms)

    entries = []
    for chunk_start_ms in range(start_ms, end_ms - chunk_size_ms, chunk_overlap_ms):
        start_sample = int(samples_per_ms * chunk_start_ms)
        if start_sample + chunk_len_samples <= info.frames:
            entries.append((wav_path, start_sample, chunk_len_samples))
    return entries


def index_annotation_windows(
    df: pd.DataFrame, chunk_size_ms: int, chunk_overlap_ms: int
) -> list[tuple[Path, int, int]]:
    """Window index over the labeled spans of real recordings (data/audio).
    Full-file background annotations carry no end time; the real file
    duration fills in, mirroring draw_data.py.
    """
    entries = []
    for _, row in df.iterrows():
        wav_path = AUDIO_FILE_BASE / row["audio_file_name"]
        end = row["end"]
        if pd.isna(end):
            info = sf.info(str(wav_path))
            end = info.frames / info.samplerate
        entries += _index_span_windows(
            wav_path, int(row["start"] * 1000), int(end * 1000), chunk_size_ms, chunk_overlap_ms
        )
    return entries


def index_pool_windows(
    pool_dir: Path, chunk_size_ms: int, chunk_overlap_ms: int
) -> list[tuple[Path, int, int]]:
    """Window index over every wav of an external noise pool
    (data/noise/<pool>, prepared by a fetch_noise_* stage at 16kHz mono).
    """
    entries = []
    for wav_path in sorted(pool_dir.glob("*.wav")):
        info = sf.info(str(wav_path))
        duration_ms = int(info.frames / info.samplerate * 1000)
        entries += _index_span_windows(wav_path, 0, duration_ms, chunk_size_ms, chunk_overlap_ms)
    return entries


@lru_cache(maxsize=8)
def _cached_mono_samples(path_str: str) -> np.ndarray:
    return load_mono_samples(Path(path_str))[0]


def draw_noise_window(entry: tuple[Path, int, int]) -> np.ndarray:
    wav_path, start_sample, chunk_len_samples = entry
    samples = _cached_mono_samples(str(wav_path))
    return samples[start_sample : start_sample + chunk_len_samples]


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
    gain_jitter_db = params["augmentation"]["gain_jitter_db"]

    annotated_data = pd.read_csv("./data/annotation_per_row_data.csv")

    signal_rows = annotated_data[annotated_data["label"] == SIGNAL_LABEL]
    noise_rows = annotated_data[annotated_data["label"] == NOISE_LABEL]

    signal_windows = extract_windows(signal_rows, chunk_size, chunk_overlap)

    # Noise sources: the labeled background recordings plus the external
    # pools (fetch_noise_* stages). Kept as separate pools because the
    # external ones hold orders of magnitude more windows — a single uniform
    # draw over all windows would almost never pick the deployed-mic noise.
    noise_pools = {
        "labeled_background": index_annotation_windows(noise_rows, chunk_size, chunk_overlap)
    }
    for pool_path in params["augmentation"]["external_noise_pools"]:
        pool_path = Path(pool_path)
        noise_pools[pool_path.name] = index_pool_windows(pool_path, chunk_size, chunk_overlap)

    for pool_name, entries in noise_pools.items():
        if not entries:
            raise ValueError(
                f"Noise pool '{pool_name}' has no candidate windows — "
                "did its fetch_noise_* stage run?"
            )
    pool_names = list(noise_pools)

    OUTPUT_AUDIO_BASE.mkdir(parents=True, exist_ok=True)
    # Prune stale augmented files from previous runs (e.g. old SNR sweep)
    for existing_file in OUTPUT_AUDIO_BASE.glob("*.wav"):
        existing_file.unlink()

    rng = random.Random(42)
    rows = []

    for snr_db in snrs_db:
        for signal_idx, (signal, signal_source) in enumerate(
            tqdm(signal_windows, desc=f"Mixing at {snr_db}dB SNR")
        ):
            for pair_idx in range(pairs_per_signal_chunk):
                # Pool first, then window: equal weight per pool regardless
                # of pool size, so real background stays represented
                noise_pool_name = rng.choice(pool_names)
                noise = draw_noise_window(rng.choice(noise_pools[noise_pool_name]))
                # Circularly shift the noise so repeated draws of the same
                # background chunk don't always align identically with the bell
                noise = np.roll(noise, rng.randrange(len(noise)))
                mixed = mix_at_snr(signal, noise, snr_db)
                # Loudness jitter: vary absolute level (SNR is unaffected since
                # both terms scale equally) so synthetic positives don't all sit
                # at the source recording's gain
                gain_db = rng.uniform(-gain_jitter_db, gain_jitter_db)
                mixed = mixed * (10 ** (gain_db / 20))
                peak = np.max(np.abs(mixed))
                if peak > 32767:
                    mixed = mixed * (32767 / peak)

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
                        "noise_pool": noise_pool_name,
                        # Leakage guard for group-aware splitting: augmented
                        # samples belong to the recording their signal came from
                        "split_group": signal_source,
                    }
                )

    pd.DataFrame(rows).to_csv(OUTPUT_ANNOTATIONS_FILE, index=False)
