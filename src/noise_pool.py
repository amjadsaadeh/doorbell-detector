"""
Shared helpers for the external background-noise pool stages
(fetch_noise_esc50.py / fetch_noise_demand.py).

Each pool stage downloads a public dataset and writes plain 16kHz mono 16-bit
wav files into data/noise/<pool>/, which augment_data.py picks up as
additional noise sources for SNR mixing. Downloaded archives are cached in
data/downloads/ (git-ignored, not a DVC output) so parameter changes only
re-run the cheap filter/resample step, not the download.
"""

from pathlib import Path

import librosa
import numpy as np
import requests
import soundfile as sf
from tqdm import tqdm

SAMPLE_RATE = 16000  # project-wide convention, see CLAUDE.md
DOWNLOAD_CACHE = Path("./data/downloads")


def download_archive(url: str, file_name: str) -> Path:
    """Downloads url into the archive cache, skipping if already present.

    Streams to a .part file first so an interrupted download is never
    mistaken for a complete archive on the next run.
    """
    DOWNLOAD_CACHE.mkdir(parents=True, exist_ok=True)
    target = DOWNLOAD_CACHE / file_name
    if target.exists():
        return target

    partial = target.with_name(file_name + ".part")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with (
            open(partial, "wb") as f,
            tqdm(
                desc=f"Downloading {file_name}",
                total=total or None,
                unit="B",
                unit_scale=True,
            ) as progress,
        ):
            for chunk in response.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                progress.update(len(chunk))
    partial.rename(target)
    return target


def write_16k_mono_wav(source, target_path: Path) -> None:
    """Downmixes/resamples an audio file (path or file-like object) to the
    project-wide 16kHz mono int16 convention and writes it as wav.
    """
    samples, _ = librosa.load(source, sr=SAMPLE_RATE, mono=True)
    pcm = np.clip(samples * 32767, -32768, 32767).astype(np.int16)
    sf.write(target_path, pcm, SAMPLE_RATE, subtype="PCM_16")


def clear_pool_dir(pool_dir: Path) -> None:
    """(Re)creates a pool output dir, pruning wavs from previous runs."""
    pool_dir.mkdir(parents=True, exist_ok=True)
    for existing in pool_dir.glob("*.wav"):
        existing.unlink()
