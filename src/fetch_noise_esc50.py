"""
Builds the ESC-50 external noise pool for augmentation.

Downloads the ESC-50 environmental sound dataset (2000 x 5s clips, 44.1kHz,
CC-licensed, https://github.com/karolpiczak/ESC-50), drops the categories
excluded in params.yaml and resamples everything to 16kHz mono into
data/noise/esc50/.

Bell-like categories are excluded by default because augmentation labels
every mixture front_doorbell: at the low end of the SNR sweep the noise
dominates the mixture, and a mostly-church-bell sample labeled
front_doorbell would teach the model that other bells are doorbells.
"""

import io
import zipfile
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

from noise_pool import clear_pool_dir, download_archive, write_16k_mono_wav

ESC50_URL = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"
ARCHIVE_ROOT = "ESC-50-master"
OUTPUT_DIR = Path("./data/noise/esc50")


def select_clips(meta: pd.DataFrame, exclude_categories: list[str]) -> pd.DataFrame:
    """Drops excluded categories, failing loudly on unknown names so a typo
    in params.yaml never silently lets a bell-like class into the pool.
    """
    unknown = set(exclude_categories) - set(meta["category"])
    if unknown:
        raise ValueError(f"Unknown ESC-50 categories in exclude list: {sorted(unknown)}")
    return meta[~meta["category"].isin(exclude_categories)]


if __name__ == "__main__":
    with open("params.yaml", "r") as file:
        params = yaml.safe_load(file)
    exclude_categories = params["noise_pools"]["esc50"]["exclude_categories"]

    archive_path = download_archive(ESC50_URL, "esc50.zip")

    with zipfile.ZipFile(archive_path) as archive:
        with archive.open(f"{ARCHIVE_ROOT}/meta/esc50.csv") as meta_file:
            meta = pd.read_csv(meta_file)
        clips = select_clips(meta, exclude_categories)

        clear_pool_dir(OUTPUT_DIR)
        for file_name in tqdm(clips["filename"], desc="Resampling ESC-50 to 16kHz mono"):
            audio_bytes = archive.read(f"{ARCHIVE_ROOT}/audio/{file_name}")
            write_16k_mono_wav(io.BytesIO(audio_bytes), OUTPUT_DIR / file_name)
