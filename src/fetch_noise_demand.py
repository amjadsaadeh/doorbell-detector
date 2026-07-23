"""
Builds the DEMAND external noise pool for augmentation.

Downloads selected environments of the DEMAND acoustic noise dataset
(https://zenodo.org/records/1227121, CC BY-SA 4.0) in their 16kHz variant
and keeps channel 1 of each 5-minute multichannel recording, written as
16kHz mono wav into data/noise/demand/. The environment list lives in
params.yaml (noise_pools.demand.environments); the default picks the
domestic environments plus a hallway, closest to the deployed setting.
"""

import io
import zipfile
from pathlib import Path

import yaml
from tqdm import tqdm

from noise_pool import clear_pool_dir, download_archive, write_16k_mono_wav

ZENODO_URL_TEMPLATE = "https://zenodo.org/records/1227121/files/{env}_16k.zip?download=1"
OUTPUT_DIR = Path("./data/noise/demand")

if __name__ == "__main__":
    with open("params.yaml", "r") as file:
        params = yaml.safe_load(file)
    environments = params["noise_pools"]["demand"]["environments"]

    clear_pool_dir(OUTPUT_DIR)
    for env in tqdm(environments, desc="Preparing DEMAND environments"):
        archive_path = download_archive(ZENODO_URL_TEMPLATE.format(env=env), f"{env}_16k.zip")
        with zipfile.ZipFile(archive_path) as archive:
            audio_bytes = archive.read(f"{env}/ch01.wav")
            # The 16k variant is already at target rate; writing through the
            # same resample path guards against a 48kHz archive slipping in.
            write_16k_mono_wav(io.BytesIO(audio_bytes), OUTPUT_DIR / f"{env}_ch01.wav")
