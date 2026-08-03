"""Extract one feature representation for exactly the files the manifest needs.

Replaces extract_mfcc_features.py / extract_stft_features.py /
extract_logmel_features.py / extract_yamnet_features.py, which were four
copies of the same driver around different transforms. The transforms now
live in features.py keyed by `feature_extraction.type`; this file is only
the driver.

Two behaviour changes over those scripts:

* It is manifest-driven. They globbed every .wav in data/audio,
  data/augmented_audio and every external noise pool -- ~1900 noise files to
  serve the few hundred background chunks that get drawn. This extracts the
  files select_chunks.py actually referenced.
* It writes a feature_config.json sidecar next to the arrays, recording the
  configuration that produced them, which draw_data.py then verifies.
"""

import functools
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm
import yaml

from audio_sources import resolve_audio_path
from features import get_spec, write_feature_config
from paths import MANIFEST_PATH


def _process_one(audio_file_name: str, features_dir: Path, feature_type: str, cfg: dict):
    spec = get_spec(feature_type)
    array = spec.extract(resolve_audio_path(audio_file_name), **cfg)
    np.save(features_dir / (Path(audio_file_name).stem + ".npy"), array)


def extract_all(
    audio_file_names: list[str], features_dir: Path, feature_type: str, cfg: dict
) -> None:
    features_dir.mkdir(parents=True, exist_ok=True)
    spec = get_spec(feature_type)
    desc = f"Extracting {feature_type} features"

    if not spec.parallel:
        # YAMNet: TF is not fork-safe, and every Pool worker would re-load
        # the model. Run serially against one loaded model instead.
        context = spec.context() if spec.context else {}
        for name in tqdm.tqdm(audio_file_names, desc=desc):
            array = spec.extract(resolve_audio_path(name), **context, **cfg)
            np.save(features_dir / (Path(name).stem + ".npy"), array)
        return

    worker = functools.partial(
        _process_one, features_dir=features_dir, feature_type=feature_type, cfg=cfg
    )
    with Pool() as pool:
        list(
            tqdm.tqdm(
                pool.imap(worker, audio_file_names),
                total=len(audio_file_names),
                desc=desc,
            )
        )


def main():
    with open("params.yaml", "r") as file:
        params = yaml.safe_load(file)

    feature_params = params["feature_extraction"]
    feature_type = feature_params["type"]
    features_dir = Path(feature_params["features_dir"])
    spec = get_spec(feature_type)
    cfg = {k: feature_params[k] for k in spec.params}

    manifest = pd.read_csv(MANIFEST_PATH)
    audio_file_names = sorted(manifest["audio_file_name"].unique())

    extract_all(audio_file_names, features_dir, feature_type, cfg)
    write_feature_config(features_dir, feature_params)

    print(f"{len(audio_file_names)} files -> {features_dir} ({feature_type})")


if __name__ == "__main__":
    main()
