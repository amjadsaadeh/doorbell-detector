"""Attach features to the chunks select_chunks.py already picked.

Everything about *which* chunks make up the dataset moved to
select_chunks.py; what is left here is purely "slice the arrays and write
the HDF5". That split is what lets one sampling decision be shared by every
feature variant.

Output is a single contiguous float32 array row-aligned with
chunk_manifest.csv, written by dataset.py -- which explains why the chunk
metadata is deliberately not duplicated into it.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pandarallel import pandarallel
from tqdm import tqdm

from dataset import write_dataset
from features import get_spec, verify_feature_config
from paths import DATA_FILE, MANIFEST_PATH


def load_features(audio_file_name: str, features_dir: Path) -> np.ndarray:
    return np.load(features_dir / Path(audio_file_name).with_suffix(".npy").name)


def main():
    tqdm.pandas()
    pandarallel.initialize(progress_bar=True)

    with open("params.yaml", "r") as file:
        params = yaml.safe_load(file)

    feature_params = params["feature_extraction"]
    features_dir = Path(feature_params["features_dir"])
    feature_type = feature_params["type"]
    spec = get_spec(feature_type)

    # Refuse to slice arrays that were built by a different configuration
    # than the one params.yaml currently asks for.
    verify_feature_config(features_dir, feature_params)

    balanced_df = pd.read_csv(MANIFEST_PATH)

    chunks = balanced_df.parallel_apply(
        lambda row: spec.slice_chunk(
            load_features(row["audio_file_name"], features_dir),
            row["chunk_start"],
            row["chunk_end"],
            feature_params,
        ),
        axis=1,
    )

    shapes = {f.shape for f in chunks}
    if len(shapes) != 1:
        raise SystemExit(
            f"inconsistent chunk shapes from {features_dir}: {sorted(shapes)}. "
            f"The trainer stacks these, so they all have to agree."
        )

    # Row-aligned with chunk_manifest.csv, which is the only place the chunk
    # metadata lives now -- see dataset.py for why it is not duplicated here.
    write_dataset(np.stack(chunks.to_numpy()), feature_type)
    print(f"{len(balanced_df)} chunks of shape {shapes.pop()} -> {DATA_FILE}")


if __name__ == "__main__":
    main()
