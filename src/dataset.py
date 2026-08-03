"""Reading and writing the chunk dataset both trainers consume.

The features used to live in `balanced_data.h5` as a pandas column of
per-row numpy arrays, alongside the chunk metadata. That column is
object-dtype, so PyTables pickled it rather than mapping it to c-types (it
warns about exactly this), and the pickled bytes are not stable: two runs
over identical data produced identical *content* and different *files*. DVC
hashes files, so draw_data always looked changed and train_model could never
be skipped -- roughly 20 minutes per `dvc repro`, permanently.

So the features are now one contiguous float32 dataset written with h5py,
which is byte-reproducible, and the metadata is not duplicated here at all:
chunk_manifest.csv already carries it, row for row, in the same order.
"""

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from features import get_spec
from paths import DATA_FILE, MANIFEST_PATH

FEATURES_KEY = "features"
FEATURE_TYPE_ATTR = "feature_type"


def write_dataset(features: np.ndarray, feature_type: str, path: Path = DATA_FILE) -> None:
    """One dataset, one attribute, no metadata copy.

    `features` is (n_chunks, ...) and must be row-aligned with
    chunk_manifest.csv -- that alignment is the whole contract between this
    file and the manifest.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset(FEATURES_KEY, data=features.astype(np.float32))
        handle.attrs[FEATURE_TYPE_ATTR] = feature_type


def read_features(path: Path = DATA_FILE) -> tuple[np.ndarray, str]:
    with h5py.File(path, "r") as handle:
        return handle[FEATURES_KEY][...], handle.attrs[FEATURE_TYPE_ATTR]


def binarize_labels(labels: pd.Series):
    """Everything that is not background is a bell.

    LabelEncoder orders alphabetically, so background=0, bell=1.
    """
    collapsed = labels.apply(lambda x: "background" if x == "background" else "bell")
    encoder = LabelEncoder()
    return encoder.fit_transform(collapsed), encoder


def load_dataset(params: dict, flatten: bool = False):
    """Chunks, labels and split groups, ready for either head.

    flatten=True gives the tree head (n, features); otherwise a trailing
    channel axis is added for the CNN. Returns the feature type too, since
    both trainers name their MLflow run after it.
    """
    X, feature_type = read_features()
    manifest = pd.read_csv(MANIFEST_PATH)

    if len(X) != len(manifest):
        raise SystemExit(
            f"{DATA_FILE} has {len(X)} chunks but {MANIFEST_PATH} has "
            f"{len(manifest)}. They are row-aligned by construction, so one of "
            f"them is stale -- re-run the draw_data stage."
        )

    # Whether this is needed is a property of the representation, declared in
    # features.py, not a flag someone has to remember to flip.
    if get_spec(feature_type).needs_log_compression:
        X = np.log(X + 1e-6)

    y, encoder = binarize_labels(manifest["label"])
    groups = manifest["split_group"].to_numpy()

    X = X.reshape(len(X), -1) if flatten else X[..., np.newaxis]
    return X, y, encoder, feature_type, groups
