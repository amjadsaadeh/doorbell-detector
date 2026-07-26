"""Quantize the trained CNN to full-integer int8 for the ESP32-S3.

Two things make this more than a format conversion:

1. **Normalization is baked into the graph.** train_cnn.py stores per-bin
   mean/std in cnn_normalization.npz and applies them outside the model. Here
   they become a Rescaling layer in front of the trained network, so the
   firmware's C frontend only has to produce log-mel energies -- everything
   after that is the interpreter's job. One less place for train/serve skew.

2. **The calibration subset is a tracked artifact.** Post-training
   quantization picks activation ranges from whatever chunks it is shown, so
   that choice is part of the model, not a runtime detail. It is written to
   data/calibration/ as a DVC output and logged to the MLflow run of the
   evaluation stage, which makes an int8 model reproducible: same weights +
   same calibration chunks = same quantized graph.

Scoring the result lives in evaluate_quantized.py, so the thing that decides
whether the model is fit to ship is a separate stage from the thing that
produced it.
"""

import json
import shutil

import numpy as np
import pandas as pd
import yaml
from tensorflow import keras

from paths import (
    CALIBRATION_CHUNKS,
    CALIBRATION_DIR,
    CALIBRATION_MANIFEST,
    EXPORT_DIR,
    MANIFEST_PATH,
)
from tflite_utils import C_ARRAY_VAR, build_export_model, convert_int8, write_c_array
from dataset import load_dataset
from train_cnn import MODEL_PATH

NORMALIZATION_PATH = MODEL_PATH.with_name("cnn_normalization.npz")
TFLITE_PATH = EXPORT_DIR / "doorbell_int8.tflite"
C_ARRAY_PATH = EXPORT_DIR / "doorbell_model_data.cc"


def select_calibration_rows(candidate_idx: np.ndarray, n_samples: int, seed: int):
    """Pick calibration chunks from the rows the model was fitted on.

    Indices are dataset-global (positions in chunk_manifest.csv), not offsets
    into some re-indexed array, so the saved selection joins straight back to
    the manifest and names actual recordings.

    The shipped model is refit on 100% of the chunks (see train_cnn.main), so
    every row is a candidate. There is nothing to exclude: no held-out set
    exists for this model, which is exactly why evaluate_quantized.py reports
    a float-vs-int8 delta instead of an accuracy.
    """
    rng = np.random.default_rng(seed)
    n_samples = min(n_samples, len(candidate_idx))
    picked = rng.choice(candidate_idx, size=n_samples, replace=False)
    picked.sort()
    return picked


def write_calibration_set(X: np.ndarray, rows: np.ndarray) -> None:
    """Persist the calibration chunks plus the manifest rows they came from.

    The .npz is what a re-quantization needs; the .csv is what a human needs
    to see which recordings and SNRs the activation ranges were fitted to.
    """
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CALIBRATION_CHUNKS, X=X[rows].astype(np.float32), row_index=rows)

    manifest = pd.read_csv(MANIFEST_PATH)
    columns = [
        c
        for c in ["audio_file_name", "chunk_start", "chunk_end", "label", "split_group"]
        if c in manifest.columns
    ]
    selected = manifest.iloc[rows][columns].copy()
    selected.insert(0, "row_index", rows)
    selected.to_csv(CALIBRATION_MANIFEST, index=False)


def skip(reason: str) -> None:
    """Leave both declared outputs valid and self-explaining.

    The stage sits in the default DAG so every candidate is measured after
    quantization rather than before, but only the cnn head produces something
    convertible.
    """
    for directory in (EXPORT_DIR, CALIBRATION_DIR):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "SKIPPED.json").write_text(
            json.dumps({"skipped": True, "reason": reason}, indent=4)
        )
    print(f"quantization skipped: {reason}")


def main():
    with open("params.yaml", "r") as file:
        params = yaml.safe_load(file)

    for directory in (EXPORT_DIR, CALIBRATION_DIR):
        if directory.exists():
            shutil.rmtree(directory)

    head = params["training"]["head"]
    if head != "cnn":
        skip(f"training.head is {head!r}; int8 quantization needs the cnn head")
        return

    # load_dataset appends the channel axis, so X is
    # (n, n_bins, n_frames, 1) here, and un-normalized: the Rescaling layer
    # added below is what consumes raw features, exactly like the firmware.
    X, _, _, _, _ = load_dataset(params)

    trained = keras.models.load_model(MODEL_PATH)
    stats = np.load(NORMALIZATION_PATH)
    export_model = build_export_model(trained, stats["mean"], stats["std"])

    rows = select_calibration_rows(
        np.arange(len(X)),
        params["quantization"]["calibration_samples"],
        params["training"]["random_state"],
    )
    write_calibration_set(X, rows)

    tflite_model = convert_int8(export_model, X[rows])

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    TFLITE_PATH.write_bytes(tflite_model)
    write_c_array(tflite_model, C_ARRAY_PATH, C_ARRAY_VAR)

    print(f"calibrated on {len(rows)} of {len(X)} chunks -> {CALIBRATION_DIR}")
    print(f"int8 model {len(tflite_model) / 1024:.1f} KB -> {TFLITE_PATH}")


if __name__ == "__main__":
    main()
