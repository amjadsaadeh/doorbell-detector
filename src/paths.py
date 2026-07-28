"""Canonical artifact locations, so stage scripts don't import each other
just to agree on a filename.

Model outputs are directories rather than fixed filenames because the
trained artifact differs by head (cnn_model.keras + normalization npz vs
xgboost_model.json) and DVC needs one static `outs` entry either way.
"""

from pathlib import Path

MANIFEST_PATH = Path("./data/chunk_manifest.csv")
DATA_FILE = Path("./data/balanced_data.h5")
DATA_QUALITY_DIR = Path("./data/data_quality")

MODEL_DIR = Path("./models/trained")
EXPORT_DIR = Path("./models/export")

# The chunks post-training quantization fits its activation ranges to. Tracked
# by DVC because that choice is part of the quantized model, not a runtime
# detail: same weights + same calibration chunks reproduce the same graph.
CALIBRATION_DIR = Path("./data/calibration")
CALIBRATION_CHUNKS = CALIBRATION_DIR / "calibration_chunks.npz"
CALIBRATION_MANIFEST = CALIBRATION_DIR / "calibration_manifest.csv"

QUANTIZED_METRICS = Path("./models/quantized_metrics.json")

# Held-out per-chunk predictions from the training run's CV folds. A DVC
# output of train_model for the same reason the calibration set is one: it
# can only be produced inside the fold loop, so once that run is over the
# only way back to it is the recorded artifact.
PREDICTIONS_DIR = Path("./data/predictions")
OOF_PREDICTIONS = PREDICTIONS_DIR / "oof_predictions.csv"

# Working directory for the Spotlight inspector (src/inspect_dataset.py):
# one wav per chunk plus the table pointing at them. Derived, disposable and
# rebuildable from the manifest, so it is git-ignored and *not* DVC-tracked.
SPOTLIGHT_DIR = Path("./data/spotlight")
SPOTLIGHT_TABLE = SPOTLIGHT_DIR / "inspection.parquet"
SPOTLIGHT_AUDIO_DIR = SPOTLIGHT_DIR / "chunks"
