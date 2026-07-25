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
