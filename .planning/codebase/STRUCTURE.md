# Codebase Structure

**Analysis Date:** 2026-05-26

## Directory Layout

```
doorbell-detector/
├── src/                          # Training pipeline scripts (run on dev machine)
│   ├── convert_labeled_data.py   # DVC stage: reshape Label Studio CSV
│   ├── dataqualityutils.py       # Shared helper: data quality metrics
│   ├── deploy.sh                 # SCP data_collector.py to Raspberry Pi
│   ├── download_audio.py         # DVC stage: download WAVs from Label Studio
│   ├── draw_data.py              # DVC stage: chunk, balance, assemble HDF5
│   ├── extract_data_quality.py   # DVC stage: compute and save quality JSON
│   ├── extract_mfcc_features.py  # DVC stage: batch MFCC extraction to .npy
│   ├── fetch_data.sh             # DVC stage: export labeled CSV from Label Studio
│   └── train_xgboost.py          # DVC stage: train XGBoost, log metrics, save model
│
├── data_collection/              # Self-contained Raspberry Pi runtime sub-project
│   ├── data_collector.py         # Runtime daemon: ring buffer + multi-trigger detection
│   ├── requirements.txt          # Minimal Pi dependencies (librosa, xgboost, pyaudio…)
│   └── systemd/
│       ├── doorbell-collector.env    # Template env file (copy to /etc/ on Pi)
│       └── doorbell-collector.service # systemd unit file for the daemon
│
├── tests/                        # Unit tests (run on dev machine)
│   ├── convert_labeled_data_test.py  # Tests for convert_labeled_data.py
│   ├── feature_extraction_test.py    # Tests for extract_mfcc_features.py
│   └── data/                     # Test fixtures
│       ├── audio/                # Sample WAV files (test_audio.wav, 2, 3)
│       ├── tmp_mfcc_data/        # Precomputed .npy fixtures for feature tests
│       ├── labeled_data.csv      # Minimal Label Studio export fixture
│       └── annotation_per_row_data.csv  # Expected conversion output fixture
│
├── notebooks/                    # Exploratory analysis (not part of DVC pipeline)
│   ├── basic_data_alaysis.ipynb  # Initial dataset exploration
│   ├── dataset_distributions.ipynb  # Label distribution plots
│   └── README.md
│
├── data/                         # All pipeline data (mostly DVC-tracked, not committed)
│   ├── labeled_data.csv          # Label Studio CSV export (DVC-tracked)
│   ├── annotation_per_row_data.csv  # Per-annotation rows (DVC-tracked)
│   ├── audio/                    # Raw WAV files, 644 files ~291 MB (DVC-tracked)
│   ├── mfcc_data/                # Per-WAV .npy MFCC arrays (DVC-tracked)
│   ├── balanced_data.h5          # Balanced chunked dataset for training (DVC-tracked)
│   └── data_quality/             # Quality metrics JSON and CSV (DVC metrics outputs)
│       ├── sample_based_quality.json
│       ├── samples_per_label.csv
│       ├── chunk_balanced_quality.json
│       └── chunks_per_label.csv
│
├── models/                       # Trained model artifacts (DVC-tracked, not committed)
│   └── xgboost_model.json        # Serialized XGBoost model (deployed to Pi)
│
├── dvclive/                      # DVCLive experiment tracking outputs (auto-generated)
│   ├── metrics.json
│   └── plots/
│       ├── metrics/              # Per-epoch metric curves (step vs. value)
│       └── sklearn/
│           └── confusion_matrix.json
│
├── .dvc/                         # DVC internals
│   └── config                    # Remote: url = ../dvc_remote (local relative path)
│
├── .planning/                    # GSD planning documents
│   └── codebase/                 # Codebase map documents
│
├── dvc.yaml                      # Pipeline stage definitions (the source of truth for stages)
├── dvc.lock                      # Hashed dependency snapshot (commit alongside dvc.yaml)
├── params.yaml                   # Hyperparameters consumed by DVC stages
├── make_background.py            # Utility: auto-label unannotated tasks as background in Label Studio
├── requirements.txt              # Full dev machine Python dependencies (pinned)
├── README.md                     # Project overview and hardware description
├── ROADMAP.md                    # Backlog and model strategy notes
├── .dvcignore                    # Files DVC should not track
└── .gitignore                    # Ignores: .python-version, *.pyc, .envrc, dvc_remote
```

## Directory Purposes

**`src/`:**
- Purpose: All training pipeline Python scripts and shell scripts
- Contains: One file per DVC stage plus shared helpers; scripts are invoked by DVC `cmd` fields
- Key files: `train_xgboost.py`, `extract_mfcc_features.py`, `draw_data.py`, `dataqualityutils.py`
- Note: Scripts resolve data paths relative to project root CWD (`./data/...`); always run via `dvc repro` or from the project root

**`data_collection/`:**
- Purpose: Self-contained sub-project for the Raspberry Pi runtime; intentionally decoupled from the training pipeline
- Contains: Single detection daemon, its own `requirements.txt`, systemd deployment config
- Key files: `data_collector.py`, `systemd/doorbell-collector.env`, `systemd/doorbell-collector.service`
- Note: Has its own minimal `requirements.txt` separate from the root `requirements.txt`

**`data/`:**
- Purpose: All pipeline data artifacts; the directory exists in git but most contents are DVC-tracked (not committed)
- Contains: CSV label files, WAV audio, NumPy MFCC arrays, HDF5 balanced dataset, JSON quality metrics
- Key files: `balanced_data.h5` (training input), `annotation_per_row_data.csv` (intermediate)
- Generated: Yes, by DVC stages; restore with `dvc pull`

**`models/`:**
- Purpose: Trained model output; committed path tracked by DVC, binary not committed to git
- Contains: `xgboost_model.json` — the artifact deployed to the Pi
- Generated: Yes, by the `train_model` DVC stage

**`dvclive/`:**
- Purpose: Auto-generated experiment tracking artifacts created by `dvclive` during `train_xgboost.py`
- Contains: `metrics.json`, per-epoch metric plots, sklearn confusion matrix JSON
- Generated: Yes, not manually edited; committed to git to enable `dvc metrics show` and `dvc plots show`

**`tests/`:**
- Purpose: Unit tests with fixture data
- Contains: Two test files (unittest-based), WAV fixtures, CSV fixtures, precomputed `.npy` fixtures
- Key files: `convert_labeled_data_test.py`, `feature_extraction_test.py`

**`notebooks/`:**
- Purpose: Ad-hoc exploratory analysis; not part of the DVC pipeline
- Contains: Jupyter notebooks for data distribution exploration

## Key File Locations

**Entry Points:**
- `dvc.yaml`: Pipeline definition — the primary entry point for all training operations
- `data_collection/data_collector.py`: Runtime daemon entry point (`if __name__ == "__main__": main()`)
- `make_background.py`: Utility script for bulk Label Studio annotation

**Configuration:**
- `params.yaml`: All training hyperparameters (chunk_size, chunk_overlap, inbalance_ratio, feature_extraction, training, model)
- `data_collection/systemd/doorbell-collector.env`: Runtime configuration template for the Pi; maps to CLI flags in `data_collector.py`
- `data_collection/systemd/doorbell-collector.service`: systemd unit; references `/etc/doorbell-collector.env` as `EnvironmentFile`
- `.dvc/config`: DVC remote storage location (currently a local relative path `../dvc_remote`)

**Core Logic:**
- `src/train_xgboost.py`: Model training with DVCLive experiment tracking
- `src/draw_data.py`: Most complex stage — chunking, balancing, MFCC slice assembly
- `src/extract_mfcc_features.py`: Feature extraction with multiprocessing pool
- `data_collection/data_collector.py`: Real-time detection with ring buffer and three trigger sources
- `src/dataqualityutils.py`: Shared quality metric helper imported by both `extract_data_quality.py` and `draw_data.py`

**Pipeline Coordination:**
- `dvc.yaml`: Stage definitions (cmd, deps, params, outs)
- `dvc.lock`: Hashed dependency state (auto-updated by `dvc repro`, always commit)

**Testing:**
- `tests/convert_labeled_data_test.py`: Tests `annotation_to_sample_per_row()` against fixture CSVs
- `tests/feature_extraction_test.py`: Tests `process_audio_data()` and `extract_mfcc_features()` against fixture WAVs

## Naming Conventions

**Files:**
- Training scripts: `snake_case.py` with descriptive action names (`extract_mfcc_features.py`, `train_xgboost.py`)
- Test files: `{module_name}_test.py` (suffix pattern, not `test_` prefix)
- Data artifacts: `snake_case` with format extension (`.csv`, `.h5`, `.npy`, `.json`)
- Shell scripts: `snake_case.sh`

**Directories:**
- All lowercase, underscore-separated (`data_collection/`, `data_quality/`, `mfcc_data/`)
- Data sub-directories named after content type

**DVC stage names in `dvc.yaml`:**
- Verb-noun pattern: `fetch_labeled_data`, `convert_labeled_data`, `download_audio`, `extract_features`, `draw_data`, `train_model`

## Where to Add New Code

**New DVC pipeline stage:**
- Implementation script: `src/new_stage_name.py`
- Register stage in: `dvc.yaml` with `cmd`, `deps`, `params`, `outs`
- Add parameters to: `params.yaml` if the stage is parameterized
- Run with: `dvc repro new_stage_name` to execute only that stage

**New shared training utility:**
- Add to: `src/dataqualityutils.py` if related to data quality, or create `src/newutils.py`
- Import pattern (match existing style): `from dataqualityutils import function_name` when called from `src/` scripts running with project root as CWD

**New feature for the runtime daemon:**
- All runtime code belongs in: `data_collection/data_collector.py`
- Expose new behavior via CLI argument in `parse_args()`, then wire into `main()`
- Document new CLI flags in: `data_collection/systemd/doorbell-collector.env` as commented examples

**New tests:**
- Location: `tests/{module_name}_test.py`
- Fixture data: `tests/data/` (audio fixtures in `tests/data/audio/`, CSV fixtures directly in `tests/data/`)
- Pattern: `unittest.TestCase` subclass; use `Path(__file__).parent / 'data'` for fixture paths

**New model type (replacing or alongside XGBoost):**
- Training script: `src/train_{model_name}.py`
- Add DVC stage to `dvc.yaml` pointing to that script
- Output to: `models/{model_name}_model.{ext}`
- Update `data_collection/data_collector.py` to load and run inference with the new model format

## Special Directories

**`data/` (DVC-tracked contents):**
- Purpose: All pipeline data artifacts
- Generated: Yes — by `dvc repro` stages
- Committed: Only the directory structure and `.gitignore`; actual data files retrieved via `dvc pull`

**`models/`:**
- Purpose: Trained model files
- Generated: Yes — by `train_model` DVC stage
- Committed: Only `.gitignore`; model binary tracked by DVC, retrieved via `dvc pull`

**`dvclive/`:**
- Purpose: Experiment tracking outputs written by DVCLive during training
- Generated: Yes — auto-created/updated each time `train_xgboost.py` runs
- Committed: Yes — committed to git so that `dvc metrics show` and `dvc plots show` work without re-running training

**`.dvc/`:**
- Purpose: DVC internal configuration; the `config` file sets the remote storage location
- Generated: Partially (cache sub-directory); `config` is hand-edited
- Committed: Yes

**`.planning/`:**
- Purpose: GSD planning and codebase map documents
- Generated: By GSD tooling
- Committed: Yes

---

*Structure analysis: 2026-05-26*
