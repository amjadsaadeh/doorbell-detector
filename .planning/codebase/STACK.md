# Technology Stack

**Analysis Date:** 2026-05-26

## Languages

**Primary:**
- Python 3 — all pipeline scripts, data collector, training, tests (system Python 3.12.3 on dev machine)

**Secondary:**
- Bash — data fetch script (`src/fetch_data.sh`), deploy script (`src/deploy.sh`)
- YAML — pipeline config (`params.yaml`), DVC pipeline definition (`dvc.yaml`)

## Runtime

**Environment:**
- Python virtualenv (`/home/pi/doorbell-detector/.venv` on the Raspberry Pi target)
- No pyenv / `.python-version` file — system Python is used

**Package Manager:**
- pip (inferred from `requirements.txt` structure)
- No lockfile beyond the pinned versions in `requirements.txt`

## Frameworks

**ML / Data Science:**
- `xgboost==2.1.3` — binary classifier (background vs bell), trained via `XGBClassifier`; served via `xgb.Booster` at inference time
- `scikit-learn==1.6.0` — train/test split, label encoding, classification metrics, confusion matrix
- `librosa==0.10.2.post1` — MFCC feature extraction from raw audio (`librosa.feature.mfcc`)
- `numpy==1.26.4` — core numerical arrays throughout pipeline and collector
- `pandas==2.2.3` — annotation CSV loading, balanced dataset assembly
- `pydub==0.25.1` — WAV parsing in feature extraction stage
- `pandarallel==1.6.5` — parallel `apply` on DataFrames in `draw_data.py`
- `numba==0.60.0` / `llvmlite==0.43.0` — JIT backend used internally by librosa

**ML Experiment Tracking:**
- `dvc==3.56.0` — pipeline orchestration, stage caching, data versioning (`dvc.yaml`, `dvc.lock`)
- `dvclive==3.48.1` — in-training metric/plot logging (`DVCLiveCallback`, `Live`)

**Audio I/O:**
- `PyAudio==0.2.14` — real-time microphone capture in `data_collector.py`

**Data Storage (HDF5):**
- `tables==3.10.1` (PyTables) — HDF5 read/write for `data/balanced_data.h5`
- `blosc2==3.0.0` — compression backend for PyTables

**HTTP / Networking:**
- `requests==2.32.3` — audio file downloads from Label Studio
- `aiohttp==3.10.10` / `aiohttp-retry==2.9.0` — async HTTP (DVC internals)

**Testing:**
- `pytest==8.3.3` — available but not the active runner; VSCode is configured to use `unittest`
- `unittest` (stdlib) — active test framework (`tests/*_test.py`)
- `requests-mock==1.12.1` — HTTP mocking in tests

**Code Quality:**
- `black==24.10.0` — code formatter

**Config:**
- `hydra-core==1.3.2` / `omegaconf==2.3.0` — available, not yet used directly in scripts
- `PyYAML==6.0.2` — `params.yaml` loading in all pipeline scripts

## Key Dependencies

**Production (runtime on Raspberry Pi — `data_collection/requirements.txt`):**

| Package | Version | Purpose |
|---------|---------|---------|
| `librosa` | 0.10.2.post1 | MFCC extraction during real-time inference |
| `numpy` | 1.26.4 | Audio buffer processing and feature arrays |
| `paho-mqtt` | 2.1.0 | MQTT broker client for external trigger |
| `PyAudio` | 0.2.14 | Microphone capture (PortAudio wrapper) |
| `PyYAML` | 6.0.2 | Config loading |
| `xgboost` | 2.1.3 | Model inference (`xgb.Booster.predict`) |
| `RPi.GPIO` | 0.7.1 | GPIO button trigger (Pi-only, commented out) |

**ML Training pipeline (root `requirements.txt`, dev/training machine):**
- All of the above, plus:
- `scikit-learn==1.6.0`, `pandas==2.2.3`, `pydub==0.25.1`, `pandarallel==1.6.5`
- `dvc==3.56.0`, `dvclive==3.48.1`
- `label-studio-sdk==1.0.7`
- `tables==3.10.1`, `blosc2==3.0.0` (HDF5 I/O)
- `matplotlib==5.24.1`, `plotly==5.24.1` (data visualisation in notebooks)
- `ipykernel==6.29.5`, `ipython==8.29.0` (Jupyter notebooks)

## Configuration

**Pipeline parameters:**
- `params.yaml` — single source of truth for chunk size, overlap, MFCC params, model hyperparameters
- Read directly with `yaml.safe_load` in every pipeline script

**Runtime (data collector):**
- All CLI flags; values injected via systemd `EnvironmentFile=/etc/doorbell-collector.env`
- Template env file: `data_collection/systemd/doorbell-collector.env`

**DVC remote:**
- Local path `../dvc_remote` (relative to repo root) — defined in `.dvc/config`

**Build:**
- No build step. Python scripts run directly.

## Platform Requirements

**Development (training machine):**
- Python 3 with pip
- HDF5 system libraries (for PyTables / `tables`)
- PortAudio (for PyAudio) only if running the collector locally

**Production (Raspberry Pi Zero W):**
- Raspberry Pi OS with ALSA/PortAudio for audio
- `seeed-2mic-voicecard` kernel driver and ALSA card (`data_collector.py` hard-codes this device name)
- systemd for service management
- Network access for optional MQTT broker connectivity

---

*Stack analysis: 2026-05-26*
