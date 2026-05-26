<!-- refreshed: 2026-05-26 -->
# Architecture

**Analysis Date:** 2026-05-26

## System Overview

```text
┌─────────────────────────────────────────────────────────────────────┐
│                      TRAINING SYSTEM (dev machine)                  │
│                                                                     │
│  Label Studio ──► fetch_data.sh ──► convert_labeled_data.py        │
│                                           │                         │
│                              download_audio.py                      │
│                                    │                                │
│                    ┌───────────────┼────────────────┐               │
│                    ▼               ▼                ▼               │
│          extract_data_quality   extract_mfcc     draw_data.py      │
│                    │            _features.py         │              │
│                    │                │                │              │
│                    ▼                ▼                ▼              │
│           data/data_quality/  data/mfcc_data/  data/balanced_      │
│                                                 data.h5             │
│                                                      │              │
│                                               train_xgboost.py     │
│                                                      │              │
│                                            models/xgboost_model    │
│                                                 .json               │
└──────────────────────────────────────────┬──────────────────────────┘
                                           │ deploy.sh (scp)
                                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 RUNTIME SYSTEM (Raspberry Pi Zero W)                │
│                                                                     │
│  seeed-2mic-voicecard ──► data_collector.py                        │
│                                   │                                 │
│                    ┌──────────────┼──────────────────┐              │
│                    │              │                  │              │
│             ML trigger      MQTT trigger      GPIO trigger          │
│           (XGBoost model)  (paho-mqtt)       (RPi.GPIO)             │
│                    │              │                  │              │
│                    └──────────────┴──────────────────┘              │
│                                   │                                 │
│                    ring buffer snapshot + post-trigger              │
│                                   │                                 │
│                         recordings/*.wav                            │
└─────────────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| fetch_data.sh | Export labeled CSV from Label Studio via REST API | `src/fetch_data.sh` |
| convert_labeled_data.py | Reshape Label Studio CSV from one-row-per-file to one-row-per-annotation | `src/convert_labeled_data.py` |
| download_audio.py | Download raw WAV files from Label Studio storage using annotation CSV | `src/download_audio.py` |
| extract_data_quality.py | Compute dataset quality metrics (class imbalance, total samples) | `src/extract_data_quality.py` |
| dataqualityutils.py | Shared helper: `get_data_quality_metrics()` used by multiple stages | `src/dataqualityutils.py` |
| extract_mfcc_features.py | Batch-extract MFCC features from all WAVs, write `.npy` per file | `src/extract_mfcc_features.py` |
| draw_data.py | Chunk annotations into windows, load MFCC slices, balance classes, write HDF5 | `src/draw_data.py` |
| train_xgboost.py | Train binary XGBoost classifier, log metrics via DVCLive, save model JSON | `src/train_xgboost.py` |
| make_background.py | Auto-annotate unannotated Label Studio tasks as background via SDK | `make_background.py` |
| data_collector.py | Runtime daemon: ring buffer, ML inference, MQTT/GPIO triggers, WAV saving | `data_collection/data_collector.py` |
| deploy.sh | SCP data_collector.py to the Raspberry Pi | `src/deploy.sh` |

## Pattern Overview

**Overall:** Two-subsystem ML project — a DVC-orchestrated training pipeline on a dev machine, and a standalone real-time detection daemon deployed to a Raspberry Pi.

**Key Characteristics:**
- Training is fully reproducible via DVC stages with hashed dependency tracking (`dvc.lock`)
- The training pipeline is a linear DAG: fetch → convert → download → (quality | features) → balance → train
- The runtime daemon is a single long-running Python process with a circular audio buffer and three independent trigger sources using `threading.Event` for synchronization
- The two systems share model parameters: training outputs `models/xgboost_model.json` and the runtime daemon loads that file at startup
- Feature extraction parameters (`n_mfcc`, `n_fft`, `chunk_size_ms`) must match between training and runtime; they are separately defined in `params.yaml` (training) and CLI flags / env file (runtime)

## Layers

**Data Ingestion Layer:**
- Purpose: Pull labeled data out of Label Studio
- Location: `src/fetch_data.sh`, `src/download_audio.py`
- Contains: Shell script for CSV export, Python script for WAV downloads
- Depends on: Label Studio REST API, env vars `LABEL_STUDIO_URL` and `API_KEY`
- Used by: DVC stages `fetch_labeled_data` and `download_audio`

**Data Preparation Layer:**
- Purpose: Convert raw Label Studio format to per-annotation rows; apply chunking and class balancing
- Location: `src/convert_labeled_data.py`, `src/draw_data.py`
- Contains: Pandas-based DataFrame reshaping, sliding-window chunking, background undersampling
- Depends on: `data/labeled_data.csv`, `data/mfcc_data/`, `params.yaml`
- Used by: DVC stages `convert_labeled_data`, `draw_data`

**Feature Extraction Layer:**
- Purpose: Transform WAV audio into MFCC feature matrices stored as NumPy arrays
- Location: `src/extract_mfcc_features.py`
- Contains: `librosa.feature.mfcc` calls, multiprocessing pool for batch processing
- Depends on: `data/audio/`, `params.yaml` (feature_extraction block)
- Used by: DVC stage `extract_features`; MFCC logic is duplicated inline in `data_collection/data_collector.py`

**Model Training Layer:**
- Purpose: Train and evaluate XGBoost binary classifier; track experiments with DVCLive
- Location: `src/train_xgboost.py`
- Contains: sklearn train/test split, XGBClassifier fit, classification report, confusion matrix plot
- Depends on: `data/balanced_data.h5`, `params.yaml` (model and training blocks)
- Used by: DVC stage `train_model`

**Data Quality Layer:**
- Purpose: Compute and persist class imbalance and sample count metrics used as DVC metrics
- Location: `src/extract_data_quality.py`, `src/dataqualityutils.py`
- Contains: Group-by-label counts, imbalance ratio calculation
- Depends on: `data/annotation_per_row_data.csv`
- Used by: DVC stage `extract_data_quality`; `draw_data.py` imports `dataqualityutils` directly

**Runtime Detection Layer:**
- Purpose: Continuously monitor microphone audio and save recordings when a doorbell is detected
- Location: `data_collection/data_collector.py`
- Contains: PyAudio stream, `collections.deque` ring buffer, XGBoost inference, MQTT client, GPIO callback
- Depends on: `models/xgboost_model.json` (optional), paho-mqtt (optional), RPi.GPIO (optional)
- Used by: systemd service on the Raspberry Pi

## Data Flow

### Training Pipeline (DVC)

1. Label Studio exports labeled CSV (`src/fetch_data.sh`) → `data/labeled_data.csv`
2. CSV reshaped to annotation-per-row format (`src/convert_labeled_data.py`) → `data/annotation_per_row_data.csv`
3. WAV files downloaded from Label Studio storage (`src/download_audio.py`) → `data/audio/` (644 files, ~291 MB)
4. Quality metrics computed in parallel (`src/extract_data_quality.py`) → `data/data_quality/`
5. Batch MFCC extraction (`src/extract_mfcc_features.py`) → `data/mfcc_data/*.npy` (one `.npy` per WAV)
6. Annotations chunked into 500 ms windows; MFCC slices loaded; background class undersampled (`src/draw_data.py`) → `data/balanced_data.h5`
7. XGBoost trained, metrics logged via DVCLive (`src/train_xgboost.py`) → `models/xgboost_model.json`

### Runtime Detection Loop (`data_collection/data_collector.py`)

1. PyAudio opens `seeed-2mic-voicecard` device at 16 kHz mono Int16
2. Every 500 ms audio chunk appended to `collections.deque` ring buffer (default 5-minute capacity)
3. **MQTT path:** `paho-mqtt` loop thread fires `threading.Event` on subscribed topic message → buffer snapshot saved to WAV
4. **GPIO path:** `RPi.GPIO` interrupt fires `threading.Event` on FALLING edge → buffer snapshot saved to WAV
5. **ML path:** Every 3rd chunk: `librosa.feature.mfcc` extracts features inline → `xgb.Booster.predict` → if probability > threshold, buffer snapshot saved to WAV
6. Post-trigger: additional `post_trigger_seconds` of audio appended before writing the WAV file

**State Management:**
- Ring buffer: module-level `collections.deque` with `maxlen`, written by main loop thread
- Trigger coordination: `threading.Event` shared between main loop and MQTT/GPIO callback threads
- `is_saving` boolean flag prevents overlapping concurrent saves (not thread-safe; GPIO/MQTT callbacks set the event but do not directly write, so saves are serialised through the main loop)

## Key Abstractions

**DVC Pipeline:**
- Purpose: Reproducible DAG of data transformation steps with file-level dependency tracking
- Examples: `dvc.yaml`, `dvc.lock`, `params.yaml`
- Pattern: Each stage declares `cmd`, `deps`, `params`, and `outs`; DVC caches outputs by MD5 hash

**Ring Buffer:**
- Purpose: Retain a rolling window of pre-trigger audio so detections can include context before the event
- Examples: `data_collection/data_collector.py` lines 303–304
- Pattern: `collections.deque(maxlen=buffer_chunks)` where `buffer_chunks = buffer_minutes * 60 / window_size_s`

**Trigger Event:**
- Purpose: Decouple out-of-band triggers (MQTT, GPIO) from the main audio loop
- Examples: `data_collection/data_collector.py` `setup_mqtt()`, `setup_gpio()`
- Pattern: `threading.Event` set by callbacks, checked and cleared in main loop

## Entry Points

**DVC pipeline:**
- Location: `dvc.yaml`
- Triggers: `dvc repro` command on dev machine
- Responsibilities: Orchestrates all training stages in dependency order

**Runtime daemon:**
- Location: `data_collection/data_collector.py` `main()`
- Triggers: systemd service start (`doorbell-collector.service`) or direct `python data_collector.py` invocation
- Responsibilities: Parse CLI args, set up audio, optionally load model, run continuous detection loop

**Label Studio bulk annotation:**
- Location: `make_background.py`
- Triggers: Manual invocation
- Responsibilities: Find unannotated tasks via Label Studio SDK and auto-label them as background

## Architectural Constraints

- **Threading:** Main loop is single-threaded; MQTT uses `paho-mqtt`'s `loop_start()` background thread; GPIO uses kernel interrupt callbacks. All share `trigger_event` and `is_saving` — `is_saving` is not protected by a lock, relying on GIL and the fact that saves are synchronous in the main loop.
- **Global state:** `CHANNELS`, `FORMAT`, `RATE` are module-level constants in `data_collection/data_collector.py`. `AUDIO_DATA_PATH`, `MFCC_FEATURES_FILE_BASE`, `AUDIO_FILE_BASE` are module-level `Path` constants in training scripts (hard-coded relative to CWD).
- **Circular imports:** None detected.
- **CWD dependency:** Training scripts in `src/` resolve data paths relative to the project root CWD (`./data/...`). They must be invoked from the project root, which DVC enforces. `data_collector.py` resolves all paths via CLI flags, making it CWD-independent.
- **Feature parameter coupling:** `n_mfcc`, `n_fft`, and `chunk_size_ms` must be identical between `params.yaml` (training) and the systemd env file / CLI flags (runtime). There is no shared config file between the two subsystems — this is a documented manual constraint.
- **Device name hard-coded:** The audio device name `"seeed-2mic-voicecard"` is hard-coded in `data_collection/data_collector.py` line 324. The process exits if that device is not found.

## Anti-Patterns

### Feature parameter duplication

**What happens:** MFCC parameters (`n_mfcc=13`, `n_fft=512`, `chunk_size=500`) are defined in `params.yaml` for training and separately specified as CLI defaults / env file values for the runtime daemon. There is no shared config source.

**Why it's wrong:** A change to `params.yaml` and model retraining will silently create a mismatch with the running daemon, causing degraded inference with no error at startup.

**Do this instead:** The deployment step (`src/deploy.sh`) should also copy `params.yaml` to the Pi and have `data_collector.py` read it as defaults, or the env file should be generated from `params.yaml` during `dvc repro`.

### Training scripts import from CWD

**What happens:** `src/draw_data.py` does `from dataqualityutils import get_data_quality_metrics` (bare import, not `src.dataqualityutils`), relying on `src/` being on `sys.path` due to CWD being the project root when DVC runs the stage.

**Why it's wrong:** Running the script directly from a different directory or importing it in tests requires manipulating `sys.path` (see `tests/convert_labeled_data_test.py` which imports `from convert_labeled_data import ...`).

**Do this instead:** Treat `src/` as a proper Python package with `__init__.py` and use relative or absolute package imports.

## Error Handling

**Strategy:** Minimal — most training scripts let exceptions propagate and terminate the DVC stage. The runtime daemon catches `OSError` on audio stream reads and reopens the stream.

**Patterns:**
- `data_collection/data_collector.py`: `try/except OSError` around `stream.read()` to handle buffer overflow by reopening the audio stream
- `data_collection/data_collector.py`: `try/except Exception` in `setup_mqtt()` around `client.connect()` — returns `None` and logs; MQTT trigger is silently disabled
- Optional dependencies (`paho-mqtt`, `RPi.GPIO`) wrapped in `try/except ImportError` with module-level booleans `MQTT_AVAILABLE`, `GPIO_AVAILABLE`
- Training scripts: no explicit error handling; failures surface as non-zero exit codes which DVC treats as stage failure

## Cross-Cutting Concerns

**Logging:** `data_collection/data_collector.py` uses Python `logging` module at INFO level with timestamp format. Training scripts use no logging — they print implicitly via tqdm progress bars and dvclive output.

**Validation:** No input validation in training scripts. `data_collector.py` validates `--model-path` existence and exits if the audio device is not found.

**Authentication:** Label Studio access uses `LABEL_STUDIO_URL` and `API_KEY` env vars (read in `src/fetch_data.sh` and `src/download_audio.py`). MQTT supports optional username/password and TLS client certificates configured via CLI flags passed from the systemd env file.

---

*Architecture analysis: 2026-05-26*
