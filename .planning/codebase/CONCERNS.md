# Codebase Concerns

**Analysis Date:** 2026-05-26

---

## High Severity

### Broken Unit Tests — Feature Extraction

**Issue:** `tests/feature_extraction_test.py` calls `extract_mfcc_features` with 5 positional arguments (`path, 0, 1000, 13, 512`) but the function signature in `src/extract_mfcc_features.py` only accepts 3 parameters (`file_path, n_mfcc, n_fft`). The tests test a function interface that no longer exists.
**Files:** `tests/feature_extraction_test.py` (lines 40, 46, 50), `src/extract_mfcc_features.py` (line 16)
**Impact:** Running `pytest` will fail on these tests. The test suite cannot be trusted to gate regressions — any CI step that checks tests will silently fail or be skipped entirely.
**Fix approach:** Reconcile the function signature with the tests. If `start`/`end` slicing was intentionally removed from the function, rewrite the tests. If it is still needed, restore the parameters.

---

### DVC Remote Configured as Local Relative Path — Breaks on Fresh Clone

**Issue:** `.dvc/config` sets `url = ../dvc_remote`, a path relative to the repository root. On any machine where `../dvc_remote` does not exist (any fresh clone, CI, collaborator's machine), every DVC pull/push/repro will fail with an opaque "remote not found" error.
**Files:** `.dvc/config` (line 5)
**Impact:** The reproducibility pipeline (`dvc repro`) cannot be executed from scratch without the local sibling directory. The ROADMAP already identifies this as a fragility fix (item 6).
**Fix approach:** Migrate to a persistent remote — DagsHub, self-hosted MinIO, or an S3-compatible bucket — and update `.dvc/config` accordingly.

---

### MQTT Password Passed as CLI Argument (Process-Visible)

**Issue:** `--mqtt-password` is accepted as a plain CLI argument in `data_collection/data_collector.py` (line 246). On Linux, process arguments are visible to all users via `/proc/<pid>/cmdline` and `ps aux`. The template env file also shows the password in plaintext: `MQTT_AUTH_ARGS=--mqtt-username myuser --mqtt-password s3cr3t`.
**Files:** `data_collection/data_collector.py` (lines 241–248), `data_collection/systemd/doorbell-collector.env` (line 82)
**Impact:** On a shared or multi-user system, the MQTT password is exposed to any process that reads `/proc`. The env file is a workaround (env vars are not visible to other users by default), but the comment in the env file example embeds the password in plaintext.
**Fix approach:** Accept the password via a dedicated env var read inside the script (`os.getenv("MQTT_PASSWORD")`) rather than a CLI flag, or read it from a credentials file referenced by path.

---

## Medium Severity

### Stale Deploy Script References Wrong Path

**Issue:** `src/deploy.sh` copies `src/data_collector.py` but the file was moved to `data_collection/data_collector.py` in commit `9c32927`. The script also copies `params.yaml`, which `data_collector.py` no longer reads (it was refactored to use CLI flags only).
**Files:** `src/deploy.sh` (lines 5–6)
**Impact:** Running `bash src/deploy.sh` will `scp` a non-existent file and silently fail, leaving the Pi running stale code.
**Fix approach:** Update the path to `data_collection/data_collector.py` and remove the `params.yaml` line (or replace with `data_collection/requirements.txt`).

---

### `ssl.PROTOCOL_TLS` Deprecated Since Python 3.10

**Issue:** `data_collection/data_collector.py` line 106 passes `tls_version=ssl.PROTOCOL_TLS` to `client.tls_set()`. `ssl.PROTOCOL_TLS` was deprecated in Python 3.10 and removed in Python 3.12. On Python 3.12+ this raises `AttributeError` at runtime, silently disabling TLS.
**Files:** `data_collection/data_collector.py` (line 106)
**Impact:** MQTT TLS connections will fail on Python 3.12+ without a clear error message.
**Fix approach:** Replace with `ssl.PROTOCOL_TLS_CLIENT` (or remove the `tls_version` argument entirely, as paho-mqtt defaults to a sensible TLS version).

---

### No HTTP Error Handling in Audio Download

**Issue:** `src/download_audio.py` calls `requests.get()` and writes `response.content` to disk without checking the HTTP status code. A 401, 404, or 500 response will silently write an HTML error page into a `.wav` file, corrupting the dataset.
**Files:** `src/download_audio.py` (lines 45–50)
**Impact:** Corrupted audio files in `data/audio/` will cause silent failures or skewed metrics in later pipeline stages, which are hard to trace back to this step.
**Fix approach:** Add `response.raise_for_status()` before `f.write(response.content)`.

---

### Training Parameters Not Validated Against Inference Parameters

**Issue:** `params.yaml` defines `feature_extraction.n_mfcc: 13` and `feature_extraction.n_fft: 512`. The training pipeline uses these values. The inference script `data_collection/data_collector.py` has separate CLI defaults (`--n-mfcc 13`, `--n-fft 512`) with no link to `params.yaml`. If either side is changed independently, the model will receive malformed feature vectors and produce garbage predictions — without any error.
**Files:** `params.yaml`, `data_collection/data_collector.py` (lines 196–210), `src/extract_mfcc_features.py`
**Impact:** Silently incorrect predictions in production after a retraining run that changes feature parameters.
**Fix approach:** Embed the feature extraction parameters into the saved model artifact (e.g., as XGBoost custom attributes or a sidecar JSON), and validate them at inference startup.

---

### Training Uses a Fixed `random_state=42` With No Cross-Validation

**Issue:** `src/train_xgboost.py` uses a single `train_test_split` with `random_state=42`. There is no k-fold or repeated cross-validation. The reported 95% F1 is from one split and may be optimistic.
**Files:** `src/train_xgboost.py` (lines 36–47)
**Impact:** Metrics logged to DVCLive may not generalise. A different random seed could show significantly different performance.
**Fix approach:** Add stratified k-fold cross-validation or at minimum evaluate multiple seeds and report the mean and standard deviation.

---

### `is_saving` Flag Is Not Thread-Safe

**Issue:** In `data_collection/data_collector.py`, `is_saving` is a plain Python `bool` used to prevent overlapping saves (lines 362, 377–385, 403–411). The MQTT callback runs in a separate thread started by `paho-mqtt`'s `loop_start()`. Although the GIL makes single assignments atomic in CPython, the check-then-set idiom (`if not is_saving: ... is_saving = True`) is not atomic and could race in theory.
**Files:** `data_collection/data_collector.py` (lines 362, 377–411)
**Impact:** Low probability under normal use because `save_audio` is synchronous and blocks the main loop. However, the pattern is fragile and would break under any future async refactor.
**Fix approach:** Replace `is_saving` with a `threading.Lock()` or `threading.Event()`.

---

### `make_background.py` Is an Undocumented Utility Script with a Hardcoded Project ID

**Issue:** `make_background.py` at the repository root auto-annotates Label Studio tasks with a hardcoded `project=1` (line 29) and a hardcoded label span (`start: 0, end: 7`). It is not wired into the DVC pipeline, has no docstring explaining when to run it, and could silently overwrite or corrupt annotations if run at the wrong time.
**Files:** `make_background.py` (lines 29, 30–44)
**Impact:** Accidental execution would bulk-annotate unannotated clips with a `background` label, potentially polluting the training dataset.
**Fix approach:** Add a docstring/comment warning, require a `--dry-run` / `--confirm` flag, or move the script into a clearly labelled `scripts/` or `tools/` directory with documentation.

---

## Low Severity

### No `pytest.ini` / `pyproject.toml` Test Configuration

**Issue:** There is no pytest configuration file. Running `pytest` from the repository root will not find tests in `tests/` without additional path configuration, and `convert_labeled_data_test.py` imports `from convert_labeled_data import ...` (bare module name) which requires `src/` to be on `sys.path`.
**Files:** `tests/convert_labeled_data_test.py` (line 2), `tests/feature_extraction_test.py` (line 5)
**Impact:** Tests may fail depending on the working directory from which `pytest` is invoked.
**Fix approach:** Add a `pytest.ini` or `[tool.pytest.ini_options]` in `pyproject.toml` that sets `testpaths = ["tests"]` and adds `src/` to `pythonpath`.

---

### `requirements.txt` Is a Fully-Pinned Dump With No Separation of Dev vs. Runtime

**Issue:** The top-level `requirements.txt` (180 lines) pins every transitive dependency including `ipykernel`, `debugpy`, `jupyter_client`, `black`, `pytest`, and `celery` — development and runtime dependencies mixed together. `celery` is not used anywhere in the codebase.
**Files:** `requirements.txt`
**Impact:** Bloated installs; any `pip install -r requirements.txt` on the Pi or in CI installs hundreds of megabytes of unused packages. The Pi Zero W has limited storage and RAM.
**Fix approach:** Split into `requirements.txt` (runtime only) and `requirements-dev.txt` (development tools), or use `pyproject.toml` with optional dependency groups.

---

### `data_collection/requirements.txt` Omits `RPi.GPIO` (Commented Out)

**Issue:** `data_collection/requirements.txt` leaves `RPi.GPIO` commented out (line 13). On the Raspberry Pi, installing from this file leaves GPIO support missing, so the GPIO trigger silently falls back to "disabled" with only a log warning.
**Files:** `data_collection/requirements.txt` (line 13)
**Impact:** Operators following the install instructions will not notice the GPIO trigger is inactive unless they read logs carefully.
**Fix approach:** Document the Pi-specific install step explicitly, or use an install extra / separate requirements file for Pi deployment.

---

### Hardcoded `seeed-2mic-voicecard` Device Name

**Issue:** `data_collection/data_collector.py` searches for an audio device named `seeed-2mic-voicecard` (line 324) and exits if not found. There is no CLI flag or environment variable to override the device name or index.
**Files:** `data_collection/data_collector.py` (lines 322–331)
**Impact:** The collector cannot be tested on a dev machine with a regular USB mic or adapted for a different Pi hat without code changes.
**Fix approach:** Expose `--audio-device` as a CLI flag (name substring or device index) with the current value as default.

---

### No Notification Output After Detection

**Issue:** The system detects the doorbell and saves a WAV file, but there is no outgoing notification (push notification, MQTT publish, Home Assistant event). The original use case (phone notification when doorbell rings) is not yet closed.
**Files:** `data_collection/data_collector.py`
**Impact:** The core user value — alerting when away from the door — is not delivered by the current code.
**Fix approach:** On ML or MQTT/GPIO trigger, publish an MQTT message to an outbound topic; integrate with Home Assistant or a push notification service (e.g., ntfy, Pushover).

---

### One TODO Comment — Unresolved Imputation Strategy

**Issue:** `src/draw_data.py` line 79 has `# TODO try imputations` next to where non-background samples are separated. No imputation logic is implemented.
**Files:** `src/draw_data.py` (line 79)
**Impact:** Low — data balancing currently works by random undersampling. Imputation would be an enhancement, not a fix.
**Fix approach:** Implement or remove the comment.

---

*Concerns audit: 2026-05-26*
