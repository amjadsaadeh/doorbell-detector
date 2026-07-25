# Doorbell Detector — Project Guide

## What This Is

Audio-based doorbell detection system running on a Raspberry Pi. The project has two
main modes: an ML-based collector (`data_collection/data_collector.py`) and a
pattern-matching detector (`data_collection/detector.py`), deployed as a systemd
service. Alongside the Pi scripts lives a DVC-managed ML pipeline (`src/`, `dvc.yaml`)
that pulls labels from Label Studio and raw audio from a self-hosted MinIO S3 bucket
to train an XGBoost bell classifier.

## ML Data Pipeline (DVC)

`uv run dvc repro` runs the full chain:
`fetch_labeled_data` (Label Studio CSV export) → `convert_labeled_data` →
`download_audio` (S3) → `extract_data_quality` / `augmentation` (SNR-mixed synthetic
`front_doorbell` samples) → `select_chunks` (chunking + balancing) →
`extract_features` → `draw_data` (slicing) → `train_model` (MLflow) →
`export_model` (int8 TFLite).

- **A feature/model variant is a parameter, not a branch.** `feature_extraction.type`
  picks an entry in `src/features.py` (`mfcc`, `logmel`, `stft`, `yamnet`) which owns
  both the transform and the matching chunk slicer; `training.head` picks `cnn` or
  `xgboost`. Sweep with `dvc exp run -S`, not by forking dvc.yaml. Historic branches
  (`cnn-mfcc`, `cnn-spectrogram`, `yamnet-features`, `stft-spectrogram-features`,
  `cnn-logmel`, `esp32-logmel`) each hard-coded one combination; they are superseded.
- **`select_chunks` is deliberately feature-independent.** It decides which chunks make
  up the dataset and in what order, writes `data/chunk_manifest.csv`, and takes no
  `feature_extraction` param — so every variant trains on exactly the same chunks and
  reuses this stage's cache. `extract_features` is downstream of it and touches only
  the files the manifest references (1205 of 2801, and the difference is ~500 MB on the
  STFT variant). Anything that changes *which* chunks exist belongs in this stage.
- **`feature_extraction.features_dir` must be distinct per geometry.**
  `extract_features.py` stamps a `feature_config.json` into the directory and
  `draw_data.py` refuses to slice arrays whose stamp disagrees with `params.yaml`. This
  is the guard for the failure that produced `data/logmel_data.esp32bak`: two branches
  wrote different geometry (13 bands @ hop 512 vs 40 @ hop 320) into one directory name.
- **`export_model` runs for every variant**, so candidates are ranked on their
  post-quantization score rather than their float32 one; `export.max_f1_drop` fails the
  stage on quantization collapse. Non-`cnn` heads write `models/export/SKIPPED.json`
  instead of a model.

- **Storage layout:** bucket `doorbell-detector` on MinIO — `raw/` (audio, Label Studio
  source storage), `annotations/` (Label Studio target-storage sync, backup only),
  `dvc/` (DVC remote, configured in `.dvc/config`).
- **Credentials** live in the git-ignored `.env`: `LABEL_STUDIO_URL`,
  `LABEL_STUDIO_API_KEY`, `AWS_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`,
  `AWS_SECRET_ACCESS_KEY`, `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`,
  `MLFLOW_TRACKING_PASSWORD`. Source it before `dvc repro`/`dvc push`.
- **Experiment tracking:** `train_xgboost.py` logs params/metrics/model/confusion-matrix
  to MLflow (experiment `doorbell-detector`) at `MLFLOW_TRACKING_URI` — a self-hosted
  server (`https://mlflow.saadeh.dev`), not managed from this repo. `dvc metrics
  show`/`dvc plots diff` no longer cover training metrics; check the MLflow UI instead.
  Runs are named `<head>-<feature_type>-<git_branch>` (feature type auto-detected from
  the `*_features` column in `balanced_data.h5`) and log `balanced_data_md5` — the md5
  DVC records for the dataset — so every run traces to an exact, `dvc pull`-able
  dataset version.
- **Label Studio auth** is a JWT personal access token: `fetch_data.sh` exchanges it
  via `/api/token/refresh` for a Bearer token (legacy `Token` header returns 401).
- **Incrementality:** `data/audio` is a `persist: true` output — unchanged labels skip
  the download stage entirely; changed labels download only missing files and prune
  removed ones. Labels are only re-fetched explicitly:
  `uv run dvc repro -f fetch_labeled_data && uv run dvc repro`.
- **Labeling convention:** bell events get span labels (e.g. `front_doorbell`);
  tag-only annotations (doorslam, voice, silence, …) have an empty `label` column in
  the export and become full-file `background` rows in the converter. The end time is
  filled with the real file duration in `select_chunks.py`.
- `convert_labeled_data.py` normalizes any Label Studio audio reference (plain
  `s3://`, presigned URL, resolver path) to canonical `s3://bucket/key` so presigned
  URL churn never dirties the pipeline.
- `features.py::_load_mono_int16_scale` downmixes to mono — stereo uploads would
  silently double the frame rate and break chunking. Audio is kept at int16 amplitude
  scale (not [-1, 1]); normalization happens globally at training time.
- Requires `ffmpeg`/`ffprobe` on the host (pydub `mediainfo`).
- **Augmentation (`augment_data.py`):** grows the minority `front_doorbell` class by
  mixing real doorbell chunks (signal) with real background chunks (noise) via simple
  addition at each `augmentation.snrs_db` target SNR (`params.yaml`; `flat_doorbell` is
  excluded — too little raw data to seed it). Each signal chunk is paired with
  `pairs_per_signal_chunk` random noise chunks per SNR; the noise is circularly
  time-shifted and each mix gets ±`gain_jitter_db` uniform loudness jitter. Rows carry
  `split_group` = source file of the signal chunk (leakage guard, see training note
  below). Output rows carry `end = (chunk_size +
  1) / 1000`, 1ms past the real chunk length; this is a deliberate metadata trick so
  `select_chunks.py`'s sliding-window loop emits exactly one `chunk_start=0` window per
  augmented file, reusing the real-annotation chunking path unmodified.
- `fixed_rate_slice` in `features.py` slices chunks using a **fixed**
  `sample_rate/hop_length` frame rate, not `array.shape[1] / file_duration`. The latter
  is biased by librosa's constant `+1` frame-count offset — negligible for long real
  files (always rounded to the same width) but dominant for exactly `chunk_size`-long
  augmented clips (rounded to a different width), which broke `np.vstack` in
  `train_xgboost.py` once augmented and real chunks were trained together. `yamnet` is
  the exception: it mean-pools the covering frames instead, because a 2s augmented clip
  yields 3 embedding frames where a 2s slice of a long file yields 4. That pooling
  removes the time axis, so `yamnet` only works with the `xgboost` head —
  `train_model.py` enforces it.
- **Train/val split is group-aware** (`prepare_split` in `train_cnn.py`): `StratifiedGroupKFold`
  grouped by `split_group` (source recording; augmented rows inherit their signal
  chunk's source file). A plain random chunk split leaks near-duplicate overlapping
  windows and SNR variants across the split and inflates validation metrics.
  `training.test_size` maps to the fold fraction (1/n_splits); the realized fraction
  is logged to MLflow as `realized_test_fraction`.

## GSD Workflow

This project uses [Get Shit Done](https://github.com/amjadsaadeh/gsd) for structured
planning and execution.

**Status:** v1.0 milestone complete — all 3 phases shipped 2026-06-25. Post-milestone
hardening (MQTT trigger fixes, audio drop fixes, template auto-trim) has continued
directly on `detector.py` outside the phase structure.
**Planning docs:** `.planning/`

### Workflow commands

```
/gsd-progress          # Check status, start next milestone
/gsd-new-milestone     # Scope v2 work (see Deferred Items in .planning/STATE.md)
```

### Phases (all complete)

1. **Script Foundation** — CLI, audio device setup, template loading, error handling
2. **Detection & Notification** — Cross-correlation loop, threshold/cooldown, MQTT publish
3. **Data Collection** — `--save` flag, ring buffer clips, timestamped WAV output

v2 deferred items (not yet scoped): multiple template files, FFT frequency-domain
matching, GPIO button trigger, Prometheus metrics/health endpoint.

## Codebase Notes

- Audio constants: 16 kHz, mono, int16 — never change without updating all scripts
- Device name `seeed-2mic-voicecard` is hardcoded in data_collector.py; detector.py has a `--device-name` override
- MQTT password is passed as a CLI arg (visible in `/proc/<pid>/cmdline`) — known limitation, not a bug to fix here
- `ssl.PROTOCOL_TLS` is deprecated in Python 3.12+ — present in both scripts, carry forward as-is
- `src/deploy.sh` references the old path `src/data_collector.py` (file moved) — broken, out of scope for this work
- Oversized templates are auto-trimmed to their most energetic window (see `0b3163a`)
- Cross-correlation is slow enough to drop audio in saved clips if not handled carefully (see `c8c2788`)
- Root `requirements.txt` was removed — `pyproject.toml`/`uv.lock` is the single source
  of pipeline dependencies (`data_collection/requirements.txt` remains for the Pi)
- All 44 tests pass; run with `PYTHONPATH=./src:. uv run pytest tests/`

## Key Files

| File | Purpose |
|------|---------|
| `data_collection/data_collector.py` | Existing collector: ML + MQTT + GPIO triggers |
| `data_collection/detector.py` | Pattern-matching detector: cross-correlation, MQTT notify, `--save` clip capture |
| `data_collection/systemd/doorbell-detector.service` | systemd unit for running detector.py on the Pi |
| `data_collection/systemd/doorbell-detector.env` | Env file consumed by the systemd unit (MQTT creds, buffer/threshold config) |
| `data_collection/requirements.txt` | Pi runtime dependencies |
| `dvc.yaml` / `dvc.lock` | ML pipeline stage definitions and lock state |
| `src/fetch_data.sh` | Label Studio CSV export (JWT token exchange) |
| `src/convert_labeled_data.py` | Export → annotation-per-row CSV; URI normalization; tag-only → background |
| `src/download_audio.py` | Incremental S3 audio download (boto3) with pruning |
| `src/augment_data.py` | SNR-mixed synthetic `front_doorbell` samples (signal+noise addition) |
| `src/select_chunks.py` | Chunking + background balancing → `chunk_manifest.csv` (feature-independent) |
| `src/features.py` | Feature registry: transform + chunk slicer per `feature_extraction.type` |
| `src/extract_features.py` | Driver: extracts the configured type for manifest-referenced files only |
| `src/draw_data.py` | Slices features onto the manifest → `balanced_data.h5` |
| `src/train_model.py` | Head dispatcher (`training.head`) + feature/head compatibility check |
| `src/train_cnn.py` | Small keyword-spotting CNN, MLflow tracking |
| `src/train_xgboost.py` | XGBoost head, MLflow tracking, shared metric/quality helpers |
| `src/export_tflite.py` | int8 TFLite + C array export, gated on `export.max_f1_drop` |
| `src/paths.py` | Canonical artifact locations shared by the stage scripts |
| `params.yaml` | ML pipeline parameters (not used by detector) |
| `.env` | Git-ignored credentials for Label Studio + MinIO + MLflow |
| `.planning/REQUIREMENTS.md` | 15 v1 requirements with REQ-IDs |
| `.planning/ROADMAP.md` | 3-phase roadmap (all complete) |
| `.planning/STATE.md` | Milestone status and deferred v2 items |

## Commands

This project uses uv for dependency and venv management, so use `uv` to run python
commands.

```
set -a; source .env; set +a       # load credentials first
uv run dvc repro                  # run/refresh the pipeline (labels NOT re-fetched)
uv run dvc repro -f fetch_labeled_data && uv run dvc repro   # refresh labels too
uv run dvc push                   # push data/model versions to MinIO
PYTHONPATH=./src:. uv run pytest tests/
```

Feature/model variants (`select_chunks` and everything above it stays cached):

```
uv run dvc exp run -S feature_extraction.type=mfcc \
  -S feature_extraction.features_dir=./data/features/mfcc-40c-320hop \
  -S feature_extraction.n_mfcc=40

uv run dvc exp run -S feature_extraction.type=stft \
  -S feature_extraction.features_dir=./data/features/stft-256fft-128hop \
  -S feature_extraction.n_fft=256 -S feature_extraction.hop_length=128 \
  -S model.log_compress=true          # raw STFT magnitudes are not log-domain

uv run dvc exp run -S feature_extraction.type=yamnet \
  -S feature_extraction.features_dir=./data/features/yamnet \
  -S training.head=xgboost            # yamnet pools the time axis; cnn is rejected

uv run dvc exp show                   # compare; full metrics are in MLflow
```
