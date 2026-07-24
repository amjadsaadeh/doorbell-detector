# Doorbell Detector — Project Guide

## What This Is

Audio-based doorbell detection system running on a Raspberry Pi. The project has two
main modes: an ML-based collector (`data_collection/data_collector.py`) and a
pattern-matching detector (`data_collection/detector.py`), deployed as a systemd
service. Alongside the Pi scripts lives a DVC-managed ML pipeline (`src/`, `dvc.yaml`)
that pulls labels from Label Studio and raw audio from a self-hosted MinIO S3 bucket
to train a bell classifier. The classifier itself is under active experimentation
across branches (XGBoost, small CNNs, YAMNet embeddings — see Model Experiments
below); the goal is a model small enough to eventually run inference on the Pi itself.

## ML Data Pipeline (DVC)

`uv run dvc repro` runs the full chain:
`fetch_labeled_data` (Label Studio CSV export) → `convert_labeled_data` →
`download_audio` (S3) → `fetch_noise_esc50` / `fetch_noise_demand` (external noise
pools) → `extract_data_quality` / `augmentation` (SNR-mixed synthetic
`front_doorbell` samples, using real + external noise) → `extract_features` (STFT
spectrogram, real + augmented audio) → `draw_data` (chunking + balancing) →
`train_model` (currently `train_cnn.py` on this branch + MLflow).

- **Storage layout:** bucket `doorbell-detector` on MinIO — `raw/` (audio, Label Studio
  source storage), `annotations/` (Label Studio target-storage sync, backup only),
  `dvc/` (DVC remote, configured in `.dvc/config`).
- **Credentials** live in the git-ignored `.env`: `LABEL_STUDIO_URL`,
  `LABEL_STUDIO_API_KEY`, `AWS_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`,
  `AWS_SECRET_ACCESS_KEY`, `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`,
  `MLFLOW_TRACKING_PASSWORD`. Source it before `dvc repro`/`dvc push`.
- **Experiment tracking:** `train_xgboost.py` / `train_cnn.py` log
  params/metrics/model/confusion-matrix to MLflow (experiment `doorbell-detector`) at
  `MLFLOW_TRACKING_URI` — a self-hosted server (`https://mlflow.saadeh.dev`), not
  managed from this repo. `dvc metrics show`/`dvc plots diff` no longer cover training
  metrics; check the MLflow UI instead. Runs are named
  `<xgboost|cnn>-<feature_type>-<git_branch>` (feature type auto-detected from the
  `*_features` column in `balanced_data.h5`) and log `balanced_data_md5` — the md5 DVC
  records for the dataset — so every run traces to an exact, `dvc pull`-able dataset
  version. `train_cnn.py` also logs `n_model_params` and per-epoch train/val loss
  curves.
- **Label Studio auth** is a JWT personal access token: `fetch_data.sh` exchanges it
  via `/api/token/refresh` for a Bearer token (legacy `Token` header returns 401).
- **Incrementality:** `data/audio` is a `persist: true` output — unchanged labels skip
  the download stage entirely; changed labels download only missing files and prune
  removed ones. Labels are only re-fetched explicitly:
  `uv run dvc repro -f fetch_labeled_data && uv run dvc repro`.
- **Labeling convention:** bell events get span labels (e.g. `front_doorbell`);
  tag-only annotations (doorslam, voice, silence, …) have an empty `label` column in
  the export and become full-file `background` rows in the converter. The end time is
  filled with the real file duration in `draw_data.py`.
- `convert_labeled_data.py` normalizes any Label Studio audio reference (plain
  `s3://`, presigned URL, resolver path) to canonical `s3://bucket/key` so presigned
  URL churn never dirties the pipeline.
- `extract_stft_features.py` downmixes to mono — stereo uploads would silently double
  the STFT frame rate and break chunking.
- Requires `ffmpeg`/`ffprobe` on the host (pydub `mediainfo`).
- **External noise pools (`fetch_noise_esc50.py` / `fetch_noise_demand.py` /
  `noise_pool.py`):** download and resample ESC-50 (environmental sounds, bell-like
  categories excluded via `params.yaml noise_pools.esc50.exclude_categories`) and
  DEMAND (domestic environments + hallway, `noise_pools.demand.environments`) into
  `data/noise/{esc50,demand}/` as 16kHz mono wav. Source archives are cached in
  `data/downloads/` (git-ignored, not a DVC output) so param changes only re-run the
  cheap filter/resample step. `augmentation.external_noise_pools` in `params.yaml`
  lists which pools feed both the `front_doorbell` augmentation mixing and the
  `background` class draw in `draw_data.py` (`external_background_ratio` controls the
  external-vs-real-background split).
- **Augmentation (`augment_data.py`):** grows the minority `front_doorbell` class by
  mixing real doorbell chunks (signal) with background chunks — real or from the
  external noise pools — via simple addition at each `augmentation.snrs_db` target SNR
  (`params.yaml`; `flat_doorbell` is excluded — too little raw data to seed it). Each
  signal chunk is paired with `pairs_per_signal_chunk` random noise chunks per SNR
  (pool chosen uniformly first, then a window within it); the noise is circularly
  time-shifted and each mix gets ±`gain_jitter_db` uniform loudness jitter. Rows carry
  `split_group` = source file of the signal chunk (leakage guard, see training note
  below). Output rows carry `end = (chunk_size +
  1) / 1000`, 1ms past the real chunk length; this is a deliberate metadata trick so
  `draw_data.py`'s sliding-window loop emits exactly one `chunk_start=0` window per
  augmented file, reusing the real-annotation chunking path unmodified.
- `get_stft_features` in `draw_data.py` slices chunks using a **fixed**
  `sample_rate/hop_length` frame rate, not `spectrogram.shape[1] / file_duration`. The latter
  is biased by the constant `+1` frame-count offset — negligible for long real
  files (always rounded to the same width) but dominant for exactly `chunk_size`-long
  augmented clips (rounded to a different width), which broke `np.vstack` in
  `train_xgboost.py` once augmented and real chunks were trained together.
- **Train/val split is group-aware** (`train_xgboost.py` and `train_cnn.py` share the
  logic): `StratifiedGroupKFold` grouped by `split_group` (source recording; augmented
  rows inherit their signal chunk's source file). A plain random chunk split leaks
  near-duplicate overlapping windows and SNR variants across the split and inflates
  validation metrics. `training.test_size` maps to the fold fraction (1/n_splits); the
  realized fraction is logged to MLflow as `realized_test_fraction`.

## Model Experiments (branches)

`train_xgboost.py` and `train_cnn.py` are both feature-representation-agnostic (they
auto-detect the `*_features` column), so the same scripts train against whatever
`params.yaml`/branch produces. Val F1 across branches so far:
`yamnet-features` (0.9956) > `cnn-mfcc` (0.9913) > `cnn-spectrogram` (0.9826, this
branch — a ~12.4k-param depthwise-separable CNN directly on STFT spectrogram chunks,
see `src/train_cnn.py::build_model`) > `xgboost-mfcc` (0.9519). Where the winning model
actually runs inference is still open — the Pi Zero W is ARMv6 and can't install
TensorFlow, librosa, or xgboost, so **on-Pi inference means a pure-numpy/scipy
reimplementation of whichever model wins**, not the training framework itself. See
`docs/superpowers/plans/2026-07-23-cnn-pi-inference.md` (in progress on this branch:
export folded-BN weights to `.npz`, pure-numpy forward pass, on-Pi benchmark, then wire
as a second-stage verifier behind `detector.py`'s cross-correlation trigger) and the
earlier `docs/superpowers/specs/2026-07-22-two-stage-audio-streaming-design.md`
(approved design for an MQTT hand-off to a LAN host instead, if on-Pi inference proves
too slow — not yet implemented, may be superseded by the numpy approach if that pans
out).

## GSD Workflow

This project uses [Get Shit Done](https://github.com/amjadsaadeh/gsd) for structured
planning and execution.

**Status:** v1.0 milestone complete — all 3 phases shipped 2026-06-25. Post-milestone
work has continued outside the GSD phase structure in two tracks: small hardening
fixes directly on `detector.py` (MQTT trigger fixes, audio drop fixes, template
auto-trim), and a much larger ML pipeline evolution (STFT features, external noise
pools, augmentation, CNN model experiments — see Model Experiments above) tracked via
ad-hoc `docs/superpowers/plans/` and `docs/superpowers/specs/` documents rather than
`.planning/` phases. `.planning/STATE.md` reflects only the original v1.0 milestone and
is stale with respect to this later work.
**Planning docs:** `.planning/` (v1.0 milestone), `docs/superpowers/` (post-milestone
plans/specs).

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
  of pipeline dependencies (`data_collection/requirements.txt` remains for the Pi, and
  still lists `xgboost` for the older `data_collector.py` ML path — unrelated to the
  CNN-on-Pi numpy work)
- All 61 tests pass; run with `PYTHONPATH=./src:. uv run pytest tests/`

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
| `src/noise_pool.py` | Shared helpers for external noise pool download/resample |
| `src/fetch_noise_esc50.py` | ESC-50 noise pool (environmental sounds, bell categories excluded) |
| `src/fetch_noise_demand.py` | DEMAND noise pool (domestic environments + hallway) |
| `src/augment_data.py` | SNR-mixed synthetic `front_doorbell` samples (signal+noise addition, real + external noise) |
| `src/extract_stft_features.py` | STFT magnitude spectrogram extraction, scipy defaults (mono-downmixed; real + augmented audio) |
| `src/draw_data.py` | Chunking, background balancing → `balanced_data.h5` |
| `src/train_xgboost.py` | XGBoost training with MLflow tracking |
| `src/train_cnn.py` | Depthwise-separable CNN training on 2D spectrogram/MFCC chunks, MLflow tracking |
| `models/cnn_model.keras` / `models/cnn_normalization.npz` | Trained CNN + per-bin normalization stats (DVC-tracked output of `train_model`) |
| `docs/superpowers/plans/2026-07-23-cnn-pi-inference.md` | In-progress plan: export CNN weights, pure-numpy forward pass, on-Pi benchmark, wire as second-stage verifier |
| `docs/superpowers/specs/2026-07-22-two-stage-audio-streaming-design.md` | Approved (not yet implemented) alternative design: MQTT hand-off to a LAN host for ML inference |
| `params.yaml` | ML pipeline parameters (not used by detector) |
| `.env` | Git-ignored credentials for Label Studio + MinIO + MLflow |
| `.planning/REQUIREMENTS.md` | 15 v1 requirements with REQ-IDs |
| `.planning/ROADMAP.md` | 3-phase roadmap (all complete) |
| `.planning/STATE.md` | v1.0 milestone status and deferred v2 items — stale re: post-milestone ML work |

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
