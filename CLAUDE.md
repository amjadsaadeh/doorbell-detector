# Doorbell Detector — Project Guide

## What This Is

Audio-based doorbell detection system running on a Raspberry Pi. The project has two
main modes: an ML-based collector (`data_collection/data_collector.py`) and a
pattern-matching detector (`data_collection/detector.py`), deployed as a systemd
service. Alongside the Pi scripts lives a DVC-managed ML pipeline (`src/`, `dvc.yaml`)
that pulls labels from Label Studio and raw audio from a self-hosted MinIO S3 bucket,
trains a small log-mel CNN, and exports an int8 TFLite model sized for an ESP32-S3.
The XGBoost head is still selectable (`training.head`) but is now the baseline, not
the default. README.md is the human-facing counterpart to this file: it carries the
full parameter reference and the per-stage command list.

## ML Data Pipeline (DVC)

`uv run dvc repro` runs the full chain:
`fetch_labeled_data` (Label Studio CSV export) → `convert_labeled_data` →
`download_audio` (S3) → `extract_data_quality` / `augmentation` (SNR-mixed synthetic
`front_doorbell` samples) → `select_chunks` (chunking + balancing) →
`extract_features` → `draw_data` (slicing) → `train_model` (MLflow) →
`quantize_model` (int8 TFLite) → `evaluate_quantized` (gate + MLflow).

- **A feature/model variant is a parameter, not a branch.** `feature_extraction.type`
  picks an entry in `src/features.py` (`mfcc`, `logmel`, `stft`, `yamnet`) which owns
  both the transform and the matching chunk slicer; `training.head` picks `cnn` or
  `xgboost`. Sweep with `dvc exp run -S`, not by forking dvc.yaml. Historic branches
  (`cnn-mfcc`, `cnn-spectrogram`, `yamnet-features`, `stft-spectrogram-features`,
  `cnn-logmel`, `esp32-logmel`) each hard-coded one combination; they are superseded.
- **`balanced_data.h5` holds one contiguous float32 array, nothing else.** Chunks used
  to be a pandas object column of per-row arrays, which PyTables pickled into bytes that
  differed run to run over identical data — so `draw_data` always looked changed to DVC
  and `train_model` could never be skipped (~20 min per `dvc repro`, permanently). It is
  now written with h5py and is byte-reproducible; `tests/dataset_test.py` asserts that.
  The chunk metadata is deliberately **not** duplicated into it: `chunk_manifest.csv`
  carries it row-for-row in the same order, and `src/dataset.py` enforces that the two
  agree on length.
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
- **Hyperparameters are per head** (`model.cnn`, `model.xgboost`). A flat block
  meant XGBoost silently received the CNN's `learning_rate: 0.001` plus
  `dropout`/`epochs` as unknown kwargs and barely trained. Whether a representation
  needs log-compression before normalization is likewise a `FeatureSpec` property
  (`needs_log_compression`, true only for `stft`), not a params flag to remember.
- **Quantization is two stages, deliberately.** `quantize_model` builds the int8
  graph; `evaluate_quantized` scores it and decides whether it ships, so the thing
  that gates the model is not the same code path that produced it. Both run for every
  variant, so candidates are ranked post-quantization rather than on float32; non-`cnn`
  heads write `SKIPPED.json` markers instead.
- **The calibration subset is a tracked artifact** (`data/calibration/`, a DVC output).
  Post-training quantization fits activation ranges to whatever chunks it is shown, so
  that choice is part of the model: same weights + same calibration chunks reproduce
  the same graph. It is drawn from the *training* fold only — calibrating on validation
  chunks would tune the ranges to the data the next stage scores against — and the
  saved `row_index` values are dataset-global, so they join straight back to
  `chunk_manifest.csv` (`calibration_manifest.csv` is the human-readable view).
- **`evaluate_quantized` logs its own MLflow run**, named after the training run with a
  `-quantized` suffix (e.g. `cnn-logmel-unified-pipeline-quantized`, tagged
  `stage=quantized`). It carries the calibration `.npz` + `.csv` as artifacts and its
  md5 as a param, so a quantized model in MLflow can be matched to a DVC-tracked
  calibration set. `quantization.max_f1_drop` fails the stage on collapse.
- **The quantized run measures a delta, not quality**, and is tagged
  `metric_scope: in-sample float-vs-int8 delta` to say so. The verdict is recorded as
  a `gate` tag and a `gate_passed` metric, and a breach ends the run as **FAILED** (the
  raise happens inside the mlflow run context) — so the evidence outlives the dead
  pipeline instead of only existing as a non-zero exit code. `quantization.on_failure`
  switches between `fail` (default; stops the pipeline so a bad model cannot be pushed
  and flashed) and `warn` (advisory, for deliberate exploration). Note the gate protects
  the pipeline, not the filesystem: on failure the `.tflite` is still on disk.
- **Every quantized run logs `diagnostics/`** — `disagreements.csv` (chunks where float
  and int8 predict differently, joined to recording/offset/label) and
  `worst_score_deviations.csv` (top 20 by |float-int8|). Aggregates say a regression
  happened; these say where. Both are written even on a passing run, so their absence
  always means the stage did not get that far. Comparing two versions of one
  model needs no held-out data, so it scores **all** chunks — which is the point.
  Measured on a single 183-chunk fold, `max_score_deviation` read 0.0078; across all
  1770 it is **0.0868**, an ~11x larger perturbation that the small fold simply could
  not see. `f1_drop` saturates and can even go slightly negative (int8 flipping one
  borderline sample the right way); `max_score_deviation` is the metric that moves
  first, and 0.0868 against a 0.5 decision threshold is the real margin to watch.

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
  Runs are named `<head>-<feature_type>-<git_branch>` (feature type read from the
  `feature_type` attribute of `balanced_data.h5`) and log `balanced_data_md5` — the md5
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
- **Train/val split is group-aware and cross-validated** (`src/splits.py`):
  `StratifiedGroupKFold` grouped by `split_group` (source recording; augmented rows
  inherit their signal chunk's source file). A plain random chunk split leaks
  near-duplicate overlapping windows and SNR variants across the split and inflates
  validation metrics. `training.test_size` maps to the fold fraction (1/n_splits).
- **Both heads train every fold** and report the mean. Scoring only the first fold is
  what made every feature variant look like val F1 1.0000: the folds are wildly uneven
  (183 / 288 / 503 / 338 / 458 chunks on the current dataset) because group sizes vary,
  and fold 0 is the smallest and easiest. The same log-mel model that scores 1.0000 on
  fold 0 scores **0.9931 ± 0.0095, worst fold 0.9742** across all five. MLflow gets the
  mean under the plain name (`val_f1_score`) plus `_std` / `_min` companions and
  per-fold `fold{i}_*` metrics. `training.n_eval_folds: null` means all folds; set it
  to 1 for a fast iteration loop, at the old credibility.
- **The shipped model is a final refit on 100% of the chunks**, trained after the CV
  loop. CV establishes what a model built this way scores; the refit is the deliverable
  and gets the ~20% of chunks every CV model held out. It has no validation split, so
  `EarlyStopping` cannot run — it trains for the mean epoch at which the folds'
  `val_loss` bottomed out (`final_fit_epochs`, logged), the only unbiased epoch estimate
  available. Its in-sample metrics are logged as `insample_full_*`: a did-it-fit check,
  never a performance claim.
- **Consequence: nothing after `train_model` can measure accuracy honestly.** The
  generalization estimate is `val_f1_score` (± `_std`) on the training run, full stop.
  If you ever need a held-out number for the shipped model, carve out a fixed holdout
  before CV — do not read one out of the quantization stage.
- **The folds' held-out predictions survive the run** as `data/predictions/oof_predictions.csv`
  (`src/oof.py`, a DVC out of `train_model`, also an MLflow artifact under
  `predictions/`). It is the *only* honest per-chunk record the pipeline produces:
  every score comes from the fold model that did not train on that chunk, whereas
  the shipped refit has seen everything, so its errors are memorization failures.
  It cannot be recomputed after the fact — the fold models are gone. Columns are
  `outcome` (TP/TN/FP/FN), `y_score`, `margin` (distance from the 0.5 threshold),
  `fold`, plus the provenance from `src/provenance.py`. Coverage is complete only
  when `training.n_eval_folds` is null; `oof_coverage` in MLflow records what it was.
- **`src/provenance.py` is the single definition of "where a chunk came from"** —
  annotation id, file, ms offsets, label, split_group, snr_db, noise_pool, keyed by
  the dataset-global `row_index`. Both the OOF table and `evaluate_quantized`'s
  `diagnostics/` use it, so a chunk is described identically wherever it turns up.
  Anything new that reports per-chunk numbers should join through it rather than
  re-picking manifest columns.
- **`src/inspect_dataset.py` + `src/inspect_dataset.sh` browse the dataset by
  outcome** in Renumics Spotlight: filterable table, audio player, spectrogram,
  optional CNN-embedding similarity map (`--embeddings`). Deliberately **not** a DVC
  stage — it reads pipeline outputs and builds `data/spotlight/`, a git-ignored,
  untracked cache (one wav per chunk, ~110 MB) that can be deleted at any time.
  `prepare` cuts the wavs, `show` serves them; the wrapper runs both.
- **Spotlight is what dragged `librosa` to 0.11 and `pyarrow` to 24** (plus
  `dill` down to <0.3.9, which HuggingFace `datasets` requires). Bumping a
  feature-extraction dependency under a cached dataset is a silent-corruption risk,
  so it was **measured, not assumed**: log-mel, MFCC and STFT features are
  **bit-identical** between librosa 0.10.2.post1 and 0.11.0 at this project's
  parameters (verified on a real recording, md5 of the float32 array). Nothing in
  `data/features/` was invalidated. Re-run that comparison before the next librosa
  bump rather than trusting it to keep holding.
- **Spotlight gotcha:** it builds a Category's value list by sorting the column's
  uniques, so a single `None` beside the strings raises
  `TypeError: '<' not supported` from inside the server process, far from the cause.
  `name_the_gaps()` fills those with `n/a` before the parquet is written — `noise_pool`
  is empty for non-pool chunks, and under `--all-chunks` every prediction column is
  empty for chunks no fold held out.

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
- All 146 tests pass; run with `PYTHONPATH=./src:. uv run pytest tests/`

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
| `src/dataset.py` | Reads/writes `balanced_data.h5`; the byte-reproducible container |
| `src/train_model.py` | Head dispatcher (`training.head`) + feature/head compatibility check |
| `src/train_cnn.py` | Small keyword-spotting CNN, MLflow tracking |
| `src/train_xgboost.py` | XGBoost head, MLflow tracking, shared metric/quality helpers |
| `src/splits.py` | Group-aware CV folds + fold-metric aggregation, shared by both heads |
| `src/oof.py` | Held-out per-chunk predictions collected across the CV folds |
| `src/provenance.py` | Where a chunk came from: the manifest join every per-chunk diagnostic uses |
| `src/inspect_dataset.py` | Spotlight inspector: prepare (project env) + show (isolated env) |
| `src/inspect_dataset.sh` | Runs both halves in order — the entry point |
| `src/tflite_utils.py` | Pure TFLite helpers (wrap, convert, interpret, C array) |
| `src/quantize_model.py` | int8 conversion + the DVC-tracked calibration set |
| `src/evaluate_quantized.py` | float-vs-int8 delta, the gate, and its diagnostics |
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
./src/inspect_dataset.sh          # browse chunks by prediction outcome (Spotlight)
```

Feature/model variants (`select_chunks` and everything above it stays cached):

Always use `--temp`: a plain `dvc exp run` executes in the workspace, where it
rewrites `params.yaml`, stages that rewrite into the git index, and leaves HEAD
detached. `--temp` runs in a throwaway worktree and touches none of it.

`-S key=value` overrides an existing param; `-S +key=value` **adds** one that
isn't in params.yaml yet (e.g. `n_mfcc`, which only the mfcc type reads).

```
# log-mel, the committed default — no overrides needed
uv run dvc repro

uv run dvc exp run --temp -S feature_extraction.type=mfcc \
  -S feature_extraction.features_dir=./data/features/mfcc-40c-320hop \
  -S +feature_extraction.n_mfcc=40    # '+' — n_mfcc is not in params.yaml

uv run dvc exp run --temp -S feature_extraction.type=stft \
  -S feature_extraction.features_dir=./data/features/stft-256fft-128hop \
  -S feature_extraction.n_fft=256 -S feature_extraction.hop_length=128
  # log-compression is automatic: features.py declares it for stft

uv run dvc exp run --temp -S feature_extraction.type=yamnet \
  -S feature_extraction.features_dir=./data/features/yamnet \
  -S training.head=xgboost            # yamnet pools the time axis; cnn is rejected

uv run dvc exp show                   # compare; full metrics are in MLflow
```
