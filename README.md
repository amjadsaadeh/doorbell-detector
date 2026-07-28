# Doorbell Detector

This repository is my playground for creating a sound event detection model for my doorbell.

# Use Case

During summer I regularly don't hear the doorbell while I am in the garden, because I am too far away from the door or because the door to the garden is closed to keep the hot air outside of our flat.
Especially when expecting a delivery this is quite annoying.
Additionally, I am hearing music during work from home and I just realized that my new headphones also suppress the sound of the doorbell.
So I would like to get a notification on my smartphone.

Existing solutions are quite invasive by modifying the electrics or using a new smart doorbell.
Since I don't own the flat I'm living in, I cannot do such modifications easily.
So I decided to try to detect the sound of the doorbell by microphone.

# Hardware Setup

A [Raspberry Pi Zero W](https://www.raspberrypi.com/products/raspberry-pi-zero-2-w/) with a [ReSpeaker 2 Mics Pi HAT](https://wiki.seeedstudio.com/ReSpeaker_2_Mics_Pi_HAT/) listens next to the door.
All audio is 16 kHz, mono, int16.

The trained model targets an **ESP32-S3** as its eventual inference host, which is why the pipeline ends in an int8 TFLite export — the model has to fit in the microcontroller's SRAM.

# How It Works

The project has two halves: a simple detector that already runs in production on the Pi, and an ML pipeline that trains a smarter model to eventually replace it.

## On the Pi: pattern-matching detector

`data_collection/detector.py` runs as a systemd service and continuously compares the microphone signal against a recorded template of the bell using normalized cross-correlation.
When the score passes a threshold, it publishes an MQTT message, which ends up as a push notification on my phone.
With the `--save` flag it also keeps a ring buffer and writes a timestamped WAV clip around every trigger — that is how the training data below gets collected.

## Data & labeling: Label Studio + S3

The recorded clips land in a self-hosted MinIO (S3-compatible) bucket:

- `raw/` — the audio clips, connected to [Label Studio](https://labelstud.io/) as source storage
- `annotations/` — Label Studio syncs its annotations here (backup)
- `dvc/` — remote storage for the DVC-versioned pipeline artifacts

In Label Studio I label the actual bell rings as time spans (`front_doorbell`, `flat_doorbell`, …).
Clips without a bell just get descriptive tags (doorslam, voice, silence, …) — the pipeline treats those whole files as background/negative examples.

Because real doorbell recordings are scarce, the pipeline also **synthesises** positives: it mixes real bell chunks with real noise at a range of signal-to-noise ratios, using two public noise corpora ([ESC-50](https://github.com/karolpiczak/ESC-50) and [DEMAND](https://zenodo.org/records/1227121)) alongside the labelled background.

## Training: DVC pipeline

Everything from label export to a deployable int8 model is one reproducible DVC graph.
A **feature representation and a model head are parameters, not branches** — you switch between MFCC / log-mel / STFT / YAMNet features and a CNN / XGBoost head by changing `params.yaml`, and the pipeline reruns only what actually depends on the change.

### Stages

| # | Stage | What it does | Main output |
|---|-------|--------------|-------------|
| 1 | `fetch_labeled_data` | Export labels from the Label Studio API | `data/labeled_data.csv` |
| 2 | `convert_labeled_data` | One annotation per row; normalise audio refs to `s3://` | `data/annotation_per_row_data.csv` |
| 3 | `download_audio` | Incremental S3 download (only missing files, prunes removed) | `data/audio/` |
| 4 | `extract_data_quality` | Label balance / counts before chunking | `data/data_quality/*` |
| 5 | `fetch_noise_esc50` | Download + prepare the ESC-50 noise pool | `data/noise/esc50/` |
| 6 | `fetch_noise_demand` | Download + prepare the DEMAND noise pool | `data/noise/demand/` |
| 7 | `augmentation` | Mix real bells with noise at each target SNR | `data/augmented_audio/` |
| 8 | `select_chunks` | Cut into chunks, balance bell vs. background, fix the row order | `data/chunk_manifest.csv` |
| 9 | `extract_features` | Compute features **only** for the files the manifest references | `data/features/<variant>/` |
| 10 | `draw_data` | Slice features onto the manifest rows | `data/balanced_data.h5` |
| 11 | `train_model` | Cross-validate, then refit on 100% of the data | `models/trained/`, `data/predictions/` |
| 12 | `quantize_model` | int8 TFLite conversion + tracked calibration set | `models/export/`, `data/calibration/` |
| 13 | `evaluate_quantized` | Compare float vs. int8, gate the result | `models/quantized_metrics.json` |

Two design points worth knowing before you change anything:

- **`select_chunks` sits *before* `extract_features`.** It decides which chunks make up the dataset, so feature extraction only touches the ~1200 files actually used instead of all ~2800. It also takes no feature parameters, which means every feature variant trains on exactly the same chunks — that is what makes comparing them meaningful.
- **`balanced_data.h5` is just the feature tensor.** One contiguous float32 array, row-aligned with `chunk_manifest.csv` — the metadata lives there, not duplicated in the HDF5. This also keeps the file byte-reproducible, so an unchanged pipeline genuinely skips retraining instead of rebuilding a model every run.
- **`train_model` trains six models.** Five cross-validation folds produce the honest score (a mean with a standard deviation), and then a final model is refit on *all* the data — that last one is what ships. See [Reading the metrics](#reading-the-metrics).
- **The folds' held-out predictions are kept**, per chunk, in `data/predictions/oof_predictions.csv` — every chunk scored by a model that never trained on it, joined to the Label Studio annotation and audio offset it came from. See [Inspecting the dataset](#inspecting-the-dataset).

# Parameters

All of these live in `params.yaml`. The **Affects** column is the first stage that reads the parameter; everything downstream of it reruns too.

### Chunking & balancing

| Parameter | Default | Affects | What it does |
|---|---|---|---|
| `chunk_size` | `2000` | `augmentation`, `select_chunks` | Length of one training chunk in ms. Changing it changes the model's input width and invalidates the whole pipeline. |
| `chunk_overlap` | `250` | `augmentation`, `select_chunks` | Sliding-window step in ms. Smaller = more, more-similar chunks. |
| `inbalance_ratio` | `1.0` | `select_chunks` | Background chunks drawn per positive chunk. `1.0` = balanced; `2.0` = twice as much background. |
| `external_background_ratio` | `0.5` | `select_chunks` | Share of background chunks taken from the public noise corpora instead of your own recordings. If one source runs out, the other tops it up. |

### Augmentation (synthetic positives)

| Parameter | Default | Affects | What it does |
|---|---|---|---|
| `augmentation.snrs_db` | `[-15 … 15]` | `augmentation` | Signal-to-noise ratios to mix bells at. One set of synthetic samples per value, so this multiplies dataset size. `-15 dB` means the bell sits ~32× below the noise in power. |
| `augmentation.pairs_per_signal_chunk` | `2` | `augmentation` | How many different noise clips each real bell chunk is mixed with, per SNR. |
| `augmentation.gain_jitter_db` | `6` | `augmentation` | Random ± loudness applied to each mix, so positives vary in absolute level and not just SNR. |
| `augmentation.external_noise_pools` | `[esc50, demand]` | `augmentation`, `select_chunks` | Which noise corpora to use, both as mixing noise and as extra background examples. |

### Noise pools

| Parameter | Default | Affects | What it does |
|---|---|---|---|
| `noise_pools.esc50.exclude_categories` | `[church_bells, clock_alarm]` | `fetch_noise_esc50` | ESC-50 classes to leave out. **Keep the bell-like ones excluded** — at low SNR the noise dominates the mix, so a bell-sounding "noise" labelled as doorbell would teach the model that any bell is *the* bell. |
| `noise_pools.demand.environments` | `[DKITCHEN, DLIVING, DWASHING, OHALLWAY]` | `fetch_noise_demand` | Which DEMAND recording environments to download. The defaults are domestic, matching where the device actually lives. |

### Feature extraction

`type` selects an entry in `src/features.py`, which owns both the transform and the matching chunk slicer. The remaining keys are that entry's parameters — irrelevant ones are ignored.

| Parameter | Default | Affects | What it does |
|---|---|---|---|
| `feature_extraction.type` | `logmel` | `extract_features` | One of `logmel`, `mfcc`, `stft`, `yamnet`. See [Choosing a variant](#choosing-a-variant). |
| `feature_extraction.features_dir` | `./data/features/logmel-40mel-320hop` | `extract_features` | Where the arrays are written. **Must be distinct per geometry** so variants can coexist; a stamped `feature_config.json` makes the pipeline refuse to reuse mismatched arrays. |
| `feature_extraction.n_mels` | `40` | `extract_features` | Number of mel bands (`logmel`). Also the model's input height. |
| `feature_extraction.n_mfcc` | *(not set)* | `extract_features` | Number of cepstral coefficients (`mfcc` only — add it when using that type). |
| `feature_extraction.n_fft` | `512` | `extract_features` | FFT window size in samples (512 = 32 ms at 16 kHz). |
| `feature_extraction.hop_length` | `320` | `extract_features` | Frame step in samples (320 = 20 ms), so a 2000 ms chunk is exactly 100 frames. This is the model's input width. |
| `feature_extraction.fmin` / `fmax` | `50` / `8000` | `extract_features` | Frequency range of the filterbank, in Hz. |
| `feature_extraction.log_offset` | `1e-6` | `extract_features` | Added before the log so digital silence stays finite instead of `-inf`. |

### Training

| Parameter | Default | Affects | What it does |
|---|---|---|---|
| `training.head` | `cnn` | `train_model` | `cnn` keeps the 2D (bins × frames) structure; `xgboost` flattens it. `yamnet` features only work with `xgboost`. |
| `training.test_size` | `0.2` | `train_model` | Fold size, so `0.2` → 5 folds. |
| `training.n_eval_folds` | `null` | `train_model` | How many folds to actually train. `null` = all of them (the trustworthy setting). Set to `1` for a fast iteration loop, accepting a much less reliable number. |
| `training.random_state` | `42` | `train_model` | Seed for the split and both heads. Offset per fold so each fold is independently reproducible. |

### Model hyperparameters

Nested per head, because the two share nothing.

| Parameter | Default | Affects | What it does |
|---|---|---|---|
| `model.cnn.learning_rate` | `0.001` | `train_model` | Adam learning rate. |
| `model.cnn.batch_size` | `32` | `train_model` | Training batch size. |
| `model.cnn.epochs` | `100` | `train_model` | Maximum epochs per fold (early stopping usually ends it sooner). |
| `model.cnn.early_stopping_patience` | `10` | `train_model` | Epochs without validation improvement before stopping a fold. |
| `model.cnn.dropout` | `0.3` | `train_model` | Dropout before the output layer. |
| `model.xgboost.n_estimators` | `25` | `train_model` | Number of boosting rounds. |
| `model.xgboost.max_depth` | `6` | `train_model` | Maximum tree depth. |
| `model.xgboost.learning_rate` | `0.1` | `train_model` | Shrinkage per round. |
| `model.xgboost.eval_metric` | `[logloss, error]` | `train_model` | Metrics XGBoost reports per round. |

### Quantization

| Parameter | Default | Affects | What it does |
|---|---|---|---|
| `quantization.calibration_samples` | `500` | `quantize_model` | How many chunks the int8 converter sees when fitting activation ranges. The exact selection is saved to `data/calibration/` and versioned, because it is part of the resulting model. |
| `quantization.max_f1_drop` | `0.01` | `evaluate_quantized` | How much F1 int8 may cost versus float before the pipeline stops. Deliberately loose — it is a collapse detector, not a drift detector. |
| `quantization.on_failure` | `fail` | `evaluate_quantized` | `fail` stops the pipeline so a bad model cannot be pushed and flashed. `warn` logs and continues — for deliberate exploration only. |

# Running It

Dependencies are managed with [uv](https://docs.astral.sh/uv/). `ffmpeg` must be installed on the machine running the pipeline.

Credentials live in a git-ignored `.env`:

```
LABEL_STUDIO_URL=...        # Label Studio instance
LABEL_STUDIO_API_KEY=...    # personal access token
AWS_ENDPOINT_URL=...        # MinIO endpoint
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
MLFLOW_TRACKING_URI=...          # MLflow tracking server
MLFLOW_TRACKING_USERNAME=...
MLFLOW_TRACKING_PASSWORD=...
```

**Load them first — every command below assumes this:**

```bash
set -a; source .env; set +a
```

## Running the pipeline

```bash
uv run dvc repro                    # run everything that is out of date
uv run dvc repro --dry              # show what *would* run, change nothing
uv run dvc dag                      # print the stage graph
uv run dvc status                   # what is stale, and why
```

Labels are deliberately *not* re-fetched on a normal run, so an unchanged label set never re-downloads audio. To pull fresh labels:

```bash
uv run dvc repro -f fetch_labeled_data && uv run dvc repro
```

## Running a single stage

```bash
uv run dvc repro <stage>            # that stage, plus any upstream that is stale
uv run dvc repro -s <stage>         # ONLY that stage, ignoring upstream
uv run dvc repro -f -s <stage>      # ONLY that stage, even if up to date
```

⚠️ **`-f` without `-s` forces the entire upstream chain**, including `fetch_labeled_data` — which fails if Label Studio is unreachable. To re-run just the training, you want `-f -s`:

```bash
uv run dvc repro -f -s train_model quantize_model evaluate_quantized
```

Every stage by name:

```bash
uv run dvc repro -s fetch_labeled_data      # 1  needs Label Studio reachable
uv run dvc repro -s convert_labeled_data    # 2
uv run dvc repro -s download_audio          # 3  needs S3 credentials
uv run dvc repro -s extract_data_quality    # 4
uv run dvc repro -s fetch_noise_esc50       # 5  large download on first run
uv run dvc repro -s fetch_noise_demand      # 6  large download on first run
uv run dvc repro -s augmentation            # 7
uv run dvc repro -s select_chunks           # 8
uv run dvc repro -s extract_features        # 9
uv run dvc repro -s draw_data               # 10
uv run dvc repro -s train_model             # 11 slowest stage: 6 models
uv run dvc repro -s quantize_model          # 12
uv run dvc repro -s evaluate_quantized      # 13
```

## Trying a different variant

Use `dvc exp run --temp`, which runs in a throwaway worktree. A plain `dvc exp run` rewrites `params.yaml` in place, stages that rewrite into git, and leaves you on a detached HEAD.

`-S key=value` overrides an existing parameter; `-S +key=value` **adds** one that is not in `params.yaml` yet.

```bash
# log-mel — the committed default, no overrides needed
uv run dvc repro

# MFCC
uv run dvc exp run --temp -S feature_extraction.type=mfcc \
  -S feature_extraction.features_dir=./data/features/mfcc-40c-320hop \
  -S +feature_extraction.n_mfcc=40

# STFT spectrogram (log compression is automatic for this type)
uv run dvc exp run --temp -S feature_extraction.type=stft \
  -S feature_extraction.features_dir=./data/features/stft-256fft-128hop \
  -S feature_extraction.n_fft=256 -S feature_extraction.hop_length=128

# YAMNet embeddings — requires the xgboost head
uv run dvc exp run --temp -S feature_extraction.type=yamnet \
  -S feature_extraction.features_dir=./data/features/yamnet \
  -S training.head=xgboost

uv run dvc exp show     # local comparison; the real record is in MLflow
```

To make a variant permanent, edit `params.yaml` by hand, run `uv run dvc repro`, and commit.

## Sharing results

```bash
uv run dvc push          # upload data + models to the S3 remote
uv run dvc pull          # fetch them on another machine
uv run dvc checkout      # restore workspace files from the local cache
git push
```

## Tests

```bash
PYTHONPATH=./src:. uv run pytest tests/
```

# Choosing a variant

Measured on the same 1770 chunks, 5-fold cross-validated:

| Variant | Head | val F1 | Worst fold | Peak activation |
|---|---|---|---|---|
| **log-mel 40×100** | cnn | **0.9931 ± 0.0095** | 0.9742 | 125 KB |
| MFCC 40×100 | cnn | 0.9901 ± 0.0109 | 0.9702 | 125 KB |
| YAMNet | xgboost | 0.9648 ± 0.0245 | 0.9350 | — |
| STFT 129×250 | cnn | 0.8893 ± 0.1941 | **0.5017** | 1008 KB |

Log-mel is the default because it wins on accuracy *and* fits the microcontroller.
STFT ties the leaders on four folds out of five and then collapses to chance on the fifth — it is unstable, not merely worse, and it needs 8× the SRAM.
YAMNet is a useful reference point but is a MobileNet producing 1024-dim embeddings, so it is not a microcontroller option at all.

# Reading the metrics

Runs are tracked in MLflow (experiment `doorbell-detector`), named `<head>-<feature_type>-<git_branch>`. Three numbers get confused easily:

- **`val_f1_score`** (± `val_f1_score_std`) — the cross-validated mean. **This is the only generalisation estimate.** Quote this one.
- **`insample_full_*`** — the final model scored on its own training data. A did-it-fit check, never a performance claim.
- **`float_insample_*` / `int8_insample_*`** on the `-quantized` run — a float-vs-int8 *delta*, tagged `metric_scope`. It says what quantization changed, not how good the model is.

The shipped model is refit on 100% of the data, so it has no held-out data of its own. If you ever need a held-out number for those exact weights, carve out a fixed holdout before cross-validation — it cannot come from the quantization stage.

When the quantization gate fails, the MLflow run is marked **FAILED** and carries `diagnostics/disagreements.csv` (every chunk where float and int8 predict differently, joined to its source recording) plus `diagnostics/worst_score_deviations.csv`. That is where you look to find out *why*.

# Inspecting the dataset

Metrics say *how often* the model is wrong. To find out *which* chunks it gets wrong — and whether they are hard samples or bad labels — there is an interactive viewer:

```bash
./src/inspect_dataset.sh                 # held-out predictions, audio, spectrograms
./src/inspect_dataset.sh --embeddings    # + a similarity map of the CNN's embeddings
./src/inspect_dataset.sh --all-chunks    # include chunks no fold held out
```

It opens [Renumics Spotlight](https://github.com/Renumics/spotlight) in a browser: one row per chunk, a filterable table, and an audio player plus spectrogram for whichever row you select. The table opens sorted with the errors first, most-confident-mistake at the top — a chunk the model was *sure* about and still got wrong is usually either genuinely hard or mislabeled.

Every row carries where the chunk came from, so a suspicious one can be fixed at the source:

| Column | What it is |
|---|---|
| `annotation_id` | the Label Studio annotation this chunk descends from (`aug_*` = synthetic, `noise_*` = external pool) |
| `audio_file_name`, `chunk_start`, `chunk_end` | the recording and the exact window inside it, in milliseconds |
| `start`, `end` | the annotated span in that recording, in seconds |
| `label`, `true_class`, `predicted_class`, `outcome` | the label, and which confusion-matrix cell the chunk landed in |
| `y_score`, `margin` | the raw probability, and its distance from the 0.5 threshold |
| `fold`, `split_group` | which fold held this chunk out, and the recording that grouped it |
| `snr_db`, `noise_pool` | for augmented and external-background rows |

**The predictions are out-of-fold.** They come from `data/predictions/oof_predictions.csv`, written by `train_model` from inside the cross-validation loop, so every score belongs to a model that never saw that chunk. That matters: the shipped model is refit on 100% of the data, so its own mistakes are memorization failures rather than generalization ones. Nothing after `train_model` can reproduce this file — if it is missing, run `uv run dvc repro train_model` or `uv run dvc pull`.

Two practical notes:

- **Spotlight requires `librosa>=0.11` and `pyarrow>=21`**, which is why this project runs them. Changing a feature-extraction dependency under a cached dataset is a real corruption risk, so it was checked rather than assumed: log-mel, MFCC and STFT features come out **bit-identical** between librosa 0.10.2.post1 and 0.11.0 at these parameters. Nothing in `data/features/` was invalidated by the upgrade. Re-run that comparison before the next librosa bump.
- **`data/spotlight/` is a disposable cache** (one wav per chunk, ~110 MB, plus the table). It is git-ignored, not DVC-tracked, and rebuilt on demand — delete it freely. Re-runs reuse the wavs unless you pass `--refresh`.

With `training.n_eval_folds` capped to 1, only that fold's chunks have a held-out prediction; `oof_coverage` on the MLflow run says what fraction of the dataset that was, and `--all-chunks` shows the rest with their prediction columns marked `n/a`.

# Repository Layout

| Path | What it is |
|------|------------|
| `src/` | DVC pipeline stages and shared helpers |
| `dvc.yaml`, `params.yaml` | pipeline definition and tunable parameters |
| `data_collection/` | everything that runs on the Pi (detector, collector, systemd units) |
| `tests/` | unit tests for the pipeline scripts |
| `.planning/` | planning docs (requirements, roadmap, state) |
| `CLAUDE.md` | working notes and gotchas, for whoever touches the pipeline next |

# Status & Ideas

v1.0 of the on-Pi detector is deployed and working (cross-correlation + MQTT + clip capture).
The ML pipeline trains a log-mel CNN and exports a 26 KB int8 TFLite model with a 125 KB activation footprint, sized for an ESP32-S3.

Ideas for later: multiple templates, frequency-domain matching, GPIO button trigger, Prometheus metrics, firmware for the ESP32-S3, and a fixed holdout so the shipped weights get a held-out score of their own.
