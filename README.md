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

## Training: DVC pipeline

The pipeline (`dvc.yaml`) is fully reproducible with DVC and runs end-to-end with one command:

1. fetch the label export from the Label Studio API
2. convert it to one-annotation-per-row (normalizing all audio references to `s3://` URIs)
3. download only missing audio files from S3 (incremental, pruned)
4. fetch external background-noise pools (ESC-50, DEMAND)
5. augment the minority `front_doorbell` class by SNR-mixing real bell chunks with
   background noise (real recordings + the external pools)
6. extract STFT spectrogram features
7. cut into chunks and balance bell vs. background
8. train a classifier, tracked with MLflow

The classifier itself is under active experimentation across branches — XGBoost on
flattened features, small CNNs directly on the 2D spectrogram/MFCC chunks, and YAMNet
embeddings. Val F1 so far: YAMNet embeddings (0.9956) > CNN/MFCC (0.9913) >
CNN/spectrogram (0.9826, this branch) > XGBoost/MFCC (0.9519). Since the Pi Zero W is
ARMv6 and can't install TensorFlow, librosa, or xgboost, getting the winning model onto
the Pi means a pure-numpy/scipy reimplementation of its forward pass, not the training
framework itself — see `docs/superpowers/plans/` for the in-progress work on that.

# Running It

Dependencies are managed with [uv](https://docs.astral.sh/uv/). Credentials live in a git-ignored `.env`:

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

```bash
set -a; source .env; set +a

uv run dvc repro          # run the pipeline (skips everything unchanged)
uv run dvc repro -f fetch_labeled_data && uv run dvc repro   # also refresh labels
uv run dvc push           # push data/model versions to the S3 remote

PYTHONPATH=./src:. uv run pytest tests/   # run the tests
```

`ffmpeg` needs to be installed on the machine running the pipeline.

# Repository Layout

| Path | What it is |
|------|------------|
| `src/` | DVC pipeline stages (fetch, convert, download, noise pools, augmentation, features, training) |
| `dvc.yaml`, `params.yaml` | pipeline definition and tunable parameters |
| `models/` | DVC-tracked trained model artifacts |
| `data_collection/` | everything that runs on the Pi (detector, collector, systemd units) |
| `tests/` | unit tests for the pipeline scripts |
| `.planning/` | v1.0 milestone planning docs (requirements, roadmap, state) |
| `docs/superpowers/` | post-milestone design specs and implementation plans (ML pipeline evolution, CNN-on-Pi inference) |

# Status & Ideas

v1.0 of the on-Pi detector is deployed and working (cross-correlation + MQTT + clip capture).
Since then, work has focused on the ML side: SNR-mixed augmentation, external noise
pools, and small-CNN model experiments aimed at running the trained model directly on
the Pi (see `docs/superpowers/plans/2026-07-23-cnn-pi-inference.md`). Other ideas for
later: multiple templates, frequency-domain matching, GPIO button trigger, and
Prometheus metrics.
