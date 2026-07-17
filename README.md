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
4. extract MFCC features
5. cut into chunks and balance bell vs. background
6. train an XGBoost classifier, tracked with dvclive

# Running It

Dependencies are managed with [uv](https://docs.astral.sh/uv/). Credentials live in a git-ignored `.env`:

```
LABEL_STUDIO_URL=...        # Label Studio instance
LABEL_STUDIO_API_KEY=...    # personal access token
AWS_ENDPOINT_URL=...        # MinIO endpoint
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
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
| `src/` | DVC pipeline stages (fetch, convert, download, features, training) |
| `dvc.yaml`, `params.yaml` | pipeline definition and tunable parameters |
| `data_collection/` | everything that runs on the Pi (detector, collector, systemd units) |
| `tests/` | unit tests for the pipeline scripts |
| `.planning/` | planning docs (requirements, roadmap, state) |

# Status & Ideas

v1.0 of the on-Pi detector is deployed and working (cross-correlation + MQTT + clip capture).
Ideas for later: multiple templates, frequency-domain matching, GPIO button trigger, Prometheus metrics, and deploying the trained XGBoost model to the Pi.
