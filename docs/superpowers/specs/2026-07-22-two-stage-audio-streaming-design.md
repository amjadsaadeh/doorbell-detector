# Two-Stage Doorbell Detection with Audio Hand-off over MQTT

**Date:** 2026-07-22
**Status:** Approved design, not yet implemented

## Problem

The Pi Zero W (ARMv6) cannot run the trained XGBoost/MFCC classifier — librosa and
xgboost have no ARMv6 wheels. Detection today is limited to time-domain
cross-correlation (NCC) in `data_collection/detector.py`. We want to hand audio to a
more powerful LAN host so the ML model can make the final call, while keeping the Pi
useful as a cheap first-stage trigger.

## Decisions Made

| Decision | Choice |
|----------|--------|
| Pi role | Stage-1 trigger with **pluggable trigger logic** (NCC or plain loudness/energy), not a dumb 24/7 streamer |
| Transport | **MQTT end-to-end** via existing Mosquitto broker at `192.168.178.108` — Pi publishes binary WAV clips on trigger |
| Final detection announcement | Stage-2 service publishes to `doorbell/detected` (the topic HA already listens on — automations unchanged) |
| Stage-2 placement | **Undecided — keep portable**: plain Python package with both a `Dockerfile` and a systemd unit; choose k8s vs. bare host later |
| Candidate delivery | MQTT **QoS 1 + persistent session** so Mosquitto queues clips while stage 2 is down |

## Architecture

```
Pi Zero W (stage 1)                    Powerful host (stage 2)
┌─────────────────────────┐            ┌──────────────────────────────┐
│ detector.py             │            │ detection_service (new)      │
│  capture loop (16 kHz)  │   MQTT     │  subscribe doorbell/candidate│
│  ring buffer (existing) │  binary    │  → decode WAV                │
│  Trigger (pluggable):   │  WAV clip  │  → MFCC + 2 s chunking       │
│   • NCC (current)       │ ─────────► │  → XGBoost model             │
│   • energy/loudness     │            │  → publish doorbell/detected │
└─────────────────────────┘            └──────────────┬───────────────┘
         Mosquitto @ 192.168.178.108 ◄────────────────┘
                    │
                    ▼ Home Assistant automation (unchanged)
```

Audio contract stays project-wide: 16 kHz, mono, int16.

## Pi Side (Stage 1) — Minimal Refactor of `detector.py`

- Extract the scoring step into a **trigger strategy** interface:
  `score(window) -> float`, compared against `--threshold`.
  - `NCCTrigger` — today's normalized cross-correlation, moved not rewritten.
  - `EnergyTrigger` — RMS loudness of the analysis window; needs no template.
  - Selected via `--trigger {ncc,energy}` (default `ncc` → current behavior unchanged).
- On trigger, reuse the existing ring-buffer + post-trigger clip assembly (the
  `--save` code path already builds exactly this clip) and publish the WAV bytes to a
  new topic `doorbell/candidate` (~300 KB for a ~10 s clip — far under Mosquitto's
  default message limit). Enabled by a new `--mqtt-candidate-topic` flag; candidate
  publishing is off when the flag is absent.
- Local `--save` and direct publishing to `doorbell/detected` remain available as
  flags → standalone/degraded single-stage mode still works.
- Dependencies stay ARMv6-safe: scipy, soundfile, pyaudio, paho-mqtt only.
- Deployment notes: device name on the real Pi is `wm8960-soundcard`; deployed env
  file uses the typo'd var `TEMPLATE_SCORE_THREHOLD`; broker is plain MQTT on
  1883/1884 (no TLS on 8883).

## Host Side (Stage 2) — New `detection_service/` Package

- Connects to Mosquitto, subscribes to `doorbell/candidate` (QoS 1, persistent
  session), decodes the WAV clip.
- Feature extraction **imports the existing `src/` pipeline code** (MFCC params and
  2 s chunking from `params.yaml`) rather than reimplementing — train/serve skew is
  the main correctness risk in this design.
- Runs the DVC-tracked XGBoost model over the chunks; if **k-of-n** chunks
  (configurable, default 1-of-n) classify as `front_doorbell`, publishes to
  `doorbell/detected`.
- Loads the model from a file path at startup (DVC-tracked artifact). Pulling from
  MLflow registry is a later enhancement.
- Optionally archives received clips to MinIO `raw/` so they enter the
  Label Studio → DVC labeling loop (data flywheel). Off by default via config.
- Publishes a periodic heartbeat on `doorbell/stage2/status` so HA can alert if the
  classifier is dead while a loudness trigger keeps spamming candidates.
- **Packaging (portable):** own entry point + config via env file;
  a `Dockerfile`; a `detection-service.service` systemd unit mirroring the Pi's
  env-file pattern. Placement (k8s vs. bare host) decided at deploy time.

## Failure Modes

| Failure | Behavior |
|---------|----------|
| Broker down | Pi keeps ring-buffering; paho reconnects with backoff; optional degraded mode: Pi publishes `doorbell/detected` directly from its own trigger |
| Stage 2 down | QoS 1 + persistent session → Mosquitto queues candidate clips until the service returns |
| Stage 2 silently dead | Heartbeat topic goes stale → HA alert |
| WiFi flaky | Clip publish retried by paho (QoS 1); trigger cooldown prevents pile-up |

## Testing

- **Trigger strategies:** pure-function unit tests (synthetic windows → expected
  scores) added to the existing suite (44 tests today,
  `PYTHONPATH=./src:. uv run pytest tests/`).
- **Stage 2 unit:** classify-clip path with a fixture WAV + real model file.
- **Stage 2 integration:** local Mosquitto container; publish fixture clip →
  expect `doorbell/detected`.
- **End-to-end smoke:** script that publishes a saved real clip to
  `doorbell/candidate` against the live broker.

## Implementation Sequence

1. Trigger-strategy refactor in `detector.py` (behavior-neutral, tests stay green).
2. Candidate clip publishing (`doorbell/candidate`, QoS 1).
3. `detection_service/` core: subscribe → classify → publish, with tests.
4. Packaging: Dockerfile + systemd unit + env file sample.
5. Deploy stage 2 to chosen host; run both stages with `--trigger ncc`.
6. Flip Pi to `--trigger energy` once stage 2 proves itself; NCC remains as fallback.

## Out of Scope

- Continuous 24/7 audio streaming (revisit if trigger-based hand-off proves too lossy).
- MLflow model registry integration for stage 2 model loading.
- TLS for MQTT (broker has no TLS listener today).
- Prometheus metrics / health endpoint (still a deferred v2 item; heartbeat topic
  covers the immediate need).
