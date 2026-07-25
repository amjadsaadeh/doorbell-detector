# ESP32-S3 On-Device Doorbell Detection

**Date:** 2026-07-25
**Status:** Approved design. Phases 1–2 implemented on branch `esp32-logmel`.

## Problem

The Pi Zero W (ARMv6) cannot run any of the trained classifiers — no TensorFlow,
librosa or xgboost wheels exist for the architecture. Two escape routes were designed
earlier: a pure-numpy forward pass on the Pi
(`docs/superpowers/plans/2026-07-23-cnn-pi-inference.md`) and an MQTT hand-off to a LAN
host (`docs/superpowers/specs/2026-07-22-two-stage-audio-streaming-design.md`). This
design takes a third: replace the Pi with an ESP32-S3, which runs the classifier
itself at roughly a fifth of the power.

## The Constraint That Shapes Everything

The trained `cnn-spectrogram` model has only ~12.4k trainable parameters, but its input
is `(129, 250, 1)` — 129 STFT bins × 250 frames per 2 s chunk. Parameter count was never
the problem; **input resolution** is:

| | 129×250 (STFT branch) | 40×100 (this branch) |
|---|---|---|
| First conv activation, int8 | 1.03 MB | 0.13 MB |
| Fits | PSRAM only | internal SRAM |
| MACs per window | ~45 M | ~5.7 M |
| Est. inference @240 MHz | 0.5–2 s | ~60 ms |

So this is a **retrain on a new feature representation**, not a port of the existing
`.keras` file.

## Decisions Made

| Decision | Choice |
|----------|--------|
| Toolchain | **TFLite Micro + ESP-NN**, keeping DVC/MLflow as the source of truth (not Edge Impulse) |
| Detection architecture | **Always-on small CNN**, no cross-correlation trigger stage |
| Feature representation | 40 log-mel bins × 100 frames, 20 ms hop |
| Clip transport | Device publishes WAV over MQTT; a LAN helper uploads to MinIO |
| Normalization | **Baked into the TFLite graph**, so firmware only produces log-mel |
| Hardware | XIAO ESP32S3 Sense (onboard PDM mic, 8 MB PSRAM), INMP441 as fallback |

## Architecture

```
ESP32-S3                                    LAN host
┌──────────────────────────────┐            ┌─────────────────────────┐
│ capture_task  I2S → ring buf │            │ clip_archiver           │
│ detect_task   log-mel → TFLM │  MQTT      │  sub doorbell/candidate │
│               threshold+cool │  WAV       │  → validate WAV         │
│ net_task      MQTT pub/sub   │ ─────────► │  → PUT MinIO raw/       │
└──────────────────────────────┘            │  → pub doorbell/clip    │
         │  doorbell/detected               └─────────────────────────┘
         ▼  Home Assistant (unchanged)
```

Audio contract is unchanged project-wide: 16 kHz, mono, int16.

## Feature Representation

```yaml
feature_extraction:
  type: logmel
  features_dir: ./data/logmel_data
  n_mels: 40
  n_fft: 512        # 32 ms @ 16 kHz
  hop_length: 320   # 20 ms → a 2000 ms chunk is exactly 100 frames
  fmin: 50
  fmax: 8000
  log_offset: 1.0e-6
```

Verified numerically before implementing: a 2 s clip yields `(40, 101)` so
`draw_data.py`'s `[0:100]` slice is in range, and the 40-filter mel bank over a
512-point FFT has **no empty filters** (min 4 nonzero bins per filter) — the usual
failure mode at this bin count.

`chunk_size: 2000` is unchanged, so the augmentation SNR sweep, sliding-window
chunking, background balancing and group-split logic all carry over untouched.

**The log lives in the extractor**, not in `train_cnn.py` (`model.log_compress: false`
on this branch). This makes the `.npy` files the exact tensor the firmware's C frontend
must reproduce, reducing train/serve parity to a single comparable artifact.

Audio stays at **int16 scale** (not normalized to ±1.0) to match what I2S hands the
firmware directly.

## Pipeline Changes

- `src/extract_logmel_features.py` — new, replaces `extract_stft_features.py` in the
  `extract_features` stage. Modeled on the `cnn-mfcc` branch's extractor.
- `src/draw_data.py` — `HOP_LENGTH` and the features directory were hardcoded
  constants; both now come from `params.yaml`, and the output column is named from
  `feature_extraction.type` (`logmel_features`) so the training scripts' existing
  `*_features` auto-detection picks it up.
- `src/train_cnn.py` — `load_dataset()` and `prepare_split()` extracted from `main()`
  so `export_tflite.py` reuses the identical split and log-compression rather than
  reimplementing them. `build_model` needs no change; it was already shape-agnostic.

## Export and the Quantization Gate

`src/export_tflite.py`, DVC stage `export_model`:

1. Prepend a `Rescaling` layer seeded from `cnn_normalization.npz`, so per-bin
   normalization becomes part of the graph. The C/TFLite boundary is then exactly:
   *C produces log-mel, the interpreter does everything after.*
2. Full-integer PTQ (int8 in and out) with a representative dataset drawn from the
   training fold.
3. Emit `models/doorbell_int8.tflite`, plus `doorbell_model_data.{cc,h}` for the
   firmware build (16-byte aligned — TFLM requires it to parse the flatbuffer in place).
4. **Score the int8 interpreter on the val fold and fail the stage** if F1 drops more
   than `export.max_f1_drop` (default 0.01) below the float reference.

That gate is the point of the stage. Silent quantization collapse is the most likely
way this branch produces a model that looks fine offline and does not work on device.

**Known tension:** baking normalization in means the model's *input* tensor is raw
log-mel with a wide dynamic range (roughly −14 to +20), quantized to a single int8
scale. Normalizing outside the model instead would give the input tensor a ~N(0,1)
range and better int8 resolution, at the cost of reimplementing normalization in C.
The gate decides: if F1 drop exceeds budget, move normalization back out to the
firmware.

## Firmware (Phases 3–6)

ESP-IDF v5.x at `firmware/`, components split so each is host-testable:

| Component | Responsibility |
|---|---|
| `audio_capture` | I2S RX → 20 ms frames → PSRAM ring buffer. PDM vs I2S-std behind one interface (Kconfig) so the mic can be swapped without a rewrite. |
| `frontend` | hann + esp-dsp FFT + mel filterbank + log. Must match `extract_logmel_features.py`. |
| `inference` | TFLM interpreter, static arena, `model_data.cc` |
| `clip` | WAV header + clip assembly from the ring buffer |
| `net` | wifi, esp-mqtt, topic routing |

Three tasks: `capture_task` (core 0, high prio) → `detect_task` (core 1) → `net_task`.

**Incremental frontend.** `detect_task` computes *one* log-mel frame per 20 ms as audio
arrives and keeps 100 in a circular buffer; every 500 ms it snapshots the window and
runs TFLM. Recomputing the whole spectrogram per stride would cost 25× more FFTs for
identical output. Detection then applies the same threshold + cooldown state machine
`detector.py` uses today.

Estimated load: ~70 ms of work per 500 ms stride ≈ **14% CPU**.

Ring buffer: 3 s pre + 6 s post @ 16 kHz int16 = 288 KB, in PSRAM.

### MQTT topics

`doorbell/detected` is unchanged so Home Assistant automations don't move.

| Topic | Dir | Payload |
|---|---|---|
| `doorbell/detected` | pub | ISO-8601 timestamp |
| `doorbell/record` | **sub** | manual recording trigger, optional payload match |
| `doorbell/candidate/<device>/<epoch_ms>/<detection\|manual>` | pub | raw WAV bytes, QoS 1 |
| `doorbell/clip` | pub *(archiver)* | JSON `{s3_uri, ts, trigger, score, device_id}` |
| `doorbell/status` | pub | LWT + heartbeat (uptime, rssi, heap, inference ms) |

Metadata rides in the **topic**, not a companion message — MQTT v3 has no user
properties, and a second message would race the payload.

## Archiver Service (Phase 7)

`clip_archiver/` — a new Python package that takes over the role the classifier played
in the 2026-07-22 spec. Since the device now classifies, this is a thin archiver:
subscribe `doorbell/candidate/#` (QoS 1, persistent session), validate the WAV header
is 16 kHz/mono/int16, upload to MinIO, publish `doorbell/clip`, heartbeat on
`doorbell/archiver/status`.

S3 keys stay **flat**: `raw/esp32_<device>_<iso8601>_<trigger>.wav`. Not date-nested —
`raw/` is Label Studio's source storage, and nesting risks it not enumerating new
objects. Flat means every recorded clip enters the labeling queue automatically.

Packaged as Dockerfile + systemd unit + env file, per the portability decision already
made in the 2026-07-22 spec. Reuses the `.env` MinIO credentials.

## Risks

**Domain shift is the biggest one.** Every training sample came through the wm8960
electret array. A different microphone means a different frequency response and noise
floor; validation F1 will not transfer intact to the ESP32's PDM mic. The manual-record
+ archiver flywheel is the mitigation, not a nice-to-have — budget a "collect ~1 week of
device audio, relabel, retrain" step (Phase 8).

**Power.** Always-on inference with always-on WiFi is ~0.25–0.4 W, not the 0.1–0.3 W
quoted for a duty-cycled MCU. Still ~5× better than the Zero W. Reaching the low end
needs WiFi modem-sleep and DFS at 160 MHz.

**Quantization.** Covered by the export gate above.

## Testing

1. **Golden-vector parity** — fixture WAVs + the Python extractor's log-mel `.npy` in
   `tests/data/golden/`; a host build of the `frontend` component asserts max deviation
   < 1e-3. The single most important test in the project.
2. **Quantization guard** — the export stage gate, plus a pytest on fixture scores.
3. **Geometry regression** — `tests/logmel_extraction_test.py` pins 40 bins / 100 frames
   and asserts the resulting activation fits the SRAM budget, so a params change can't
   silently blow it.
4. **Host unit tests** — ring buffer, WAV assembly, threshold/cooldown state machine.
5. **Archiver integration** — local Mosquitto + MinIO, publish fixture clip → assert
   object exists and `doorbell/clip` published.
6. **On-device benchmark** — a bench build runs golden fixtures from flash, printing
   latency and agreement with the Python reference.
7. **End-to-end** — real doorbell through a speaker → `doorbell/detected` + clip in
   MinIO.

## Phases

| # | Phase | Needs hardware |
|---|---|---|
| 0 | Order hardware | — |
| 1 | Log-mel DVC branch + retrain | no |
| 2 | int8 export + quantization guard | no |
| 3 | Firmware skeleton: capture, WiFi, MQTT, heartbeat | yes |
| 4 | C frontend + golden-vector parity | yes |
| 5 | TFLM inference + benchmark | yes |
| 6 | Clip capture, MQTT trigger, candidate publish | yes |
| 7 | `clip_archiver` + S3 | no |
| 8 | Deploy, collect device audio, retrain | yes |

## Out of Scope

MQTT TLS (the broker has no TLS listener), battery/solar operation, fleet management,
and retiring the Pi (both run in parallel until the ESP32 proves out).

One exception: **use an OTA-capable partition table from day one.** OTA itself is
deferred, but a wireless device that must be physically unplugged to reflash is
miserable, and adding OTA later without repartitioning is only possible if the layout
allows for it now.
