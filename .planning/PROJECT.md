# Doorbell Detector — Pattern-Matching Detector

## What This Is

A new `data_collection/detector.py` script for the doorbell-detector project that listens
continuously for a doorbell using audio template matching (cross-correlation). When the
doorbell is detected it logs to the console, publishes an MQTT event, and optionally saves
the audio clip to disk to grow the training dataset. It is a sibling to the existing
`data_collection/data_collector.py`, sharing its audio setup and MQTT code.

## Core Value

Detect the doorbell reliably without requiring the trained ML model — a simpler,
always-available first line of detection based on what the doorbell actually sounds like.

## Requirements

### Validated

- ✓ Continuous audio capture with ring buffer and PyAudio — existing
- ✓ MQTT trigger publishing with optional auth and TLS — existing
- ✓ Save audio clips to WAV files — existing
- ✓ Seeed 2-mic voicecard audio device support — existing

### Active

- [ ] Load a reference doorbell WAV file via `--template <path>`
- [ ] Detect the doorbell by cross-correlating incoming audio against the template
- [ ] Log a timestamped detection event to the console
- [ ] Publish an MQTT detection message to a configurable topic
- [ ] `--save` flag: save a WAV clip automatically on each detection
- [ ] Configurable similarity threshold (`--threshold`) to tune sensitivity
- [ ] Cooldown period after detection to suppress duplicate triggers
- [ ] MQTT auth and TLS options (same flags as data_collector.py)

### Out of Scope

- ML/XGBoost inference — that belongs to data_collector.py
- GPIO trigger — not needed for this script
- Real-time template recording at startup — path to existing file is sufficient

## Context

The existing `data_collection/data_collector.py` has three trigger sources (ML, MQTT,
GPIO). It is well-structured for reuse: `save_audio()`, `setup_mqtt()`, and the audio
device selection loop can be shared or copied. The MQTT config (auth, TLS) is already
parameterized via CLI flags.

Template matching approach: normalized cross-correlation in the time domain. Load the
template WAV, resample to 16 kHz if needed, normalize. Maintain a rolling audio buffer
of the same length. Each chunk, slide the buffer and compute peak cross-correlation. If
the score exceeds `--threshold` and we are outside the cooldown window, trigger.

Known issues in the existing codebase (from codebase map) to avoid repeating:
- MQTT password as a CLI arg is visible in `/proc/<pid>/cmdline` — carry this forward as
  a known constraint rather than fix it here
- `ssl.PROTOCOL_TLS` is deprecated in Python 3.12+ — copy from existing code as-is

## Constraints

- **Runtime**: Python 3.x on Raspberry Pi; pyaudio, librosa, numpy, paho-mqtt available
- **Audio format**: 16 kHz, mono, int16 — same as the rest of the project
- **Device**: seeed-2mic-voicecard (hardcoded, same as data_collector.py)
- **Scope**: single new script; no changes to existing scripts required

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Separate script (not a flag on data_collector.py) | User preference; keeps concerns separate and the new script self-contained | — Pending |
| Cross-correlation in time domain | Simple, well-understood, no training required | — Pending |
| Reuse MQTT setup from data_collector.py | Same broker config; consistency for users | — Pending |
| `--save` as optional flag (not default) | Lets the script run as pure detector without filling disk | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-05-26 after initialization*
