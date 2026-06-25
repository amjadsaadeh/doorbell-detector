# Roadmap: Pattern-Matching Doorbell Detector

## Overview

A new `data_collection/detector.py` script delivered in three phases: first, the script
skeleton with audio device setup and template loading; second, the live detection loop with
cross-correlation, threshold/cooldown, and MQTT notification; third, the optional audio
clip saving feature for growing the training dataset.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Script Foundation** - Audio device setup, template loading, and CLI argument scaffolding
- [ ] **Phase 2: Detection & Notification** - Cross-correlation loop, threshold/cooldown, console logging, and MQTT publishing
- [ ] **Phase 3: Data Collection** - Optional WAV clip saving on detection with configurable ring buffer

## Phase Details

### Phase 1: Script Foundation
**Goal**: The script starts cleanly, opens the audio device, loads the template WAV, and exits with clear errors on missing inputs
**Depends on**: Nothing (first phase)
**Requirements**: AUD-01, AUD-02, AUD-03, DET-01
**Success Criteria** (what must be TRUE):
  1. Running the script without `--template` prints a clear usage error and exits non-zero
  2. Running with `--template missing.wav` prints a clear file-not-found error and exits non-zero
  3. Running with a valid template and no audio device prints a clear device-not-found error and exits non-zero
  4. Running with a valid template and audio device available starts without crashing and enters the capture loop
**Plans**: 1 plan
Plans:
- [ ] 01-01-PLAN.md — Create detector.py: CLI parsing, template loading, audio device setup, capture loop stub

### Phase 2: Detection & Notification
**Goal**: The script detects the doorbell reliably and notifies via console and MQTT, with configurable sensitivity and duplicate suppression
**Depends on**: Phase 1
**Requirements**: DET-02, DET-03, DET-04, NOT-01, NOT-02, NOT-03, NOT-04
**Success Criteria** (what must be TRUE):
  1. Playing the doorbell sound near the mic causes a timestamped log line to appear on stdout within one buffer window
  2. The same detection does not fire again within the cooldown window (`--cooldown-seconds`)
  3. An MQTT message is published to `--mqtt-detect-topic` (default `doorbell/detected`) on each detection
  4. The script connects to an MQTT broker with username/password when `--mqtt-username` and `--mqtt-password` are provided
  5. The script connects over TLS when `--mqtt-tls` is passed, using the provided CA/cert/key flags
**Plans**: 2 plans
Plans:
- [x] 02-01-PLAN.md — Detection core: compute_score(), --threshold/--cooldown-seconds, replace capture loop stub with detection + console log
- [ ] 02-02-PLAN.md — MQTT notification: setup_mqtt_publisher(), MQTT CLI args, publish on detection, update env file and service ExecStart

### Phase 3: Data Collection
**Goal**: When `--save` is set, every detection automatically saves a WAV clip containing pre- and post-trigger audio to disk
**Depends on**: Phase 2
**Requirements**: DATA-01, DATA-02, DATA-03, DATA-04
**Success Criteria** (what must be TRUE):
  1. Running with `--save` produces a timestamped WAV file in `--save-dir` after each detection
  2. Running without `--save` produces no WAV files
  3. The saved clip contains audio from before the detection (ring buffer) plus post-trigger audio
  4. Ring buffer length and post-trigger duration are controlled by `--buffer-minutes` and `--post-trigger-seconds`
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Script Foundation | 0/1 | Not started | - |
| 2. Detection & Notification | 1/2 | In Progress|  |
| 3. Data Collection | 0/? | Not started | - |
