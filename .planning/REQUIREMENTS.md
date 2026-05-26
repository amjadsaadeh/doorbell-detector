# Requirements — Pattern-Matching Doorbell Detector

## v1 Requirements

### Detection

- [ ] **DET-01**: Script loads a reference doorbell sound from a WAV file specified via `--template <path>`
- [ ] **DET-02**: Script continuously listens to the microphone and cross-correlates incoming audio against the template
- [ ] **DET-03**: A detection fires when the normalized cross-correlation score exceeds `--threshold` (default 0.7)
- [ ] **DET-04**: A configurable cooldown period (`--cooldown-seconds`, default 10) suppresses repeated detections after one fires

### Notification

- [ ] **NOT-01**: On detection, a timestamped log message is printed to the console
- [ ] **NOT-02**: On detection, an MQTT message is published to a configurable topic (`--mqtt-detect-topic`, default `doorbell/detected`)
- [ ] **NOT-03**: MQTT connection supports optional username/password auth (`--mqtt-username`, `--mqtt-password`)
- [ ] **NOT-04**: MQTT connection supports optional TLS (`--mqtt-tls`, `--mqtt-tls-ca`, `--mqtt-tls-certfile`, `--mqtt-tls-keyfile`, `--mqtt-tls-insecure`)

### Data Collection

- [ ] **DATA-01**: When `--save` flag is set, save a WAV clip (ring buffer + post-trigger audio) on each detection
- [ ] **DATA-02**: Saved clips are written to `--save-dir` (default: `recordings/`) with timestamped filenames
- [ ] **DATA-03**: Ring buffer size is configurable via `--buffer-minutes` (default: 0.5)
- [ ] **DATA-04**: Post-trigger recording length is configurable via `--post-trigger-seconds` (default: 3.0)

### Audio Setup

- [ ] **AUD-01**: Script opens the seeed-2mic-voicecard input device (with `--device-name` override)
- [ ] **AUD-02**: Audio is captured at 16 kHz, mono, int16 — matching the project's standard format
- [ ] **AUD-03**: Script logs a clear error and exits gracefully if the audio device is not found or the template file is missing

## v2 Requirements (Deferred)

- Multiple template files (doorbell has multiple rings)
- Frequency-domain matching (FFT fingerprint) as alternative to cross-correlation
- GPIO button trigger
- Prometheus metrics / health endpoint

## Out of Scope

- ML/XGBoost inference — belongs to `data_collector.py`
- Recording a template live at startup — path to existing WAV is sufficient
- Changes to `data_collector.py` or any existing script

## Traceability

| REQ-ID | Phase | Status |
|--------|-------|--------|
| DET-01 | Phase 1 | pending |
| DET-02 | Phase 2 | pending |
| DET-03 | Phase 2 | pending |
| DET-04 | Phase 2 | pending |
| NOT-01 | Phase 2 | pending |
| NOT-02 | Phase 2 | pending |
| NOT-03 | Phase 2 | pending |
| NOT-04 | Phase 2 | pending |
| DATA-01 | Phase 3 | pending |
| DATA-02 | Phase 3 | pending |
| DATA-03 | Phase 3 | pending |
| DATA-04 | Phase 3 | pending |
| AUD-01 | Phase 1 | pending |
| AUD-02 | Phase 1 | pending |
| AUD-03 | Phase 1 | pending |
