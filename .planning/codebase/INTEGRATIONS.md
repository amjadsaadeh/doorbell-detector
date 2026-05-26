# External Integrations

**Analysis Date:** 2026-05-26

## APIs & External Services

**Label Studio (data annotation platform):**
- Used for: exporting labeled annotation CSV and downloading annotated audio files
- SDK: `label-studio-sdk==1.0.7` (`make_background.py`)
- REST API also called directly via `curl` (`src/fetch_data.sh`) and `requests` (`src/download_audio.py`)
- Auth: Token-based (`Authorization: Token <API_KEY>`)
- Env vars: `LABEL_STUDIO_URL`, `API_KEY`
- Operations:
  - `GET /api/projects/1/export?exportType=CSV` — exports annotation CSV
  - `GET <remote_audio_path>` — downloads individual audio WAV files
  - `client.tasks.list(project=1, ...)` — queries unannotated tasks (SDK)
  - `client.annotations.create(...)` — bulk-annotates background tasks (SDK)

## Protocols Used

**MQTT (IoT messaging):**
- Library: `paho-mqtt==2.1.0`
- Role: optional external trigger source in the live data collector
- Direction: subscriber only (`data_collection/data_collector.py`)
- Default topic: `doorbell/trigger`
- Default port: 1883 (plain) or 8883 (TLS)
- Behavior: any incoming message on the subscribed topic (or a specific payload if `--mqtt-trigger-value` is set) fires a recording save
- Connection: persistent with `loop_start()` (background thread); keepalive 60 s

**HTTP/HTTPS:**
- Used by `src/download_audio.py` and `src/fetch_data.sh` to pull data from Label Studio
- Library: `requests==2.32.3` (Python scripts), `curl` (shell script)

## Authentication Mechanisms

**Label Studio:**
- API token passed as HTTP header: `Authorization: Token ${API_KEY}`
- Token sourced from `API_KEY` environment variable

**MQTT broker (optional, all disabled by default):**

| Method | CLI flags | Env var group |
|--------|-----------|---------------|
| Username + password | `--mqtt-username`, `--mqtt-password` | `MQTT_AUTH_ARGS` |
| TLS with CA bundle | `--mqtt-tls`, `--mqtt-tls-ca` | `MQTT_TLS_ARGS` |
| Mutual TLS (client cert) | `--mqtt-tls-certfile`, `--mqtt-tls-keyfile` | `MQTT_TLS_ARGS` |
| Insecure TLS (no cert verify) | `--mqtt-tls-insecure` | `MQTT_TLS_ARGS` |

## Data Storage

**Local filesystem:**
- `data/audio/` — downloaded WAV recordings (DVC-tracked)
- `data/mfcc_data/` — pre-computed MFCC `.npy` arrays (DVC-tracked)
- `data/balanced_data.h5` — balanced training dataset in HDF5 format (DVC-tracked)
- `models/xgboost_model.json` — trained XGBoost model (DVC-tracked)
- `recordings/` — WAV files saved by the live collector on the Raspberry Pi (runtime output, not tracked)

**DVC remote:**
- Type: local path (`../dvc_remote`)
- Config: `.dvc/config`, remote named `private`
- Used for: versioning large data files and model artifacts outside git

**No cloud storage, no database, no object store** currently configured.

## Hardware Interfaces

**Microphone — ReSpeaker 2 Mics Pi HAT:**
- Interface: ALSA / PortAudio (`PyAudio`)
- Device name (hard-coded): `seeed-2mic-voicecard`
- Parameters: 16 kHz sample rate, 1 channel, 16-bit PCM (`paInt16`)
- Discovery: `data_collector.py` iterates `pyaudio.PyAudio().get_device_count()` and matches on `"seeed-2mic-voicecard"` in the device name; exits with error if not found

**GPIO button (Raspberry Pi, optional):**
- Library: `RPi.GPIO` (Pi-only, optional import)
- Mode: BCM pin numbering
- Trigger: falling-edge interrupt with 300 ms debounce
- Pull: internal pull-up enabled
- Configured via `--gpio-pin <BCM_PIN>` (injected from `GPIO_ARGS` in env file)

## Deployment & Process Management

**systemd service:**
- Unit file: `data_collection/systemd/doorbell-collector.service`
- Runs as user `pi` on the Raspberry Pi
- Configuration injected via `EnvironmentFile=/etc/doorbell-collector.env`
- Restart policy: `Restart=always`, `RestartSec=5`
- Dependencies: `network-online.target`, `sound.target`
- Logs: journald (`journalctl -u doorbell-collector -f`)

**SSH deployment:**
- Script: `src/deploy.sh`
- Copies collector script and params to Raspberry Pi over SSH
- Target host sourced from `MIC_SSH_URI` environment variable

## Environment Variable Surface

| Variable | Used by | Purpose |
|----------|---------|---------|
| `LABEL_STUDIO_URL` | `src/fetch_data.sh`, `src/download_audio.py`, `make_background.py` | Label Studio base URL |
| `API_KEY` | `src/fetch_data.sh`, `src/download_audio.py`, `make_background.py` | Label Studio API token |
| `MIC_SSH_URI` | `src/deploy.sh` | SSH target for deploying to Raspberry Pi |
| `SAVE_DIR` | systemd env file → `data_collector.py` | Directory for triggered recordings |
| `BUFFER_MINUTES` | systemd env file → `data_collector.py` | Ring buffer duration |
| `POST_TRIGGER_SECONDS` | systemd env file → `data_collector.py` | Post-trigger recording length |
| `CHUNK_SIZE_MS` | systemd env file → `data_collector.py` | Audio window size |
| `N_MFCC` | systemd env file → `data_collector.py` | MFCC coefficient count |
| `N_FFT` | systemd env file → `data_collector.py` | FFT window size |
| `THRESHOLD` | systemd env file → `data_collector.py` | ML probability threshold |
| `SKIP_CHUNKS` | systemd env file → `data_collector.py` | Startup warmup chunk count |
| `MANUAL_TRIGGER_ARGS` | systemd env file → `data_collector.py` | Enable manual-trigger-only mode |
| `MODEL_PATH_ARGS` | systemd env file → `data_collector.py` | Path to XGBoost model file |
| `MQTT_ARGS` | systemd env file → `data_collector.py` | MQTT host/port/topic flags |
| `MQTT_AUTH_ARGS` | systemd env file → `data_collector.py` | MQTT username/password flags |
| `MQTT_TLS_ARGS` | systemd env file → `data_collector.py` | MQTT TLS flags and cert paths |
| `GPIO_ARGS` | systemd env file → `data_collector.py` | GPIO pin flag |

## Webhooks & Callbacks

**Incoming:** None.

**Outgoing:** None. The system is purely a local consumer/detector; notifications to smartphones are out of scope in the current implementation (noted in README and ROADMAP as a future goal).

---

*Integration audit: 2026-05-26*
