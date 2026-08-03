# ESP32-S3 Sense doorbell recorder

Firmware for a Seeed XIAO ESP32S3 Sense that replaces the Raspberry Pi Zero W
recording path (`data_collection/detector.py` / `data_collector.py`). Scope is
deliberately limited to recording, not detection:

1. Subscribes to an MQTT topic (`ESP32_MQTT_TRIGGER_TOPIC`, default
   `doorbell/trigger`); any message on it records a clip to the SD card —
   `kPreTriggerSeconds` of look-back audio (from a continuously-filling ring
   buffer) plus `kPostTriggerSeconds` more, written as
   `doorbell_manual_YYYYMMDD_HHMMSS.wav` under `/recordings/`.
2. Once a day at ~3am local time (`kUploadHour` in `include/config.h`),
   uploads everything under `/recordings/` to the MinIO bucket configured at
   build time, then deletes each file that uploaded successfully.

No cross-correlation detection, no on-device ML, no camera, no GPIO button,
no OTA, no TLS (matches the existing LAN-only MQTT/MinIO deployment) — see the
project plan this was built from for the full rationale.

## Prerequisites

- [PlatformIO Core](https://platformio.org/install/cli) (`pio` on your PATH).
  If you don't have it: `uv tool install platformio` works well and needs no
  system Python changes.
- A Seeed XIAO ESP32S3 Sense connected over USB for flashing/monitoring.

## One-time setup

```bash
cd firmware/esp32-sense
cp .env.esp32.example .env.esp32
# edit .env.esp32: WiFi creds, MQTT broker, and the *same* MinIO
# endpoint/access key/secret the Python pipeline's root .env already uses
# (AWS_ENDPOINT_URL / AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY) — copy the
# values across, don't invent new credentials.
```

`.env.esp32` is git-ignored; nothing in it is ever committed. `build.sh`
sources it and forwards every value into `platformio.ini`'s `${sysenv.*}`
build flags, so credentials are baked into the compiled binary rather than
typed on the device or left in source.

## Building, flashing, monitoring

```bash
./build.sh run                  # compile only
./build.sh run -t upload         # compile + flash over USB
./build.sh device monitor        # serial monitor (115200 baud)
```

Expected serial output on a healthy boot: SD card mount, PSRAM ring-buffer
allocation, I2S/PDM mic init, WiFi connect, MQTT connect + subscribe, then
NTP sync once WiFi is up.

## Running the unit tests (no hardware needed)

```bash
./build.sh test -e native
```

This builds and runs `test/test_sigv4/test_sigv4.cpp` against the published
AWS SigV4 "get-vanilla" test vector — the canonical request, string-to-sign,
signing-key derivation, and final signature are all checked independently, so
the hand-rolled request-signing code (`lib/sigv4core/`) is verified without
needing a live MinIO instance or the board attached.

## End-to-end verification checklist

Once flashed, to confirm the two behaviors actually work:

1. Publish any message to the trigger topic (`mosquitto_pub -h <broker> -t
   doorbell/trigger -m x`) and confirm a new
   `/recordings/doorbell_manual_*.wav` appears on the SD card, playable and
   ~`kPreTriggerSeconds + kPostTriggerSeconds` long (9s with the defaults).
2. Either wait for the configured upload hour, or temporarily set
   `kUploadHour` in `include/config.h` to the current hour and reflash, and
   confirm the file disappears from the SD card and a corresponding object
   appears in the MinIO bucket at
   `esp32-recordings/<ESP32_DEVICE_ID>/<filename>.wav`.

## Layout

```
platformio.ini         # board/framework/build_flags; secrets via ${sysenv.*}
build.sh                # sources .env.esp32, wraps `pio`
.env.esp32.example      # template for the git-ignored .env.esp32
include/config.h        # compile-time constants (pins, buffer sizes, timing)
src/                    # hardware-dependent modules (audio, SD, WiFi, MQTT,
                        # NTP, S3 upload, scheduler, main)
lib/sigv4core/          # pure AWS SigV4 request signing — no Arduino
                        # dependency, so it also builds for `env:native`
test/test_sigv4/        # native unit tests for lib/sigv4core
```

## Known limitations (see the project plan for the full list)

- Credentials are compiled-in `#define`s, recoverable via physical flash
  access — fine for a private home LAN, not hardened against physical theft.
- No TLS on MQTT or the MinIO upload.
- No SD free-space monitoring if uploads keep failing.
