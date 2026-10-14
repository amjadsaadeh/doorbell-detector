# ESP32-S3 Sense doorbell recorder

Firmware for a Seeed XIAO ESP32S3 Sense that replaces the Raspberry Pi Zero W
recording path (`data_collection/detector.py` / `data_collector.py`). Scope is
deliberately limited to recording, not detection:

1. Subscribes to an MQTT topic (`ESP32_MQTT_TRIGGER_TOPIC`, default
   `doorbell/trigger`); any message on it records a clip to the SD card —
   `kPreTriggerSeconds` of look-back audio (from a continuously-filling ring
   buffer) plus `kPostTriggerSeconds` more, written as
   `doorbell_manual_YYYYMMDD_HHMMSS.wav` under `/recordings/`.
2. On any message to `doorbell-detector/upload` (`kMqttUploadTopic` in
   `include/config.h`), uploads everything under `/recordings/` to the MinIO bucket configured at
   build time, then deletes each file that uploaded successfully.

No cross-correlation detection, no on-device ML, no camera, no GPIO button,
no TLS (matches the existing LAN-only MQTT/MinIO deployment) — see the
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

Two values have a format the firmware is strict about:

- `ESP32_MINIO_BUCKET` is the bucket, optionally followed by a key prefix:
  `doorbell-detector/raw` puts clips in bucket `doorbell-detector` under
  `raw/esp32-recordings/<ESP32_DEVICE_ID>/`. `ESP32_MINIO_ENDPOINT` is
  `host:port` with no scheme — it is signed verbatim as the `Host` header.
- `ESP32_MINIO_FIRMWARE_PREFIX` (optional, default
  `doorbell-detector/firmware`) is where OTA images live — deliberately not
  under `raw/`, which is Label Studio's source storage.
- `ESP32_TZ` must be a POSIX TZ string (`CET-1CEST,M3.5.0,M10.5.0/3` for
  Germany), not an IANA name like `Europe/Berlin`, which newlib silently treats
  as UTC. The boot log's `tz:` line warns when the value has no offset.

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

Uploads run at 115200 baud without esptool's stub loader (`platformio.ini`):
through usbipd on WSL2 the stub stops answering after its baud switch. From
WSL, attach the board first with `usbipd attach --wsl --busid <id>` from
Windows; the port is `/dev/ttyACM0`.

A healthy boot logs:

```
tz: "CET-1CEST,M3.5.0,M10.5.0/3"
Doorbell recorder ready.
wifi: connected, ip 192.168.178.167
ntp: synced, local time 2026-10-01 11:39:43
mqtt: connected, subscribe doorbell/trigger ok, doorbell/<id>/ota ok, doorbell-detector/upload ok
```

then `mqtt: trigger …`, `record: started …` and `record: saved …` per trigger,
and one `upload: <key> -> HTTP <status>` line per file. A failed upload prints
MinIO's error body when it gets one; `-3 send payload failed` with "Connection
reset by peer" means MinIO rejected the request from its headers alone (e.g.
`SignatureDoesNotMatch`) — `mc admin trace --errors <alias>` shows the reason.

Opening the serial port resets the chip (USB-Serial-JTAG), so a monitor always
starts from a fresh boot. Nothing uploads at boot; publish to the upload
topic to test uploading.

## Updating over WiFi (OTA)

```bash
./ota.sh                  # build HEAD, upload, ask the board to update, wait
./ota.sh --allow-dirty    # same, from an uncommitted tree
```

`ota.sh` builds the image, uploads it to
`<ESP32_MINIO_FIRMWARE_PREFIX>/<ESP32_DEVICE_ID>/<version>.bin` with the
board's own `.env.esp32` credentials, and publishes `"<version> <md5>"` to
`doorbell/<ESP32_DEVICE_ID>/ota`. The board downloads it with the same SigV4
signing it uses for recordings, checks the MD5, and reboots into it. The
script exits 0 only once the board reports the new version running. The
version is `git describe --always --dirty`, also printed at boot as `fw: …`.

The board publishes its state, retained, to `doorbell/<ESP32_DEVICE_ID>/status`
— once it is off USB, that is the only place to see it:

```
fw=<version> state=running
fw=<version> state=downloading to=<new>
fw=<version> state=rebooting to=<new>
fw=<version> state=failed to=<new> reason=<why>
fw=<version> state=rolled-back from=<new>
```

**Rollback.** A new image boots unconfirmed and is only marked valid once it
reaches the MQTT broker — proof it can receive the next update. If it doesn't
within 5 minutes (`kOtaConfirmTimeoutMs`), crashes, or hits a fatal init error
first, the bootloader boots the previous image, which reports
`state=rolled-back`. An update requested mid-recording waits until the clip is
saved; a doorbell trigger *during* a download still records, but may drop
audio while flash is being written.

The first OTA-capable build has to be flashed over USB; images flashed over
USB are never put on probation.

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
2. Publish any message to `doorbell-detector/upload` (`mosquitto_pub -h
   <broker> -t doorbell-detector/upload -m x`) — the upload pass sends
   everything on the card. Confirm the log shows
   `HTTP 200` and the object appears in MinIO at
   `<ESP32_MINIO_BUCKET>/esp32-recordings/<ESP32_DEVICE_ID>/<filename>.wav`.

## Layout

```
platformio.ini         # board/framework/build_flags; secrets via ${sysenv.*}
build.sh                # sources .env.esp32, wraps `pio`
ota.sh                  # build + ship over WiFi (tools/ota_publish.py)
.env.esp32.example      # template for the git-ignored .env.esp32
include/config.h        # compile-time constants (pins, buffer sizes, timing)
src/                    # hardware-dependent modules (audio, SD, WiFi, MQTT,
                        # NTP, S3 request signing + upload, OTA,
                        # scheduler, main)
tools/ota_publish.py    # host side of OTA: MinIO upload + MQTT request
lib/sigv4core/          # pure AWS SigV4 request signing — no Arduino
                        # dependency, so it also builds for `env:native`
test/test_sigv4/        # native unit tests for lib/sigv4core
```

## Known limitations (see the project plan for the full list)

- Credentials are compiled-in `#define`s, recoverable via physical flash
  access — fine for a private home LAN, not hardened against physical theft.
- No TLS on MQTT or the MinIO upload.
- No SD free-space monitoring if uploads keep failing.
