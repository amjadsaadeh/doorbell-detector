---
status: partial
phase: 02-detection-notification
source: [02-VERIFICATION.md]
started: 2026-06-25T21:00:00Z
updated: 2026-06-25T21:00:00Z
---

## Current Test

[awaiting human testing]

## Tests

### 1. End-to-end doorbell detection on Pi
expected: Playing doorbell sound near mic causes `Doorbell detected! score=...` log line within one 500 ms buffer window; same sound within cooldown window (10 s) produces no second log line
result: [pending]

### 2. MQTT publish to real broker
expected: On each detection, an ISO-format timestamp message is published to `doorbell/detected` (or configured topic) and received by a subscribed MQTT client
result: [pending]

### 3. MQTT username/password auth
expected: `--mqtt-username myuser --mqtt-password s3cr3t` connects successfully to an auth-required broker; omitting either flag leaves auth disabled without errors
result: [pending]

### 4. MQTT TLS
expected: `--mqtt-tls --mqtt-tls-ca /etc/ssl/certs/ca-certificates.crt` connects to a TLS broker; ssl.PROTOCOL_TLS still functional under Python 3.11 on Pi
result: [pending]

### 5. Systemd service with blank MQTT_HOST
expected: Service file with `MQTT_HOST=` (blank) starts cleanly — no MQTT connection attempted, no errors from undefined MQTT_USERNAME/MQTT_PASSWORD env vars
result: [pending]

## Summary

total: 5
passed: 0
issues: 0
pending: 5
skipped: 0
blocked: 0

## Gaps
