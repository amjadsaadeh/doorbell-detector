---
status: partial
phase: 01-script-foundation
source: [01-VERIFICATION.md]
started: 2026-05-26T00:00:00Z
updated: 2026-05-26T00:00:00Z
---

## Current Test

Awaiting human testing on Pi hardware.

## Tests

### 1. Capture loop on Pi hardware

Run `python3 data_collection/detector.py --template <doorbell.wav> --device-name seeed-2mic-voicecard` on the Raspberry Pi with the seeed-2mic-voicecard attached.

expected: Logs "Using audio device N: ..." and "Capture loop started — chunk=... Ctrl-C to stop", runs until Ctrl-C without crashing, exits cleanly with "Interrupted — shutting down"
result: [pending]

## Summary

total: 1
passed: 0
issues: 0
pending: 1
skipped: 0
blocked: 0

## Gaps
