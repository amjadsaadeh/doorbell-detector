# Doorbell Detector — Project Guide

## What This Is

Audio-based doorbell detection system running on a Raspberry Pi. The project has two
main modes: an ML-based collector (`data_collection/data_collector.py`) and a
pattern-matching detector (`data_collection/detector.py`), deployed as a systemd
service.

## GSD Workflow

This project uses [Get Shit Done](https://github.com/amjadsaadeh/gsd) for structured
planning and execution.

**Status:** v1.0 milestone complete — all 3 phases shipped 2026-06-25. Post-milestone
hardening (MQTT trigger fixes, audio drop fixes, template auto-trim) has continued
directly on `detector.py` outside the phase structure.
**Planning docs:** `.planning/`

### Workflow commands

```
/gsd-progress          # Check status, start next milestone
/gsd-new-milestone     # Scope v2 work (see Deferred Items in .planning/STATE.md)
```

### Phases (all complete)

1. **Script Foundation** — CLI, audio device setup, template loading, error handling
2. **Detection & Notification** — Cross-correlation loop, threshold/cooldown, MQTT publish
3. **Data Collection** — `--save` flag, ring buffer clips, timestamped WAV output

v2 deferred items (not yet scoped): multiple template files, FFT frequency-domain
matching, GPIO button trigger, Prometheus metrics/health endpoint.

## Codebase Notes

- Audio constants: 16 kHz, mono, int16 — never change without updating all scripts
- Device name `seeed-2mic-voicecard` is hardcoded in data_collector.py; detector.py has a `--device-name` override
- MQTT password is passed as a CLI arg (visible in `/proc/<pid>/cmdline`) — known limitation, not a bug to fix here
- `ssl.PROTOCOL_TLS` is deprecated in Python 3.12+ — present in both scripts, carry forward as-is
- `src/deploy.sh` references the old path `src/data_collector.py` (file moved) — broken, out of scope for this work
- Oversized templates are auto-trimmed to their most energetic window (see `0b3163a`)
- Cross-correlation is slow enough to drop audio in saved clips if not handled carefully (see `c8c2788`)

## Key Files

| File | Purpose |
|------|---------|
| `data_collection/data_collector.py` | Existing collector: ML + MQTT + GPIO triggers |
| `data_collection/detector.py` | Pattern-matching detector: cross-correlation, MQTT notify, `--save` clip capture |
| `data_collection/systemd/doorbell-detector.service` | systemd unit for running detector.py on the Pi |
| `data_collection/systemd/doorbell-detector.env` | Env file consumed by the systemd unit (MQTT creds, buffer/threshold config) |
| `data_collection/requirements.txt` | Pi runtime dependencies |
| `params.yaml` | ML pipeline parameters (not used by detector) |
| `.planning/REQUIREMENTS.md` | 15 v1 requirements with REQ-IDs |
| `.planning/ROADMAP.md` | 3-phase roadmap (all complete) |
| `.planning/STATE.md` | Milestone status and deferred v2 items |
