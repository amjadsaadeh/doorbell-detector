# Doorbell Detector — Project Guide

## What This Is

Audio-based doorbell detection system running on a Raspberry Pi. The project has two
main modes: an ML-based collector (`data_collection/data_collector.py`) and a new
pattern-matching detector (`data_collection/detector.py`, in progress).

## GSD Workflow

This project uses [Get Shit Done](https://github.com/amjadsaadeh/gsd) for structured
planning and execution.

**Current phase:** Phase 1 — Script Foundation
**Planning docs:** `.planning/`

### Workflow commands

```
/gsd-plan-phase 1     # Plan Phase 1 before coding
/gsd-progress         # Check status, advance workflow
/gsd-discuss-phase 1  # Talk through approach before planning
```

### Phase order

1. **Script Foundation** — CLI, audio device setup, template loading, error handling
2. **Detection & Notification** — Cross-correlation loop, threshold/cooldown, MQTT publish
3. **Data Collection** — `--save` flag, ring buffer clips, timestamped WAV output

## Codebase Notes

- Audio constants: 16 kHz, mono, int16 — never change without updating all scripts
- Device name `seeed-2mic-voicecard` is hardcoded in data_collector.py; detector.py adds `--device-name` override
- MQTT password is passed as a CLI arg (visible in `/proc/<pid>/cmdline`) — known limitation, not a bug to fix here
- `ssl.PROTOCOL_TLS` is deprecated in Python 3.12+ — already present in data_collector.py, carry forward as-is
- `src/deploy.sh` references the old path `src/data_collector.py` (file moved) — broken, out of scope for this work

## Key Files

| File | Purpose |
|------|---------|
| `data_collection/data_collector.py` | Existing collector: ML + MQTT + GPIO triggers |
| `data_collection/detector.py` | New: pattern-matching detector (to be built) |
| `data_collection/requirements.txt` | Pi runtime dependencies |
| `params.yaml` | ML pipeline parameters (not used by detector) |
| `.planning/REQUIREMENTS.md` | 15 v1 requirements with REQ-IDs |
| `.planning/ROADMAP.md` | 3-phase roadmap |
