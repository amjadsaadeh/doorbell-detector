---
phase: 01-script-foundation
plan: 01
subsystem: audio
tags: [pyaudio, librosa, numpy, argparse, python]

# Dependency graph
requires: []
provides:
  - data_collection/detector.py with CLI entry point, template loading, device discovery, and capture loop stub
affects: [02-detection-notification, 03-data-collection]

# Tech tracking
tech-stack:
  added: [librosa, pyaudio, numpy]
  patterns:
    - "Audio constants block (CHANNELS=1, FORMAT=paInt16, RATE=16000) at module top level"
    - "logging.basicConfig + log = logging.getLogger(__name__) setup verbatim from data_collector.py"
    - "find_device() extracted as reusable function (vs inline loop in data_collector.py)"
    - "try/finally teardown: stream.stop_stream() -> stream.close() -> p.terminate()"

key-files:
  created:
    - data_collection/detector.py
  modified: []

key-decisions:
  - "load_template() does not catch exceptions — callers handle FileNotFoundError and generic Exception separately"
  - "find_device() returns None (not raises) when device absent, allowing caller to terminate p before sys.exit"
  - "--template uses required=True so argparse auto-generates usage error (exit 2) without manual validation"
  - "Capture loop stub discards all audio in Phase 1 — Phase 2 will add cross-correlation"

patterns-established:
  - "Helper functions load_template() and find_device() are importable for Phase 2 reuse"
  - "sys.exit(1) for all runtime failures after argument parsing; sys.exit(2) delegated to argparse"
  - "OSError in stream.read() caught per-iteration with log.warning and continue (matching data_collector.py)"

requirements-completed: [AUD-01, AUD-02, AUD-03, DET-01]

# Metrics
duration: 4min
completed: 2026-05-26
---

# Phase 1 Plan 01: Script Foundation Summary

**detector.py CLI skeleton with pyaudio device discovery, librosa template loading at 16 kHz, and graceful error exits for all three failure modes**

## Performance

- **Duration:** 4 min
- **Started:** 2026-05-26T17:05:45Z
- **Completed:** 2026-05-26T17:09:00Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments
- Created `data_collection/detector.py` as a standalone sibling to `data_collector.py`
- All three Phase 1 error exits verified: missing `--template` arg (exit 2), missing template file (exit 1), missing audio device (exit 1)
- Audio constants match project standard exactly: CHANNELS=1, FORMAT=pyaudio.paInt16, RATE=16000
- Capture loop stub runs indefinitely, reads and discards audio, exits cleanly on Ctrl-C

## Task Commits

Each task was committed atomically:

1. **Task 1: Create detector.py with CLI parsing, template loading, and device setup** - `4afc00f` (feat)

**Plan metadata:** (docs commit — see final commit)

## Files Created/Modified
- `data_collection/detector.py` — CLI entry point: parse_args(), load_template(), find_device(), main() with capture loop stub

## Decisions Made
- `load_template()` does not catch exceptions internally — main() wraps the call in try/except so it can handle both FileNotFoundError and generic Exception with specific log messages before sys.exit(1).
- `find_device()` returns None rather than raising, keeping cleanup (p.terminate()) in main() before the exit.
- `--template` uses `required=True` in argparse so missing-arg behavior (usage message + exit 2) is handled automatically without manual validation code.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- `librosa`, `pyaudio`, and `numpy` are not installed in the system Python on this dev machine. Installed via `uv pip install` into a `.venv` at the project root for verification. Note: `.venv` is not committed (dev-only setup). The Pi runtime already has these packages per `data_collection/requirements.txt`.
- pyaudio required `python3-dev` and `portaudio19-dev` system headers (`sudo apt-get install`) before it could build.
- ALSA prints diagnostic messages to stderr when no sound card is present on the dev machine; this is expected and does not affect exit codes.

## Verification Results

| Criterion | Command | Result |
|-----------|---------|--------|
| Syntax valid | `python3 -c "import ast; ast.parse(...)"` | OK |
| Missing --template → exit 2 | `python3 detector.py` | "the following arguments are required: --template", exit=2 |
| Missing file → exit 1 | `python3 detector.py --template /tmp/nonexistent.wav` | "Template file not found: ...", exit=1 |
| Bad device name → exit 1 | `python3 detector.py --template /tmp/test.wav --device-name nonexistent_xyz` | "Audio device 'nonexistent_device_xyz' not found", exit=1 |
| Audio constants | `grep "^CHANNELS\|^FORMAT\|^RATE"` | 3 matches |
| Function count | `grep -c "^def "` | 4 (load_template, find_device, parse_args, main) |
| required=True present | `grep "required=True"` | match |
| sys.exit(1) count | `grep -c "sys.exit(1)"` | 3 matches |
| seeed-2mic-voicecard default | `grep "seeed-2mic-voicecard"` | match (default value) |

## Next Phase Readiness
- Phase 2 (Detection & Notification) can import `load_template()` and `find_device()` directly
- Audio constants block is established as the project-wide standard
- Capture loop stub in main() is the precise integration point for Phase 2 cross-correlation logic
- No blockers for Phase 2

---
*Phase: 01-script-foundation*
*Completed: 2026-05-26*

## Self-Check: PASSED

- `data_collection/detector.py` exists: FOUND
- Commit `4afc00f` exists: FOUND (git rev-parse confirms)
