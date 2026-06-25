---
phase: 03-data-collection
plan: 01
subsystem: audio
tags: [python, wave, collections, threading, pyaudio, ring-buffer, wav-clip]

# Dependency graph
requires:
  - phase: 02-detection-notification
    provides: main() capture loop, compute_score(), MQTT publisher, --threshold and --cooldown-seconds CLI args
provides:
  - save_clip(frames, filepath): stdlib wave writer, runs safely in daemon thread
  - 4 new CLI args: --save, --save-dir, --buffer-minutes, --post-trigger-seconds
  - Ring buffer (collections.deque) accumulating pre-trigger audio chunks in main()
  - Post-trigger chunk collection and daemon thread dispatch on each detection
affects: [future-training-data-ingestion, dataset-labeling]

# Tech tracking
tech-stack:
  added: [collections (stdlib), threading (stdlib), wave (stdlib), pathlib.Path (stdlib)]
  patterns:
    - Ring buffer using collections.deque with maxlen for bounded pre-trigger audio
    - Daemon thread pattern for non-blocking disk I/O on detection events
    - TDD RED/GREEN commit sequence for feature gating

key-files:
  created: []
  modified:
    - data_collection/detector.py
    - tests/detector_test.py

key-decisions:
  - "stdlib wave module for WAV writing — no new pip dependencies"
  - "Daemon threads for disk I/O — main capture loop unblocked after post-trigger collection"
  - "ring_buf=None when --save is False — no deque allocation in non-save mode"
  - "Post-trigger collection is synchronous (blocks capture loop for ~3s) — acceptable because cooldown (10s default) far exceeds post-trigger duration"

patterns-established:
  - "Ring buffer pattern: collections.deque(maxlen=N) with list() snapshot at detection time"
  - "Thread pattern: threading.Thread(target=save_clip, daemon=True).start() per detection"

requirements-completed: [DATA-01, DATA-02, DATA-03, DATA-04]

# Metrics
duration: 3min
completed: 2026-06-25
---

# Phase 3 Plan 01: Data Collection Summary

**Ring buffer + daemon-thread WAV clip saving: pre-trigger deque + post-trigger stream reads write timestamped doorbell_YYYYMMDD_HHMMSS.wav files on each detection**

## Performance

- **Duration:** 3 min
- **Started:** 2026-06-25T20:22:08Z
- **Completed:** 2026-06-25T20:25:09Z
- **Tasks:** 2 (Task 1 TDD: 2 commits RED+GREEN; Task 2: 1 commit)
- **Files modified:** 2

## Accomplishments
- Added `save_clip(frames, filepath)` using stdlib `wave` — writes int16 mono 16 kHz WAV, catches all errors internally
- Added 4 new CLI args: `--save` (store_true), `--save-dir` (default "recordings"), `--buffer-minutes` (default 0.5), `--post-trigger-seconds` (default 3.0)
- Wired `collections.deque(maxlen=buffer_chunks)` ring buffer into the capture loop; each chunk appended before `compute_score()`
- On detection: snapshot `list(ring_buf)`, collect post-trigger chunks, start `threading.Thread(daemon=True)` to write WAV without blocking the capture loop
- 11 new tests (TestSaveClip: 3, TestSaveArgs: 8) all passing; 7 prior tests unchanged

## Task Commits

Each task was committed atomically:

1. **Task 1 RED — failing tests for save_clip() and save CLI args** - `b6022cc` (test)
2. **Task 1 GREEN — save_clip(), imports, 4 CLI args** - `400b685` (feat)
3. **Task 2 — ring buffer and threaded WAV save wired into main()** - `3250c40` (feat)

**Plan metadata:** *(docs commit follows)*

_Note: Task 1 followed TDD RED/GREEN gate sequence._

## Files Created/Modified
- `/home/orchid/projects/doorbell-detector/data_collection/detector.py` - Added 3 stdlib imports (collections, threading, wave, pathlib.Path), updated docstring, added save_clip(), 4 CLI args, ring buffer init in main(), ring buffer append in capture loop, detection-branch save block
- `/home/orchid/projects/doorbell-detector/tests/detector_test.py` - Added TestSaveClip (3 tests) and TestSaveArgs (8 tests); updated module docstring

## Decisions Made
- WAV writing uses stdlib `wave` only — no new pip dependencies introduced (confirmed: requirements.txt unchanged)
- `ring_buf = None` when `--save` is False — zero allocation overhead for non-save runs
- Post-trigger collection is synchronous (main loop blocked ~3s) — accepted because default cooldown (10s) >> default post-trigger (3s); daemon thread handles the slow disk write

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

One pytest warning observed during `test_exception_logged_on_bad_path`: `PytestUnraisableExceptionWarning` from `wave.Wave_write.__del__` when `_file` attribute was never set (Python stdlib bug, CPython issue — not in our code). The test passes correctly because `save_clip()` catches the `FileNotFoundError` before `__del__` runs; the `__del__` warning is cosmetic only and does not affect test correctness or production behaviour.

## TDD Gate Compliance

- RED gate: `b6022cc` — `test(03-01)` commit with 11 failing tests confirmed before implementation
- GREEN gate: `400b685` — `feat(03-01)` commit with all 18 tests passing

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Phase 3 is the final phase. All three phases are complete:
1. Phase 1: CLI, device setup, template loading
2. Phase 2: Cross-correlation detection, threshold/cooldown, MQTT publish
3. Phase 3: Ring buffer, timestamped WAV clip saving

The detector is feature-complete for v1. Next steps (v2 deferred items):
- Multiple template files
- FFT frequency-domain matching
- Prometheus metrics / health endpoint
- GPIO button trigger

---
*Phase: 03-data-collection*
*Completed: 2026-06-25*
