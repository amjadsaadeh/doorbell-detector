---
phase: 02-detection-notification
plan: 01
subsystem: audio
tags: [numpy, cross-correlation, pyaudio, scipy, detection, cooldown]

# Dependency graph
requires:
  - phase: 01-script-foundation
    provides: detector.py skeleton with template loading, device setup, and capture loop stub
provides:
  - compute_score() cross-correlation function (normalized by template energy)
  - --threshold and --cooldown-seconds CLI flags
  - Per-chunk detection loop with cooldown gate and "Doorbell detected!" log output
affects:
  - 02-02: MQTT notification plan — will fire publish after detection log line
  - 03: data-collection plan — save flag wired to detection event

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Normalized cross-correlation: np.correlate(mode='full') / template_energy for scale-invariant matching"
    - "Cooldown gate via time.monotonic() float comparison — no threading required"

key-files:
  created:
    - tests/detector_test.py
  modified:
    - data_collection/detector.py

key-decisions:
  - "Normalize cross-correlation by template energy so score is scale-invariant — matching audio at any volume produces a consistent score"
  - "Use time.monotonic() (not time.time()) for cooldown gate — immune to system clock adjustments"
  - "Guard zero-energy template in compute_score() with early return 0.0 to prevent division by zero"

patterns-established:
  - "TDD cycle: test commit (0e56faa) → feat commit (9e7a666) — established for Phase 2 feature work"
  - "compute_score accepts raw bytes from PyAudio directly — avoids intermediate copies"

requirements-completed: [DET-02, DET-03, DET-04, NOT-01]

# Metrics
duration: 3min
completed: 2026-06-25
---

# Phase 2 Plan 01: Detection Loop Summary

**Cross-correlation detection loop with normalized score, configurable threshold (0.7), and 10 s cooldown gate added to detector.py**

## Performance

- **Duration:** 3 min
- **Started:** 2026-06-25T19:22:20Z
- **Completed:** 2026-06-25T19:25:37Z
- **Tasks:** 2 (+ 1 TDD RED commit)
- **Files modified:** 2

## Accomplishments

- Added `compute_score(audio_bytes, template)` using `np.correlate(mode='full')` normalized by template energy; silence returns exactly 0.0 and a self-matching signal returns ~16384
- Added `--threshold` (default 0.7) and `--cooldown-seconds` (default 10.0) to `parse_args()`
- Replaced Phase 1 discard stub (`_data`) with per-chunk scoring, cooldown gate, and `log.info("Doorbell detected! score=%.4f")` output
- Added 7-test TDD suite covering silence score, high score on match, return type, and all four CLI flag permutations

## Task Commits

Each task was committed atomically:

1. **TDD RED — failing tests** - `0e56faa` (test)
2. **Task 1: compute_score() + CLI flags** - `9e7a666` (feat)
3. **Task 2: detection loop** - `f01f6c3` (feat)

## Files Created/Modified

- `data_collection/detector.py` - Added `import datetime`, `import time`, `compute_score()`, `--threshold`/`--cooldown-seconds` flags, and detection loop replacing Phase 1 stub
- `tests/detector_test.py` - New test file: 7 tests covering compute_score behavior and parse_args defaults/overrides

## Decisions Made

- Normalized cross-correlation by template energy so the threshold is meaningful across different template amplitudes
- Used `time.monotonic()` for cooldown to avoid clock-skew issues
- Guarded zero-energy template with early `return 0.0` to prevent ZeroDivisionError

## Deviations from Plan

None — plan executed exactly as written.

The module-level docstring still refers to "Phase 1 skeleton" and the Phase 2 will-extend note. This is stale documentation but does not affect behaviour; it will naturally be resolved when the full Phase 2 description is written.

## Issues Encountered

None.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- `compute_score()` is importable and tested; 02-02 (MQTT notification) can call it directly
- `--threshold` and `--cooldown-seconds` flags are live; users can tune sensitivity immediately
- Detection log format `"Doorbell detected! score=%.4f"` is established — if 02-02 logs a publish event it should use a consistent adjacent format

---
*Phase: 02-detection-notification*
*Completed: 2026-06-25*
