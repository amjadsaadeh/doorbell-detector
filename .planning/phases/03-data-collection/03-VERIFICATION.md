---
phase: 03-data-collection
verified: 2026-06-25T20:40:00Z
status: passed
score: 7/7 must-haves verified
overrides_applied: 0
---

# Phase 3: Data Collection Verification Report

**Phase Goal:** When `--save` is set, every detection automatically saves a WAV clip containing pre- and post-trigger audio to disk
**Verified:** 2026-06-25T20:40:00Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| #   | Truth                                                                                           | Status     | Evidence                                                                                          |
|-----|-------------------------------------------------------------------------------------------------|------------|---------------------------------------------------------------------------------------------------|
| 1   | Running with --save produces a timestamped WAV file in --save-dir after each detection          | VERIFIED   | Lines 410-425: strftime("doorbell_%Y%m%d_%H%M%S.wav"), filepath=str(save_dir/ts), daemon thread  |
| 2   | Running without --save produces no WAV files and allocates no ring buffer                       | VERIFIED   | Line 358: ring_buf=None in else branch; detection block gated on `if args.save and ring_buf is not None` |
| 3   | The saved WAV contains pre-trigger audio (ring buffer) followed by post-trigger audio           | VERIFIED   | Lines 412-423: pre_frames=list(ring_buf), post_frames collected from stream, args=(pre_frames+post_frames, filepath) |
| 4   | --buffer-minutes controls ring buffer depth; default 0.5 min = 30 s at 500 ms chunks           | VERIFIED   | Line 253: default=0.5; Line 351: buffer_chunks=max(1, int(args.buffer_minutes*60/chunk_size_s))  |
| 5   | --post-trigger-seconds controls post-trigger duration; default 3.0 s                           | VERIFIED   | Line 259: default=3.0; Line 413: post_chunks=max(1, int(args.post_trigger_seconds/chunk_size_s)) |
| 6   | WAV filename is doorbell_YYYYMMDD_HHMMSS.wav inside --save-dir (default: recordings)            | VERIFIED   | Line 247: default="recordings"; Line 410: strftime("doorbell_%Y%m%d_%H%M%S.wav")                |
| 7   | Disk I/O runs in a daemon thread; the capture loop is not blocked by file writes                | VERIFIED   | Lines 421-425: threading.Thread(target=save_clip, daemon=True).start() — main loop resumes after .start() |

**Score:** 7/7 truths verified

### Required Artifacts

| Artifact                     | Expected                                             | Status     | Details                                                            |
|------------------------------|------------------------------------------------------|------------|--------------------------------------------------------------------|
| `data_collection/detector.py` | save_clip(), 4 new CLI args, ring buffer + threaded save in main() | VERIFIED | All present: save_clip() at line 124, args at lines 239-261, ring_buf init at lines 348-358, thread dispatch at lines 421-425 |
| `tests/detector_test.py`      | TestSaveClip and TestSaveArgs test classes            | VERIFIED   | TestSaveClip (line 86, 3 tests), TestSaveArgs (line 123, 8 tests); all 18 tests pass |

### Key Link Verification

| From                                       | To                      | Via                                        | Status  | Details                                           |
|--------------------------------------------|-------------------------|--------------------------------------------|---------|---------------------------------------------------|
| ring_buf (collections.deque in main)        | save_clip() daemon thread | list(ring_buf) snapshot at detection time  | WIRED   | Line 412: `pre_frames = list(ring_buf)`           |
| post-trigger collection (stream.read loop)  | save_clip() daemon thread | post_frames concatenated with pre_frames   | WIRED   | Lines 414-423: post_frames built then passed as `pre_frames + post_frames` |
| threading.Thread                           | save_clip               | daemon=True, started on each detection     | WIRED   | Lines 421-425: `threading.Thread(target=save_clip, ..., daemon=True).start()` |

### Data-Flow Trace (Level 4)

| Artifact                     | Data Variable   | Source                              | Produces Real Data | Status    |
|------------------------------|-----------------|-------------------------------------|--------------------|-----------|
| `data_collection/detector.py` | ring_buf frames | PyAudio stream.read() in capture loop | Yes — live PCM bytes from microphone | FLOWING |
| `data_collection/detector.py` | post_frames     | PyAudio stream.read() in post-trigger loop | Yes — live PCM bytes from microphone | FLOWING |

### Behavioral Spot-Checks

| Behavior                                   | Command                                                                                   | Result            | Status |
|--------------------------------------------|-------------------------------------------------------------------------------------------|-------------------|--------|
| All 18 tests pass (including TestSaveClip + TestSaveArgs) | `.venv/bin/python -m pytest tests/detector_test.py -v`                  | 18 passed, 1 warning (CPython stdlib Wave_write.__del__ bug — not our code) | PASS |
| stdlib wave used in save_clip(), not soundfile | `grep -n "wave.open" data_collection/detector.py`                          | Line 132: `with wave.open(filepath, "wb") as wf:` | PASS |
| ring_buf only allocated when --save active | `grep -n "deque(maxlen\|ring_buf = None" data_collection/detector.py`      | Line 352 (deque), Line 358 (None) — in if/else branches | PASS |
| Audio constants unchanged                  | `grep -n "CHANNELS\|FORMAT\|RATE" data_collection/detector.py \| head -3` | CHANNELS=1, FORMAT=pyaudio.paInt16, RATE=16000 | PASS |
| No new pip dependencies                    | Check requirements.txt                                                                    | wave, collections, threading, pathlib are all stdlib — requirements.txt unchanged | PASS |

### Probe Execution

Step 7c: SKIPPED — no probe-*.sh files declared in PLAN or present in scripts/.

### Requirements Coverage

| Requirement | Source Plan | Description                                                         | Status    | Evidence                                                                 |
|-------------|-------------|---------------------------------------------------------------------|-----------|--------------------------------------------------------------------------|
| DATA-01     | 03-01-PLAN  | When `--save` flag set, save WAV clip (ring buffer + post-trigger) on each detection | SATISFIED | `if args.save and ring_buf is not None:` block at lines 409-429 |
| DATA-02     | 03-01-PLAN  | Saved clips written to `--save-dir` with timestamped filenames      | SATISFIED | `save_dir = Path(args.save_dir)`, `strftime("doorbell_%Y%m%d_%H%M%S.wav")` |
| DATA-03     | 03-01-PLAN  | Ring buffer size configurable via `--buffer-minutes` (default 0.5) | SATISFIED | `buffer_chunks = max(1, int(args.buffer_minutes * 60 / chunk_size_s))`   |
| DATA-04     | 03-01-PLAN  | Post-trigger recording configurable via `--post-trigger-seconds` (default 3.0) | SATISFIED | `post_chunks = max(1, int(args.post_trigger_seconds / chunk_size_s))`  |

### Anti-Patterns Found

No anti-patterns detected. Scanned both `data_collection/detector.py` and `tests/detector_test.py` for TBD, FIXME, XXX, TODO, HACK, PLACEHOLDER, stub returns, and empty implementations — none found.

Note: `test_exception_logged_on_bad_path` produces a `PytestUnraisableExceptionWarning` from `wave.Wave_write.__del__` (CPython stdlib bug when `_file` is never set). This is cosmetic — the test passes correctly because `save_clip()` catches `FileNotFoundError` internally. Not flagged as a blocker.

### Human Verification Required

None. All behaviors are verifiable programmatically and the test suite confirms correct WAV header parameters (nchannels, sampwidth, framerate, nframes) via the `TestSaveClip` round-trip tests.

### Gaps Summary

No gaps. Phase goal fully achieved.

---

_Verified: 2026-06-25T20:40:00Z_
_Verifier: Claude (gsd-verifier)_
