---
phase: 01-script-foundation
verified: 2026-05-26T19:11:30Z
status: human_needed
score: 4/5 must-haves verified
overrides_applied: 0
human_verification:
  - test: "Run 'python3 detector.py --template doorbell.wav' on the Pi with seeed-2mic-voicecard attached"
    expected: "Logs 'Using audio device N: ...' and 'Capture loop started', then runs until Ctrl-C without crashing"
    why_human: "Requires physical Raspberry Pi with seeed-2mic-voicecard hardware; cannot verify device open and capture loop on dev machine without sound card"
---

# Phase 1: Script Foundation Verification Report

**Phase Goal:** The script starts cleanly, opens the audio device, loads the template WAV, and exits with clear errors on missing inputs
**Verified:** 2026-05-26T19:11:30Z
**Status:** human_needed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Running without --template prints a usage error to stderr and exits non-zero | VERIFIED | `.venv/bin/python3 detector.py` → "error: the following arguments are required: --template", exit=2 |
| 2 | Running with --template pointing to a missing file prints a file-not-found error to stderr and exits non-zero | VERIFIED | `--template /tmp/nonexistent_xyz_abc.wav` → "[ERROR] Template file not found: ...", exit=1 |
| 3 | Running with a valid template but no matching audio device prints a device-not-found error to stderr and exits non-zero | VERIFIED | `--template /tmp/test.wav --device-name nonexistent_device_xyz` → "[ERROR] Audio device 'nonexistent_device_xyz' not found", exit=1 |
| 4 | Running with a valid template and a matching audio device opens the device and enters the capture loop without crashing | UNCERTAIN | Code path is correct and wired; cannot execute end-to-end without seeed-2mic-voicecard hardware — needs human verification on Pi |
| 5 | Audio is captured at 16 kHz, mono, int16 — matching CHANNELS=1, FORMAT=pyaudio.paInt16, RATE=16000 | VERIFIED | Lines 27-29: `CHANNELS = 1`, `FORMAT = pyaudio.paInt16`, `RATE = 16000` — exact match; stream.open() passes all three constants at lines 141-143 |

**Score:** 4/5 truths verified (1 uncertain — hardware-gated)

### ROADMAP Success Criteria Coverage

| # | Success Criterion | Status | Evidence |
|---|------------------|--------|----------|
| 1 | Running without --template prints a clear usage error and exits non-zero | VERIFIED | Behavioral spot-check: exit=2, message contains "required: --template" |
| 2 | Running with --template missing.wav prints a clear file-not-found error and exits non-zero | VERIFIED | Behavioral spot-check: exit=1, "[ERROR] Template file not found: ..." logged |
| 3 | Running with a valid template and no audio device prints a clear device-not-found error and exits non-zero | VERIFIED | Behavioral spot-check: exit=1, "[ERROR] Audio device 'nonexistent_device_xyz' not found" logged |
| 4 | Running with a valid template and audio device available starts without crashing and enters the capture loop | UNCERTAIN | Logic is correct (wired); hardware unavailable on dev machine — needs Pi |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `data_collection/detector.py` | CLI entry point with arg parsing, audio device setup, template loading, and graceful error exits | VERIFIED | File exists, 171 lines, passes `ast.parse()` syntax check, exports `main`, `parse_args`, `find_device`, `load_template` at correct line positions |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `parse_args()` | `main()` | `--template required=True` — missing triggers argparse exit(2) before any device open | VERIFIED | Line 84: `required=True` in argparse; `args = parse_args()` is first call in `main()` at line 108; device open does not occur until after template load |
| `load_template()` | `librosa.load()` | Resamples WAV to RATE=16000 if source SR differs | VERIFIED | Line 54: `librosa.load(path, sr=RATE, mono=True)` — `sr=RATE` enforces 16000 Hz resampling |
| `find_device()` | `pyaudio.PyAudio()` | Iterates get_device_count(), matches args.device_name substring in info['name'] | VERIFIED | Lines 69-73: iterates `range(p.get_device_count())`, checks `device_name in info["name"] and info["maxInputChannels"] > 0`, returns index or None |

### Data-Flow Trace (Level 4)

This phase does not render dynamic data to a UI. The capture loop reads PCM bytes and discards them (Phase 1 stub by design). No Level 4 trace required.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Missing --template → exit 2 with usage error | `.venv/bin/python3 detector.py` | "error: the following arguments are required: --template", exit=2 | PASS |
| Missing template file → exit 1 with file-not-found | `.venv/bin/python3 detector.py --template /tmp/nonexistent_xyz_abc.wav` | "[ERROR] Template file not found: /tmp/nonexistent_xyz_abc.wav", exit=1 | PASS |
| Valid template + bad device → exit 1 with device error | `.venv/bin/python3 detector.py --template /tmp/test_template_verify.wav --device-name nonexistent_device_xyz` | "[ERROR] Audio device 'nonexistent_device_xyz' not found", exit=1 | PASS |
| Valid template + real device → capture loop | Requires seeed-2mic-voicecard on Pi | Cannot test on dev machine | SKIP (hardware) |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| AUD-01 | 01-01-PLAN.md | Script opens seeed-2mic-voicecard input device (with --device-name override) | VERIFIED | `--device-name` arg with default "seeed-2mic-voicecard" at line 88-91; `find_device()` performs substring match; `p.open()` at line 140 uses `input_device_index=device_idx` |
| AUD-02 | 01-01-PLAN.md | Audio captured at 16 kHz, mono, int16 — matching project standard format | VERIFIED | Constants block lines 27-29: `CHANNELS=1`, `FORMAT=pyaudio.paInt16`, `RATE=16000`; `p.open()` passes all three |
| AUD-03 | 01-01-PLAN.md | Script logs a clear error and exits gracefully if audio device not found or template file missing | VERIFIED | Three distinct exit paths: argparse exit(2) for missing --template, `sys.exit(1)` after `log.error()` for missing file (lines 114-115), `sys.exit(1)` after `log.error()` for missing device (lines 129-131); `p.terminate()` called before device-not-found exit |
| DET-01 | 01-01-PLAN.md | Script loads a reference doorbell sound from a WAV file specified via --template <path> | VERIFIED | `load_template(args.template)` called at line 112; `librosa.load(path, sr=RATE, mono=True)` performs actual load and resampling; result bound to `template` variable; length logged at line 120 |

All 4 requirement IDs declared in PLAN frontmatter are accounted for. No orphaned requirements: REQUIREMENTS.md traceability table maps AUD-01, AUD-02, AUD-03, DET-01 to Phase 1 — all covered.

### Anti-Patterns Found

None. Scan for TODO/FIXME/HACK/placeholder text, empty returns, and hardcoded empty data returned zero matches. The Phase 1 capture loop stub (`_data = stream.read(...)` discarded) is intentional and documented in the module docstring and PLAN — it is not a hidden placeholder; Phase 2 is the designated integration point.

### Human Verification Required

#### 1. Capture Loop on Pi Hardware

**Test:** On a Raspberry Pi with seeed-2mic-voicecard attached and librosa/pyaudio installed, run:
```
python3 data_collection/detector.py --template /path/to/doorbell.wav
```
**Expected:** Log output shows "Using audio device N: ..." followed by "Capture loop started — chunk=8000 samples (500 ms), Ctrl-C to stop"; script runs indefinitely reading audio until Ctrl-C; Ctrl-C produces "Interrupted — shutting down" and clean exit.
**Why human:** Requires physical seeed-2mic-voicecard hardware. ALSA/PortAudio on the dev machine has no sound cards — `find_device()` returns None for any device name, making it impossible to exercise the stream-open and capture-loop code path without the Pi.

## Gaps Summary

No blockers. The single uncertain truth (SC #4, capture loop with real hardware) is gated on Pi hardware availability, not on any code deficiency. The logic for device open and capture loop is fully present, correctly wired, and matches the plan specification exactly. Three of the four ROADMAP success criteria are demonstrably passing via automated behavioral spot-checks.

---

_Verified: 2026-05-26T19:11:30Z_
_Verifier: Claude (gsd-verifier)_
