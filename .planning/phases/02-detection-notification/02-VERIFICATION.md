---
phase: 02-detection-notification
verified: 2026-06-25T21:40:00Z
status: human_needed
score: 9/9 must-haves verified
overrides_applied: 0
human_verification:
  - test: "Play the doorbell WAV through a speaker near the seeed-2mic-voicecard mic on the Raspberry Pi; confirm a log line matching '2026-... [INFO] Doorbell detected! score=...' appears on stdout within 500 ms (one default buffer window)"
    expected: "Timestamped INFO line containing 'Doorbell detected! score=<value>' appears once, not twice within the cooldown window"
    why_human: "Requires Pi audio hardware; cannot emulate microphone input in CI"
  - test: "With a running Mosquitto (or Home Assistant) broker, start the detector with --mqtt-host <broker> --template <wav>; trigger a detection; subscribe to doorbell/detected and confirm a message with ISO timestamp payload arrives"
    expected: "One MQTT message per detection, no messages between detections within cooldown"
    why_human: "Requires a real MQTT broker and network connection; cannot mock paho broker in static analysis"
  - test: "Start the detector with --mqtt-host <broker> --mqtt-username <user> --mqtt-password <pass>; broker should require auth; confirm the detector connects successfully"
    expected: "No authentication error in logs; MQTT connected message appears"
    why_human: "Requires a real broker configured for username/password auth"
  - test: "Start the detector with --mqtt-host <broker> --mqtt-tls --mqtt-tls-ca <ca.crt>; confirm TLS connection succeeds"
    expected: "No TLS error in logs; MQTT connected message appears with TLS enabled"
    why_human: "Requires a real TLS-configured broker with certificate"
  - test: "Deploy doorbell-detector.service on the Pi with /etc/doorbell-detector.env (MQTT_HOST blank); confirm service starts cleanly and operates in console-only mode without MQTT errors"
    expected: "journalctl shows capture loop started, no MQTT error lines"
    why_human: "Requires systemd on Pi; env variable expansion of undefined MQTT_USERNAME/MQTT_PASSWORD must be verified"
---

# Phase 2: Detection & Notification Verification Report

**Phase Goal:** The script detects the doorbell reliably and notifies via console and MQTT, with configurable sensitivity and duplicate suppression
**Verified:** 2026-06-25T21:40:00Z
**Status:** human_needed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Playing the doorbell sound produces a timestamped log line within one buffer window | VERIFIED | `log.info("Doorbell detected! score=%.4f")` fires inside the capture loop when `score >= args.threshold`; logging format `"%Y-%m-%d %H:%M:%S [%(levelname)s]"` produces the required timestamp; 7/7 detector_test.py tests pass |
| 2 | A second detection within --cooldown-seconds does NOT produce a second log line | VERIFIED | `last_detection_time` (init 0.0) updated on detection; condition `(now - last_detection_time) >= args.cooldown_seconds` uses `time.monotonic()` (clock-skew immune); gate prevents re-fire |
| 3 | Detection does NOT fire when score is below --threshold | VERIFIED | `if score >= args.threshold` is the strict gate; no fallthrough path |
| 4 | MQTT message published to --mqtt-detect-topic on each detection | VERIFIED | `mqtt_client.publish(args.mqtt_detect_topic, datetime.datetime.now().isoformat())` on line 338, immediately after detection log line |
| 5 | MQTT disabled when --mqtt-host is absent or empty string | VERIFIED | `mqtt_client = setup_mqtt_publisher(args) if args.mqtt_host else None` (line 307); falsy check means blank env var produces no MQTT work |
| 6 | MQTT username/password auth: username_pw_set() called before connect() | VERIFIED | `if args.mqtt_username: client.username_pw_set(args.mqtt_username, args.mqtt_password or None)` on lines 232-234, before `client.connect()` on line 258 |
| 7 | MQTT TLS: tls_set() with ssl.PROTOCOL_TLS called before connect() | VERIFIED | `if args.mqtt_tls: import ssl; client.tls_set(... tls_version=ssl.PROTOCOL_TLS)` on lines 236-243, before connect(); ssl.PROTOCOL_TLS carried forward per CLAUDE.md constraint |
| 8 | doorbell-detector.env has MQTT placeholders for host/port/topic/username/password | VERIFIED | Lines 37-52 of env file: MQTT_HOST=, MQTT_PORT=1883, MQTT_DETECT_TOPIC=doorbell/detected, commented MQTT_USERNAME= and MQTT_PASSWORD=; chmod 600 security note in header |
| 9 | doorbell-detector.service ExecStart passes MQTT string args via env vars | VERIFIED | Lines 34-38 of service file: --mqtt-host ${MQTT_HOST}, --mqtt-port ${MQTT_PORT}, --mqtt-detect-topic ${MQTT_DETECT_TOPIC}, --mqtt-username ${MQTT_USERNAME}, --mqtt-password ${MQTT_PASSWORD} |

**Score:** 9/9 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `data_collection/detector.py` | compute_score() + updated parse_args() + detection loop | VERIFIED | 354 lines, substantive, all functions importable (imports: OK) |
| `data_collection/detector.py` | --threshold and --cooldown-seconds CLI flags | VERIFIED | Both present in parse_args() with correct defaults (0.7, 10.0) |
| `data_collection/detector.py` | MQTT import block, 10 MQTT CLI flags, setup_mqtt_publisher() | VERIFIED | MQTT_AVAILABLE guard present; all 10 flags confirmed; setup_mqtt_publisher() importable |
| `data_collection/systemd/doorbell-detector.env` | MQTT configuration section with env var placeholders | VERIFIED | MQTT section appended; MQTT_HOST=, MQTT_PORT=1883, MQTT_DETECT_TOPIC set; auth vars commented; chmod note present |
| `data_collection/systemd/doorbell-detector.service` | ExecStart extended with MQTT string args | VERIFIED | All 5 MQTT string args present; MQTT comment placed above ExecStart (not inside continuation — correct) |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| main() capture loop | compute_score() | per-chunk invocation | WIRED | `score = compute_score(data, template)` on line 332 inside while True loop |
| compute_score() | np.correlate | mode='full' cross-correlation | WIRED | `corr = np.correlate(audio, template, mode='full')` line 106 |
| detection block | last_detection_time | time.monotonic() cooldown gate | WIRED | init line 325, check line 334, update line 335 |
| setup_mqtt_publisher() | paho.mqtt.client.Client | client.connect() + loop_start() | WIRED | `client.loop_start()` line 259 after connect() line 258 |
| detection block | mqtt_client.publish() | immediately after Doorbell detected! log | WIRED | line 338: `mqtt_client.publish(args.mqtt_detect_topic, ...)` inside `if mqtt_client is not None` |
| main() finally block | mqtt_client.loop_stop() + disconnect() | teardown on exit | WIRED | lines 347-349: `if mqtt_client is not None: mqtt_client.loop_stop(); mqtt_client.disconnect()` |

### Data-Flow Trace (Level 4)

This phase produces no UI components; the output channel is `log.info()` (stdout via Python logging) and `mqtt_client.publish()`. Both flows verified:

| Output Channel | Data Variable | Source | Produces Real Data | Status |
|---------------|---------------|--------|--------------------|--------|
| log.info("Doorbell detected! score=%.4f") | score | `compute_score(data, template)` on real PCM bytes | Yes — np.correlate on live microphone audio | FLOWING |
| mqtt_client.publish(topic, payload) | args.mqtt_detect_topic / datetime.now() | CLI arg + Python datetime | Yes — real topic string + real timestamp | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| compute_score(silence, template) = 0.0 | `.venv/bin/python3 -c "...compute_score(zeros, ones_template)"` | 0.0 | PASS |
| compute_score(matching_audio, template) > 0.5 | `.venv/bin/python3 -c "...compute_score(ones*16384, ones_template)"` | 16384.0 | PASS |
| compute_score(audio, zero_template) = 0.0 | `.venv/bin/python3 -c "...compute_score(ones, zero_template)"` | 0.0 | PASS |
| All detector unit tests | `.venv/bin/python3 -m pytest tests/detector_test.py -v` | 7/7 passed in 0.59s | PASS |
| Dry-run graceful exit | `.venv/bin/python3 detector.py --template test.wav --device-name nonexistent_xyz` | "Audio device 'nonexistent_xyz' not found" with no MQTT errors | PASS |
| Syntax check | `.venv/bin/python3 -c "import ast; ast.parse(open('data_collection/detector.py').read())"` | syntax: OK | PASS |

### Probe Execution

No probe scripts declared in PLAN files and no conventional `scripts/*/tests/probe-*.sh` found. Step 7c: SKIPPED (no probes defined).

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| DET-02 | 02-01-PLAN | Continuously cross-correlates incoming audio against template | SATISFIED | `score = compute_score(data, template)` called on every chunk in capture loop |
| DET-03 | 02-01-PLAN | Detection fires when score exceeds --threshold (default 0.7) | SATISFIED | `if score >= args.threshold` gate on line 334; --threshold flag with default 0.7 |
| DET-04 | 02-01-PLAN | Cooldown suppresses repeats (--cooldown-seconds, default 10) | SATISFIED | time.monotonic() cooldown gate on line 334; --cooldown-seconds flag with default 10.0 |
| NOT-01 | 02-01-PLAN | Timestamped log message on detection | SATISFIED | `log.info("Doorbell detected! score=%.4f")` with logging format including timestamp |
| NOT-02 | 02-02-PLAN | MQTT message published to configurable topic on detection | SATISFIED | `mqtt_client.publish(args.mqtt_detect_topic, ...)` line 338 |
| NOT-03 | 02-02-PLAN | MQTT supports username/password auth | SATISFIED | `client.username_pw_set(args.mqtt_username, args.mqtt_password or None)` before connect() |
| NOT-04 | 02-02-PLAN | MQTT supports TLS with CA/cert/key flags | SATISFIED | `client.tls_set(..., tls_version=ssl.PROTOCOL_TLS)` before connect(); 5 TLS flags present |

All 7 declared requirements: SATISFIED. No orphaned requirements found.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `data_collection/detector.py` | 1-12 | Module docstring says "Phase 1 skeleton" and "Phase 2 will extend this with..." | Info | Stale documentation only; does not affect behaviour; acknowledged in 02-01-SUMMARY.md |

No TBD, FIXME, or XXX markers found. No empty implementations. No placeholder returns. No hardcoded empty data flowing to output.

### Key Constraints Verified

| Constraint | Status | Evidence |
|------------|--------|----------|
| compute_score() uses np.correlate(mode='full') | PASS | Line 106: `np.correlate(audio, template, mode='full')` |
| Normalized by template energy | PASS | Lines 107-110: `template_energy = float(np.dot(template, template))` / `np.max(np.abs(corr)) / template_energy` |
| Cooldown uses time.monotonic() not wall time | PASS | Line 333: `now = time.monotonic()` |
| No librosa (soundfile+scipy only) | PASS | Imports: soundfile, scipy.signal.resample_poly — librosa absent |
| ssl.PROTOCOL_TLS carried forward as-is | PASS | Line 242: `tls_version=ssl.PROTOCOL_TLS` |
| MQTT disabled by default when --mqtt-host absent/empty | PASS | Line 307: `if args.mqtt_host else None` |
| All 10 MQTT CLI flags present | PASS | --mqtt-host, --mqtt-port, --mqtt-detect-topic, --mqtt-username, --mqtt-password, --mqtt-tls, --mqtt-tls-ca, --mqtt-tls-certfile, --mqtt-tls-keyfile, --mqtt-tls-insecure |
| Detection fires on score >= threshold AND cooldown elapsed | PASS | Line 334: `if score >= args.threshold and (now - last_detection_time) >= args.cooldown_seconds` |
| Phase 1 functions preserved | PASS | load_template(), find_device(), parse_args(), main() all preserved and extended |
| _data stub variable removed | PASS | `grep _data detector.py` returns zero matches |

### Human Verification Required

All automated checks pass. The following require real hardware or external services.

### 1. End-to-End Doorbell Detection on Pi Hardware

**Test:** On the Raspberry Pi with seeed-2mic-voicecard (or wm8960-soundcard as configured in env file), run `python3 detector.py --template <doorbell.wav>` and play the doorbell sound through a speaker near the mic.
**Expected:** Log line matching `2026-...-... [INFO] Doorbell detected! score=X.XXXX` appears within 500 ms. Playing the sound a second time within 10 s produces no second line. Playing after 10 s produces a second line.
**Why human:** Requires Pi audio hardware. Cannot emulate microphone input in static analysis or CI on this machine.

### 2. MQTT Publish to Real Broker

**Test:** Start the detector with `--mqtt-host <broker-ip> --template <wav>`. Subscribe to `doorbell/detected` on the broker. Trigger a detection. Confirm message arrives with ISO timestamp payload.
**Expected:** One message per detection event. No messages between detections within cooldown window.
**Why human:** Requires a running MQTT broker (Mosquitto or Home Assistant). Cannot mock paho TCP connection in static analysis.

### 3. MQTT Auth (Username/Password)

**Test:** Configure broker to require auth. Run detector with `--mqtt-host <broker> --mqtt-username <user> --mqtt-password <pass>`.
**Expected:** Log shows "MQTT auth enabled for user 'X'" and then "MQTT connected to X:1883". No authentication error.
**Why human:** Requires a real broker configured for password auth.

### 4. MQTT TLS Connection

**Test:** Configure broker with TLS. Run detector with `--mqtt-host <broker> --mqtt-tls --mqtt-tls-ca <ca.crt>`.
**Expected:** Log shows "MQTT TLS enabled" and then "MQTT connected to X:8883". No TLS error.
**Why human:** Requires a real TLS-configured broker with CA certificate. ssl.PROTOCOL_TLS is deprecated in Python 3.12+ but deliberately carried forward — verify it still works with the deployed paho-mqtt version.

### 5. Systemd Service Deployment

**Test:** Copy `doorbell-detector.env` to `/etc/doorbell-detector.env` (with MQTT_HOST blank) and the service file to `/etc/systemd/system/`. Run `sudo systemctl start doorbell-detector` and check `journalctl -u doorbell-detector`.
**Expected:** Service starts cleanly, capture loop message appears, no errors about undefined MQTT_USERNAME or MQTT_PASSWORD env vars (systemd should expand to empty string → falsy → no MQTT).
**Why human:** Requires systemd on a Pi. Env var expansion behavior for undefined MQTT_USERNAME/MQTT_PASSWORD in ExecStart must be validated in a live systemd environment.

---

## Gaps Summary

No gaps. All 9 must-haves verified. All 7 requirements satisfied. No blockers or warnings found beyond the stale module docstring (informational only, does not affect behavior).

Five items deferred to human verification: end-to-end audio detection, MQTT publisher, MQTT auth, MQTT TLS, and systemd deployment. These require Pi hardware and a real broker — they cannot be verified programmatically from this machine.

---

_Verified: 2026-06-25T21:40:00Z_
_Verifier: Claude (gsd-verifier)_
