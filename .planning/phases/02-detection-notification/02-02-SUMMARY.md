---
phase: 02-detection-notification
plan: 02
subsystem: mqtt
tags: [paho-mqtt, mqtt, tls, systemd, notification]

# Dependency graph
requires:
  - phase: 02-detection-notification
    plan: 01
    provides: compute_score() detection loop with "Doorbell detected!" log line and cooldown gate
provides:
  - setup_mqtt_publisher() with full auth/TLS pattern from data_collector.py
  - MQTT optional import block (MQTT_AVAILABLE guard)
  - 10 MQTT CLI flags in parse_args()
  - publish-on-detection with ISO timestamp payload
  - MQTT teardown in finally block (loop_stop + disconnect)
  - doorbell-detector.env MQTT section with host/port/topic/auth/TLS vars
  - doorbell-detector.service ExecStart extended with all MQTT string args
affects:
  - 03: data-collection plan — detector.py fully functional with MQTT; save flag is the last addition

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Optional paho-mqtt import via try/except + MQTT_AVAILABLE flag — graceful degradation when library absent"
    - "MQTT publisher pattern: Client() + optional auth/TLS + on_connect callback + connect() + loop_start()"
    - "Empty-string guard: if args.mqtt_host gates all MQTT work — blank env var = console-only mode"

key-files:
  created: []
  modified:
    - data_collection/detector.py
    - data_collection/systemd/doorbell-detector.env
    - data_collection/systemd/doorbell-detector.service

key-decisions:
  - "Omit on_message and subscribe() — detector publishes detections, it does not subscribe to broker topics"
  - "Publish ISO-format datetime as payload so consumers know when the detection fired"
  - "Empty MQTT_HOST in env file = MQTT disabled (Python falsy string check) — no ArgParse required=False needed"
  - "ssl.PROTOCOL_TLS carried forward as-is from data_collector.py per CLAUDE.md constraint"
  - "MQTT password as CLI arg remains visible in /proc/<pid>/cmdline — known limitation per CLAUDE.md"

patterns-established:
  - "setup_mqtt_publisher() mirrors setup_mqtt() from data_collector.py — same auth/TLS structure, publish vs subscribe role"
  - "Teardown guard: if mqtt_client is not None checks before loop_stop()/disconnect() — same pattern as data_collector.py"

requirements-completed: [NOT-02, NOT-03, NOT-04]

# Metrics
duration: 12min
completed: 2026-06-25
---

# Phase 2 Plan 02: MQTT Notification Summary

**Optional paho-mqtt publisher wired into detector.py: setup_mqtt_publisher() with auth/TLS, 10 CLI flags, detection-triggered publish, and systemd env/service updates**

## Performance

- **Duration:** 12 min
- **Started:** 2026-06-25T19:21:00Z
- **Completed:** 2026-06-25T19:33:57Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- Added `try/except` paho-mqtt import block with `MQTT_AVAILABLE` flag for graceful degradation when paho-mqtt is absent
- Added 10 MQTT CLI flags in a new `parse_args()` argument group: host, port, detect-topic, username, password, tls, tls-ca, tls-certfile, tls-keyfile, tls-insecure
- Added `setup_mqtt_publisher(args)` replicating the auth/TLS pattern from `data_collector.py` (without subscribe — detector publishes only)
- Wired `mqtt_client` into `main()`: initialized when `--mqtt-host` is set, publishes ISO timestamp on each detection, torn down in `finally`
- Updated `doorbell-detector.env` with MQTT section (host/port/topic + commented auth/TLS vars) and `chmod 600` security note
- Extended `doorbell-detector.service` ExecStart with all 5 MQTT string args via env vars

## Task Commits

Each task was committed atomically:

1. **Task 1: paho-mqtt import, CLI args, setup_mqtt_publisher()** - `59248f1` (feat)
2. **Task 2: Wire MQTT into main(), env file, service ExecStart** - `31da475` (feat)

## Files Created/Modified

- `data_collection/detector.py` - Optional paho-mqtt import block, 10 MQTT CLI flags, `setup_mqtt_publisher()`, detection publish, finally teardown
- `data_collection/systemd/doorbell-detector.env` - MQTT section appended with host/port/topic defaults and commented auth/TLS vars; chmod 600 note in header
- `data_collection/systemd/doorbell-detector.service` - ExecStart extended with `--mqtt-host`, `--mqtt-port`, `--mqtt-detect-topic`, `--mqtt-username`, `--mqtt-password` via env vars

## Decisions Made

- Reused `data_collector.py` auth/TLS pattern verbatim (same broker, same credentials model)
- Used falsy empty-string check (`if args.mqtt_host`) so blank `MQTT_HOST=` in env disables MQTT without errors
- ISO timestamp as publish payload so Home Assistant / consumers know exact detection time
- `ssl.PROTOCOL_TLS` carried forward as-is (deprecated in 3.12+, known constraint per CLAUDE.md)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Moved MQTT comment outside ExecStart continuation**
- **Found during:** Task 2 (service ExecStart update)
- **Issue:** Plan specified adding a `# comment` line inside the ExecStart multi-line continuation. In systemd unit files, a line starting with `#` terminates the continuation and is treated as a comment/separator — the MQTT args following it would be silently dropped, breaking the service.
- **Fix:** Placed the comment on its own line immediately above the ExecStart directive (valid systemd comment position) instead of inside the `\`-continuation value.
- **Files modified:** `data_collection/systemd/doorbell-detector.service`
- **Verification:** All MQTT vars appear in grep output; ExecStart is syntactically valid
- **Committed in:** `31da475` (Task 2 commit)

**2. [Rule 2 - Missing Critical] Added chmod 600 security note to env file header**
- **Found during:** Task 2 (env file update) — threat model T-02-05 has disposition "mitigate"
- **Issue:** Threat register T-02-05 (Information Disclosure: MQTT_PASSWORD in /etc/doorbell-detector.env) required documenting that the env file must be `chmod 600` owned by root. The plan body did not include this documentation.
- **Fix:** Added a `SECURITY:` comment block to the env file header with `chown root:root` and `chmod 600` instructions.
- **Files modified:** `data_collection/systemd/doorbell-detector.env`
- **Verification:** `grep -i "chmod" data_collection/systemd/doorbell-detector.env` matches
- **Committed in:** `31da475` (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (1 bug, 1 missing critical from threat model)
**Impact on plan:** Both fixes necessary for correctness and security. No scope creep.

## Issues Encountered

None beyond the two auto-fixed deviations above.

## User Setup Required

None — MQTT is disabled by default (blank `MQTT_HOST=`). To enable:
1. Edit `/etc/doorbell-detector.env` and set `MQTT_HOST=<broker>` (and optionally username/password)
2. `sudo systemctl daemon-reload && sudo systemctl restart doorbell-detector`

## Known Stubs

None — MQTT publish is fully wired. `mqtt_client.publish()` fires on every detection with a real ISO timestamp payload.

## Threat Flags

No new security-relevant surface beyond what the plan's threat model already covers (T-02-04 through T-02-SC).

## Next Phase Readiness

- `setup_mqtt_publisher()` is importable and tested via the import verification step
- `detector.py` is complete for Phases 1 and 2; only the `--save` flag (Phase 3 / plan 03) remains
- MQTT integration works in console-only mode when `MQTT_HOST` is blank — Pi can run immediately without broker
- `doorbell-detector.service` is ready to deploy once env file is configured on the Pi

---
*Phase: 02-detection-notification*
*Completed: 2026-06-25*
