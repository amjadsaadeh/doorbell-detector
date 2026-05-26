# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-26)

**Core value:** Detect the doorbell reliably without requiring the trained ML model — a simpler, always-available first line of detection based on what the doorbell actually sounds like.
**Current focus:** Phase 1 — Script Foundation

## Current Position

Phase: 1 of 3 (Script Foundation)
Plan: 0 of ? in current phase
Status: Ready to plan
Last activity: 2026-05-26 — Roadmap created

Progress: [░░░░░░░░░░] 0%

## Performance Metrics

**Velocity:**
- Total plans completed: 0
- Average duration: —
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| - | - | - | - |

**Recent Trend:**
- Last 5 plans: —
- Trend: —

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Separate script (not a flag on data_collector.py) — keeps concerns separate
- Cross-correlation in time domain — simple, no training required
- Reuse MQTT setup from data_collector.py — consistency for users
- `--save` as optional flag — avoids filling disk during pure detection use

### Pending Todos

None yet.

### Blockers/Concerns

- MQTT password visible in `/proc/<pid>/cmdline` — known constraint, carry forward as-is
- `ssl.PROTOCOL_TLS` deprecated in Python 3.12+ — copy from existing code as-is

## Deferred Items

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| v2 | Multiple template files | Deferred | Init |
| v2 | FFT frequency-domain matching | Deferred | Init |
| v2 | GPIO button trigger | Deferred | Init |
| v2 | Prometheus metrics / health endpoint | Deferred | Init |

## Session Continuity

Last session: 2026-05-26
Stopped at: Roadmap created, ready to plan Phase 1
Resume file: None
