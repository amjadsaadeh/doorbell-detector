#include "ntp_time.h"

#include <Arduino.h>

#include "config.h"

namespace ntp_time {

namespace {
// Any wall-clock time before this is treated as "not yet synced" — an
// unsynced ESP32 reports an epoch near 0 (1970), so this is a cheap,
// reliable-enough heuristic without tracking a separate success flag from
// configTzTime's async callback.
constexpr time_t kMinPlausibleEpoch = 1700000000; // 2023-11-14

uint32_t g_last_sync_attempt_ms = 0;
} // namespace

void sync() {
  configTzTime(TZ_STRING, "pool.ntp.org", "time.nist.gov");
  g_last_sync_attempt_ms = millis();
}

bool is_synced() { return time(nullptr) >= kMinPlausibleEpoch; }

tm local_now() {
  time_t now = time(nullptr);
  tm result{};
  localtime_r(&now, &result);
  return result;
}

void maybe_resync() {
  if (millis() - g_last_sync_attempt_ms >= kNtpResyncIntervalMs) {
    sync();
  }
}

} // namespace ntp_time
