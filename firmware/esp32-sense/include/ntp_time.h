// NTP wall-clock sync. The XIAO ESP32S3 Sense has no onboard RTC, so local
// time is only meaningful once this has synced at least once after a WiFi
// connection, and needs periodic resync since the internal clock drifts.
#pragma once

#include <ctime>

namespace ntp_time {

// Starts the sync (configTzTime with TZ_STRING); non-blocking, call once
// WiFi is connected. Safe to call again to force a resync.
void sync();

bool is_synced();

// Local time per TZ_STRING. Only meaningful if is_synced() is true.
tm local_now();

// Calls sync() again if kNtpResyncIntervalMs has elapsed since the last
// successful sync. Call periodically from networkTask.
void maybe_resync();

} // namespace ntp_time
