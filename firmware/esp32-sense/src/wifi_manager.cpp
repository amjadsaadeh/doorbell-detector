#include "wifi_manager.h"

#include <Arduino.h>
#include <WiFi.h>

#include "config.h"

namespace wifi_manager {

namespace {
uint32_t g_last_attempt_ms = 0;
constexpr uint32_t kReconnectIntervalMs = 10000;
} // namespace

void begin() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  g_last_attempt_ms = millis();
}

bool is_connected() { return WiFi.status() == WL_CONNECTED; }

void maybe_reconnect() {
  if (is_connected()) {
    return;
  }
  if (millis() - g_last_attempt_ms >= kReconnectIntervalMs) {
    WiFi.disconnect();
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    g_last_attempt_ms = millis();
  }
}

} // namespace wifi_manager
