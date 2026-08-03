#include "mqtt_client.h"

#include <Arduino.h>
#include <PubSubClient.h>
#include <WiFi.h>

#include <cstring>

#include "audio_capture.h"
#include "config.h"
#include "wifi_manager.h"

namespace mqtt_client {

namespace {

WiFiClient g_wifi_client;
PubSubClient g_client(g_wifi_client);
uint32_t g_last_reconnect_attempt_ms = 0;
constexpr uint32_t kReconnectIntervalMs = 5000;

void on_message(char *topic, uint8_t *payload, unsigned int length) {
  (void)topic;
  (void)payload;
  (void)length;
  // Any message on the trigger topic fires a recording — matches
  // data_collection/data_collector.py's default (no mqtt_trigger_value set).
  audio_capture::request_recording();
}

bool connect() {
  const char *client_id_suffix = DEVICE_ID;
  String client_id = String("esp32-sense-") + client_id_suffix;

  bool ok;
  if (strlen(MQTT_USERNAME) > 0) {
    ok = g_client.connect(client_id.c_str(), MQTT_USERNAME, MQTT_PASSWORD);
  } else {
    ok = g_client.connect(client_id.c_str());
  }

  if (ok) {
    g_client.subscribe(MQTT_TRIGGER_TOPIC);
  }
  return ok;
}

} // namespace

void begin() {
  g_client.setServer(MQTT_HOST, MQTT_PORT);
  g_client.setCallback(on_message);
}

void loop_tick() {
  if (!wifi_manager::is_connected()) {
    return;
  }

  if (!g_client.connected()) {
    if (millis() - g_last_reconnect_attempt_ms >= kReconnectIntervalMs) {
      g_last_reconnect_attempt_ms = millis();
      connect();
    }
    return;
  }

  g_client.loop();
}

bool is_connected() { return g_client.connected(); }

} // namespace mqtt_client
