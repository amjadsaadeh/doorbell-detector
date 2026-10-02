#include "mqtt_client.h"

#include <Arduino.h>
#include <PubSubClient.h>
#include <WiFi.h>

#include <cstring>
#include <string>

#include "audio_capture.h"
#include "config.h"
#include "ota.h"
#include "wifi_manager.h"

namespace mqtt_client {

namespace {

WiFiClient g_wifi_client;
PubSubClient g_client(g_wifi_client);
uint32_t g_last_reconnect_attempt_ms = 0;
constexpr uint32_t kReconnectIntervalMs = 5000;

std::string g_ota_topic;
std::string g_status_topic;

void on_message(char *topic, uint8_t *payload, unsigned int length) {
  if (g_ota_topic == topic) {
    ota::request(reinterpret_cast<const char *>(payload), length);
    return;
  }
  Serial.printf("mqtt: trigger on %s (%u bytes)\n", topic, length);
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
    const bool subscribed = g_client.subscribe(MQTT_TRIGGER_TOPIC);
    const bool ota_subscribed = g_client.subscribe(g_ota_topic.c_str());
    Serial.printf("mqtt: connected, subscribe %s %s, %s %s\n",
                  MQTT_TRIGGER_TOPIC, subscribed ? "ok" : "FAILED",
                  g_ota_topic.c_str(), ota_subscribed ? "ok" : "FAILED");
    ota::on_mqtt_connected();
  } else {
    Serial.printf("mqtt: connect failed, state %d\n", g_client.state());
  }
  return ok;
}

} // namespace

void begin() {
  const std::string device_prefix =
      std::string(kMqttDeviceTopicPrefix) + DEVICE_ID;
  g_ota_topic = device_prefix + "/ota";
  g_status_topic = device_prefix + "/status";
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

void publish_status(const char *payload) {
  if (g_client.connected()) {
    g_client.publish(g_status_topic.c_str(), payload, /*retained=*/true);
  }
}

} // namespace mqtt_client
