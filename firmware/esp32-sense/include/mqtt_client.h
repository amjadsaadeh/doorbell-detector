// MQTT client: connects, subscribes to MQTT_TRIGGER_TOPIC, and forwards any
// message on it to audio_capture::request_recording() — matching the Pi
// scripts' "any payload triggers a recording" semantics (no payload-value
// matching is required for this build). Also subscribes to the device's OTA
// topic and routes it to ota::request().
#pragma once

namespace mqtt_client {

void begin();

// Services the MQTT client (must be called frequently from networkTask) and
// reconnects if the connection dropped.
void loop_tick();

bool is_connected();

// Publishes retained to "doorbell/<DEVICE_ID>/status"; dropped when offline
// (the next connect publishes the current status anyway).
void publish_status(const char *payload);

} // namespace mqtt_client
