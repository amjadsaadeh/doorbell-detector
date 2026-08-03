// MQTT client: connects, subscribes to MQTT_TRIGGER_TOPIC, and forwards any
// message on it to audio_capture::request_recording() — matching the Pi
// scripts' "any payload triggers a recording" semantics (no payload-value
// matching is required for this build).
#pragma once

namespace mqtt_client {

void begin();

// Services the MQTT client (must be called frequently from networkTask) and
// reconnects if the connection dropped.
void loop_tick();

bool is_connected();

} // namespace mqtt_client
