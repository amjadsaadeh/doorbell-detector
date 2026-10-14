// On-demand upload of everything on the SD card to MinIO. An MQTT message on
// kMqttUploadTopic calls request(); tick() (polled from networkTask) then runs
// the pass. The SD directory itself is the handoff from audio_capture — no
// "file ready" signal is needed, a scan at upload time is sufficient.
#pragma once

namespace upload_scheduler {

void begin();

// Asks for an upload pass. Safe to call from the MQTT callback: it only sets
// a flag, so the (slow) pass never blocks the MQTT client's keepalive.
void request();

// Call frequently from networkTask. Runs one upload pass if one was requested.
void tick();

} // namespace upload_scheduler
