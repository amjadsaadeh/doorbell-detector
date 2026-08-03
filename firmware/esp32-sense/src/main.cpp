// XIAO ESP32S3 Sense doorbell recorder.
//
// Scope: (1) MQTT-triggered recording to SD, (2) once-daily upload of the SD
// card's contents to MinIO. No cross-correlation detection, no on-device ML,
// no camera use — see firmware/esp32-sense/README.md and the project plan
// this was built from.
//
// Concurrency: Arduino's loop() already runs as its own FreeRTOS task
// (pinned to core 1 by default) — that's where audio_capture lives, so it
// spawns no task of its own. One extra task, networkTask (core 0), services
// WiFi/MQTT and the daily upload scheduler.
#include <Arduino.h>

#include "audio_capture.h"
#include "mqtt_client.h"
#include "ntp_time.h"
#include "sd_storage.h"
#include "upload_scheduler.h"
#include "wifi_manager.h"

namespace {

void network_task(void *) {
  wifi_manager::begin();
  mqtt_client::begin();
  upload_scheduler::begin();

  bool ntp_started = false;

  for (;;) {
    wifi_manager::maybe_reconnect();

    if (wifi_manager::is_connected()) {
      if (!ntp_started) {
        ntp_time::sync();
        ntp_started = true;
      } else {
        ntp_time::maybe_resync();
      }

      mqtt_client::loop_tick();
      upload_scheduler::tick();
    }

    vTaskDelay(pdMS_TO_TICKS(100));
  }
}

} // namespace

void setup() {
  Serial.begin(115200);

  if (!sd_storage::begin()) {
    Serial.println("FATAL: SD card init failed");
    while (true) {
      delay(1000);
    }
  }

  if (!audio_capture::begin()) {
    Serial.println(
        "FATAL: audio_capture init failed (PSRAM or I2S/PDM mic)");
    while (true) {
      delay(1000);
    }
  }

  // Core 0 for networking, matching the design's split from Arduino's
  // loop()/core 1 running the audio capture path.
  xTaskCreatePinnedToCore(network_task, "networkTask", 8192, nullptr, 1,
                          nullptr, 0);

  Serial.println("Doorbell recorder ready.");
}

void loop() { audio_capture::loop_tick(); }
