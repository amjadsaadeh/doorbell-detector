#include <Arduino.h>
#include <WiFi.h>

#include <cstring>
#include <ctime>

#include "audio_capture.h"
#include "model_detector.h"
#include "mqtt_client.h"
#include "ntp_time.h"
#include "ota.h"
#include "sd_storage.h"
#include "upload_scheduler.h"
#include "wifi_manager.h"

namespace {

void network_task(void *) {
  wifi_manager::begin();
  upload_scheduler::begin();

  bool ntp_started = false;
  bool was_connected = false;
  bool was_synced = false;

  for (;;) {
    wifi_manager::maybe_reconnect();

    const bool connected = wifi_manager::is_connected();
    if (connected != was_connected) {
      was_connected = connected;
      if (connected) {
        Serial.printf("wifi: connected, ip %s\n",
                      WiFi.localIP().toString().c_str());
      } else {
        Serial.println("wifi: disconnected");
      }
    }
    if (!was_synced && ntp_time::is_synced()) {
      was_synced = true;
      tm t = ntp_time::local_now();
      char stamp[32];
      std::strftime(stamp, sizeof(stamp), "%Y-%m-%d %H:%M:%S", &t);
      Serial.printf("ntp: synced, local time %s\n", stamp);
    }

    if (connected) {
      if (!ntp_started) {
        ntp_time::sync();
        ntp_started = true;
      } else {
        ntp_time::maybe_resync();
      }

      mqtt_client::loop_tick();
      upload_scheduler::tick();
    }
    // Outside the connected branch: a new image that never gets online must
    // still hit its confirmation timeout and roll back.
    ota::tick();

    vTaskDelay(pdMS_TO_TICKS(100));
  }
}

} // namespace

void setup() {
  Serial.begin(115200);
  ntp_time::apply_tz();
  ota::begin();

  if (!sd_storage::begin()) {
    Serial.println("FATAL: SD card init failed");
    ota::roll_back_if_unconfirmed();
    while (true) {
      delay(1000);
    }
  }

  if (!audio_capture::begin()) {
    Serial.println(
        "FATAL: audio_capture init failed (PSRAM or I2S/PDM mic)");
    ota::roll_back_if_unconfirmed();
    while (true) {
      delay(1000);
    }
  }

  mqtt_client::begin();
  if (!model_detector::begin()) {
    Serial.println("FATAL: model initialization failed");
    ota::roll_back_if_unconfirmed();
    while (true) delay(1000);
  }

  // Core 0 for networking, matching the design's split from Arduino's
  // loop()/core 1 running the audio capture path.
  if (xTaskCreatePinnedToCore(network_task, "networkTask", 8192, nullptr, 1,
                              nullptr, 0) != pdPASS) {
    Serial.println("FATAL: network task allocation failed");
    ota::roll_back_if_unconfirmed();
    while (true) delay(1000);
  }

  // An IANA name like "Europe/Berlin" is not understood by newlib and is
  // silently treated as UTC; a POSIX TZ string always carries an offset digit.
  const bool tz_has_offset = std::strpbrk(TZ_STRING, "0123456789") != nullptr;
  Serial.printf("tz: \"%s\"%s\n", TZ_STRING,
                tz_has_offset ? ""
                              : " -- not a POSIX TZ string, local time is UTC");

  Serial.println("Doorbell recorder ready.");
}

void loop() { audio_capture::loop_tick(); }
