#include "ota.h"

#include <Arduino.h>
#include <HTTPClient.h>
#include <Preferences.h>
#include <Update.h>
#include <WiFiClient.h>
#include <esp_ota_ops.h>

#include <cctype>
#include <cstdio>
#include <cstring>
#include <string>

#include "audio_capture.h"
#include "config.h"
#include "mqtt_client.h"
#include "s3_request.h"

// Weak hook in the Arduino core (esp32-hal-misc.c): returning true stops it
// from marking a PENDING_VERIFY image valid at boot, leaving that to tick().
extern "C" bool verifyRollbackLater() { return true; }

namespace ota {

namespace {

constexpr const char *kPrefsNamespace = "ota";
// Version an update was just written for, saved before the reboot into it.
// After the reboot, a mismatch with FW_VERSION means the bootloader rolled
// back: the new image never confirmed.
constexpr const char *kPrefsPendingKey = "pending";

enum class BootState { kNormal, kPendingVerify, kRolledBack };

BootState g_boot_state = BootState::kNormal;
std::string g_rejected_version; // valid when kRolledBack

bool g_request_pending = false;
char g_req_version[64];
char g_req_md5[33];

std::string firmware_prefix() {
  return std::strlen(MINIO_FIRMWARE_PREFIX) > 0 ? MINIO_FIRMWARE_PREFIX
                                                : kDefaultFirmwarePrefix;
}

std::string read_pending_version() {
  Preferences prefs;
  prefs.begin(kPrefsNamespace, true);
  std::string v = prefs.getString(kPrefsPendingKey, "").c_str();
  prefs.end();
  return v;
}

void write_pending_version(const char *version) {
  Preferences prefs;
  prefs.begin(kPrefsNamespace, false);
  if (version == nullptr) {
    prefs.remove(kPrefsPendingKey);
  } else {
    prefs.putString(kPrefsPendingKey, version);
  }
  prefs.end();
}

bool running_image_pending_verify() {
  esp_ota_img_states_t state;
  return esp_ota_get_state_partition(esp_ota_get_running_partition(),
                                     &state) == ESP_OK &&
         state == ESP_OTA_IMG_PENDING_VERIFY;
}

void publish(const std::string &state_and_details) {
  const std::string payload =
      std::string("fw=") + FW_VERSION + " state=" + state_and_details;
  mqtt_client::publish_status(payload.c_str());
}

void fail(const std::string &reason) {
  Serial.printf("ota: %s failed: %s\n", g_req_version, reason.c_str());
  publish(std::string("failed to=") + g_req_version + " reason=" + reason);
}

bool is_md5_hex(const char *s) {
  if (std::strlen(s) != 32) {
    return false;
  }
  for (const char *p = s; *p; ++p) {
    if (!std::isxdigit(static_cast<unsigned char>(*p))) {
      return false;
    }
  }
  return true;
}

// Returns only on failure; on success the board reboots into the new image.
void run_update() {
  const std::string key =
      std::string(DEVICE_ID) + "/" + g_req_version + ".bin";
  Serial.printf("ota: downloading %s/%s\n", firmware_prefix().c_str(),
                key.c_str());
  publish(std::string("downloading to=") + g_req_version);

  WiFiClient client;
  HTTPClient http;
  if (!s3_request::prepare("GET", firmware_prefix(), key, http, client)) {
    fail("cannot start request");
    return;
  }

  const int status = http.GET();
  if (status != HTTP_CODE_OK) {
    fail(status > 0 ? "HTTP " + std::to_string(status)
                    : HTTPClient::errorToString(status).c_str());
    http.end();
    return;
  }

  const int size = http.getSize();
  if (size <= 0) {
    fail("no content length");
    http.end();
    return;
  }
  if (!Update.begin(size)) {
    fail(Update.errorString());
    http.end();
    return;
  }
  Update.setMD5(g_req_md5);

  const size_t written = Update.writeStream(http.getStream());
  http.end();
  if (written != static_cast<size_t>(size)) {
    Update.abort();
    fail("short write " + std::to_string(written) + "/" +
         std::to_string(size));
    return;
  }
  // end() verifies the MD5 and switches the boot partition.
  if (!Update.end()) {
    fail(Update.getError() == UPDATE_ERROR_MD5 ? "md5" : Update.errorString());
    return;
  }

  write_pending_version(g_req_version);
  Serial.printf("ota: %u bytes written, md5 ok, rebooting into %s\n",
                static_cast<unsigned>(written), g_req_version);
  publish(std::string("rebooting to=") + g_req_version);
  delay(500); // let the status publish leave the socket
  ESP.restart();
}

} // namespace

void begin() {
  const std::string pending = read_pending_version();
  if (running_image_pending_verify()) {
    g_boot_state = BootState::kPendingVerify;
  } else if (!pending.empty() && pending != FW_VERSION) {
    g_boot_state = BootState::kRolledBack;
    g_rejected_version = pending;
    write_pending_version(nullptr);
  }

  Serial.printf("fw: %s%s\n", FW_VERSION,
                g_boot_state == BootState::kPendingVerify
                    ? " (new image, awaiting confirmation)"
                : g_boot_state == BootState::kRolledBack
                    ? (" (rolled back from " + g_rejected_version + ")")
                          .c_str()
                    : "");
}

void request(const char *payload, size_t length) {
  char buf[128];
  const size_t n = length < sizeof(buf) - 1 ? length : sizeof(buf) - 1;
  std::memcpy(buf, payload, n);
  buf[n] = '\0';

  char version[sizeof(g_req_version)];
  char md5[sizeof(g_req_md5)];
  if (std::sscanf(buf, "%63s %32s", version, md5) != 2 || !is_md5_hex(md5)) {
    Serial.printf("ota: ignoring malformed request \"%s\"\n", buf);
    return;
  }
  if (std::strcmp(version, FW_VERSION) == 0) {
    Serial.printf("ota: already running %s\n", version);
    publish("running");
    return;
  }

  std::strcpy(g_req_version, version);
  std::strcpy(g_req_md5, md5);
  g_request_pending = true;
  Serial.printf("ota: update to %s requested\n", g_req_version);
}

void tick() {
  if (g_boot_state == BootState::kPendingVerify) {
    if (mqtt_client::is_connected()) {
      esp_ota_mark_app_valid_cancel_rollback();
      write_pending_version(nullptr);
      g_boot_state = BootState::kNormal;
      Serial.println("ota: image confirmed");
      publish("running");
    } else if (millis() >= kOtaConfirmTimeoutMs) {
      Serial.println("ota: image never reached MQTT, rolling back");
      esp_ota_mark_app_invalid_rollback_and_reboot();
    }
    // A request is not acted on until this image is confirmed: the slot it
    // would overwrite holds the only known-good image.
    return;
  }

  if (!g_request_pending || audio_capture::is_recording()) {
    return; // a recording finishes first; flash writes would drop its audio
  }
  g_request_pending = false;
  run_update();
}

void roll_back_if_unconfirmed() {
  if (g_boot_state == BootState::kPendingVerify) {
    Serial.println("ota: new image cannot start, rolling back");
    esp_ota_mark_app_invalid_rollback_and_reboot();
  }
}

void on_mqtt_connected() {
  if (g_boot_state == BootState::kRolledBack) {
    publish("rolled-back from=" + g_rejected_version);
  } else if (g_boot_state == BootState::kNormal) {
    publish("running");
  }
  // kPendingVerify: tick() confirms and publishes "running".
}

} // namespace ota
