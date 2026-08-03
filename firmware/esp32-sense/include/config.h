// Compile-time constants. Secrets (WiFi, MQTT, MinIO) are injected as
// preprocessor defines from platformio.ini's build_flags — see
// .env.esp32.example for the full list of ESP32_* env vars that feed them.
#pragma once

#include <cstddef>
#include <cstdint>

// --- Audio format (project-wide constant, matches the Pi scripts) ---------
constexpr uint32_t kSampleRateHz = 16000;
constexpr int kBitsPerSample = 16;
constexpr int kChannels = 1;

// XIAO ESP32S3 Sense onboard PDM microphone pins.
constexpr int kPdmClkPin = 42;
constexpr int kPdmDataPin = 41;

// XIAO ESP32S3 Sense onboard SD card slot (SPI mode).
constexpr int kSdCsPin = 21;

// --- Ring buffer / recording ------------------------------------------------
// Mirrors the Pi's BUFFER_SECONDS / POST_TRIGGER_MINUTES defaults.
constexpr float kPreTriggerSeconds = 3.0f;
constexpr float kPostTriggerSeconds = 6.0f;
// Extra headroom in the clip buffer so a slightly-late post-trigger read
// never overruns the allocation.
constexpr float kClipBufferMarginSeconds = 1.0f;

constexpr size_t kBytesPerSample = kBitsPerSample / 8;
constexpr size_t kBytesPerSecond = kSampleRateHz * kBytesPerSample * kChannels;
constexpr size_t kRingBufferBytes =
    static_cast<size_t>(kPreTriggerSeconds * kBytesPerSecond);
constexpr size_t kClipBufferBytes = static_cast<size_t>(
    (kPreTriggerSeconds + kPostTriggerSeconds + kClipBufferMarginSeconds) *
    kBytesPerSecond);

// PDM read chunk size — small enough for responsive trigger handling, large
// enough to avoid excessive per-call overhead.
constexpr size_t kPdmReadChunkBytes = 1024;

constexpr const char *kRecordingsDir = "/recordings";
// In-progress recordings are written as "<name>.wav.tmp" and renamed to
// "<name>.wav" only after a clean close, so a crash mid-write can never leave
// a half-written file for the upload scanner to pick up.
constexpr const char *kTempSuffix = ".tmp";

// --- Networking / MQTT ------------------------------------------------------
#ifndef MQTT_PORT
#define MQTT_PORT 1883
#endif

// --- Daily upload ------------------------------------------------------------
constexpr int kUploadHour = 3; // local time, from NTP + TZ_STRING
constexpr uint32_t kSchedulerPollIntervalMs = 10 * 60 * 1000; // 10 min
constexpr uint32_t kNtpResyncIntervalMs = 6UL * 60 * 60 * 1000; // 6 hours

// --- S3 / MinIO --------------------------------------------------------------
constexpr const char *kS3Service = "s3";
constexpr const char *kS3KeyPrefix = "esp32-recordings";
