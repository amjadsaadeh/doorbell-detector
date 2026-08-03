#include "audio_capture.h"

#include <Arduino.h>
#include <I2S.h>
#include <esp_heap_caps.h>

#include <cstdio>
#include <cstring>
#include <string>

#include "config.h"
#include "ntp_time.h"
#include "sd_storage.h"
#include "wav_writer.h"

namespace audio_capture {

namespace {

enum class State { kIdle, kRecording };

TaskHandle_t g_loop_task = nullptr;
State g_state = State::kIdle;

// Pre-trigger ring buffer (PSRAM). Owned exclusively by loop_tick()/core 1.
uint8_t *g_ring_buf = nullptr;
size_t g_ring_write_pos = 0;
size_t g_ring_filled = 0; // caps at kRingBufferBytes once it has wrapped once

// In-progress clip assembly (PSRAM). Sized for pre+post+margin.
uint8_t *g_clip_buf = nullptr;
size_t g_clip_len = 0;
size_t g_clip_target_len = 0;
std::string g_clip_final_path;

uint8_t g_read_chunk[kPdmReadChunkBytes];

void ring_append(const uint8_t *data, size_t len) {
  for (size_t i = 0; i < len; ++i) {
    g_ring_buf[g_ring_write_pos] = data[i];
    g_ring_write_pos = (g_ring_write_pos + 1) % kRingBufferBytes;
  }
  g_ring_filled = min(g_ring_filled + len, kRingBufferBytes);
}

// Copies the ring buffer out in chronological (oldest-to-newest) order.
void ring_snapshot_into(uint8_t *dst) {
  if (g_ring_filled < kRingBufferBytes) {
    // Buffer hasn't wrapped yet: contents from [0, g_ring_filled) are already
    // in chronological order (only reached during the first
    // kPreTriggerSeconds after boot).
    memcpy(dst, g_ring_buf, g_ring_filled);
    return;
  }
  const size_t tail_len = kRingBufferBytes - g_ring_write_pos;
  memcpy(dst, g_ring_buf + g_ring_write_pos, tail_len);
  memcpy(dst + tail_len, g_ring_buf, g_ring_write_pos);
}

std::string make_filename() {
  char name[64];
  if (ntp_time::is_synced()) {
    tm t = ntp_time::local_now();
    std::strftime(name, sizeof(name), "doorbell_manual_%Y%m%d_%H%M%S.wav",
                   &t);
  } else {
    // NTP hasn't synced yet (e.g. a trigger fires seconds after boot) —
    // fall back to a millis()-based name so the recording isn't dropped.
    std::snprintf(name, sizeof(name), "doorbell_manual_boot_%lu.wav",
                  static_cast<unsigned long>(millis()));
  }
  return std::string(kRecordingsDir) + "/" + name;
}

void begin_recording() {
  g_clip_final_path = make_filename();
  ring_snapshot_into(g_clip_buf);
  g_clip_len = g_ring_filled; // pre-trigger portion (may be < kRingBufferBytes
                              // only in the first kPreTriggerSeconds after boot)
  g_clip_target_len =
      g_clip_len + static_cast<size_t>(kPostTriggerSeconds * kBytesPerSecond);
  g_state = State::kRecording;
}

void finish_recording() {
  const std::string tmp_path = sd_storage::temp_path_for(g_clip_final_path);

  // Block indefinitely for the SD mutex rather than timing out: the only
  // other holder is the daily upload pass, which always releases it in
  // bounded time, and dropping a just-captured clip because an upload was
  // mid-transfer would be worse than a few extra seconds of write latency.
  if (sd_storage::lock()) {
    File f = SD.open(tmp_path.c_str(), FILE_WRITE);
    if (f) {
      auto header = wav_writer::build_header(
          g_clip_len, kSampleRateHz, kBitsPerSample, kChannels);
      f.write(header.data(), header.size());
      f.write(g_clip_buf, g_clip_len);
      f.close();
      sd_storage::commit_temp_file(g_clip_final_path);
    }
    sd_storage::unlock();
  }

  g_state = State::kIdle;
}

} // namespace

bool begin() {
  g_loop_task = xTaskGetCurrentTaskHandle();

  g_ring_buf = static_cast<uint8_t *>(
      heap_caps_malloc(kRingBufferBytes, MALLOC_CAP_SPIRAM));
  g_clip_buf = static_cast<uint8_t *>(
      heap_caps_malloc(kClipBufferBytes, MALLOC_CAP_SPIRAM));
  if (g_ring_buf == nullptr || g_clip_buf == nullptr) {
    // Almost always means board_build.arduino.memory_type isn't configured
    // for PSRAM (qio_opi) — fail loudly rather than let later writes
    // corrupt memory.
    return false;
  }

  // This core's I2SClass (framework-arduinoespressif32 3.20017.x) predates
  // the newer setPinsPdmRx()/I2S_MODE_PDM_RX API documented for later cores;
  // setAllPins(sck, fs, sd, outSd, inSd) + PDM_MONO_MODE is what actually
  // compiles against the pinned platform version. The PDM clock line maps to
  // the "FS" pin slot and the PDM data line to "SD" in this API's naming.
  I2S.setAllPins(-1, kPdmClkPin, kPdmDataPin, -1, -1);
  if (!I2S.begin(PDM_MONO_MODE, kSampleRateHz, kBitsPerSample)) {
    return false;
  }

  return true;
}

void loop_tick() {
  int n = I2S.read(g_read_chunk, sizeof(g_read_chunk));
  if (n <= 0) {
    return;
  }

  if (g_state == State::kIdle) {
    ring_append(g_read_chunk, static_cast<size_t>(n));

    if (ulTaskNotifyTake(pdTRUE, 0) > 0) {
      begin_recording();
    }
    return;
  }

  // kRecording: append straight into the clip buffer (the ring buffer is
  // intentionally not fed during this window, matching the Pi's behavior of
  // reading post-trigger audio directly rather than through the pre-trigger
  // buffer).
  size_t remaining = g_clip_target_len - g_clip_len;
  size_t to_copy = min(remaining, static_cast<size_t>(n));
  memcpy(g_clip_buf + g_clip_len, g_read_chunk, to_copy);
  g_clip_len += to_copy;

  if (g_clip_len >= g_clip_target_len) {
    finish_recording();
  }
}

void request_recording() {
  if (g_loop_task != nullptr) {
    xTaskNotifyGive(g_loop_task);
  }
}

bool is_recording() { return g_state == State::kRecording; }

} // namespace audio_capture
