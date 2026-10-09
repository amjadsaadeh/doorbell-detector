#include "audio_capture.h"

#include <Arduino.h>
#include <driver/i2s.h>
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

constexpr i2s_port_t kI2sPort = I2S_NUM_0;

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
  Serial.printf("record: started %s\n", g_clip_final_path.c_str());
}

void finish_recording() {
  const std::string tmp_path = sd_storage::temp_path_for(g_clip_final_path);

  // Block indefinitely for the SD mutex rather than timing out: the only
  // other holder is the upload pass, which always releases it in
  // bounded time, and dropping a just-captured clip because an upload was
  // mid-transfer would be worse than a few extra seconds of write latency.
  bool saved = false;
  if (sd_storage::lock()) {
    File f = SD.open(tmp_path.c_str(), FILE_WRITE);
    if (f) {
      auto header = wav_writer::build_header(
          g_clip_len, kSampleRateHz, kBitsPerSample, kChannels);
      f.write(header.data(), header.size());
      f.write(g_clip_buf, g_clip_len);
      f.close();
      saved = sd_storage::commit_temp_file(g_clip_final_path);
    }
    sd_storage::unlock();
  }
  Serial.printf("record: %s %s (%u bytes)\n", saved ? "saved" : "FAILED to save",
                g_clip_final_path.c_str(), static_cast<unsigned>(g_clip_len));

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

  // Use the IDF I2S driver directly. The Arduino I2S library on this core
  // (framework-arduinoespressif32 3.20017.x) configures the DMA for stereo
  // and reads only half of each DMA buffer per event, so PDM_MONO_MODE
  // delivered ~8 kS/s while the WAV header claimed 16 kHz (recordings played
  // too fast and pitched up). ONLY_LEFT gives one 16-bit sample per frame at
  // exactly kSampleRateHz. In PDM RX the clock pin is the "WS" slot.
  i2s_config_t cfg = {};
  cfg.mode = static_cast<i2s_mode_t>(I2S_MODE_MASTER | I2S_MODE_RX |
                                     I2S_MODE_PDM);
  cfg.sample_rate = kSampleRateHz;
  cfg.bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT;
  cfg.channel_format = I2S_CHANNEL_FMT_ONLY_LEFT;
  cfg.communication_format = I2S_COMM_FORMAT_STAND_I2S;
  cfg.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1;
  cfg.dma_buf_count = 8;
  cfg.dma_buf_len = 512;
  if (i2s_driver_install(kI2sPort, &cfg, 0, nullptr) != ESP_OK) {
    return false;
  }
  i2s_pin_config_t pins = {};
  pins.mck_io_num = I2S_PIN_NO_CHANGE;
  pins.bck_io_num = I2S_PIN_NO_CHANGE;
  pins.ws_io_num = kPdmClkPin;
  pins.data_out_num = I2S_PIN_NO_CHANGE;
  pins.data_in_num = kPdmDataPin;
  if (i2s_set_pin(kI2sPort, &pins) != ESP_OK) {
    return false;
  }

  return true;
}

void loop_tick() {
  size_t bytes_read = 0;
  i2s_read(kI2sPort, g_read_chunk, sizeof(g_read_chunk), &bytes_read,
           pdMS_TO_TICKS(100));
  const int n = static_cast<int>(bytes_read);
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
