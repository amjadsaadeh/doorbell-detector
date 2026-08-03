// SD card lifecycle: mount, path helpers, directory listing/deletion, and the
// mutex that guards the SD/SPI peripheral from the two cores that touch it
// (audio_capture writing a clip, upload_scheduler scanning/deleting).
#pragma once

#include <FS.h>
#include <SD.h>

#include <string>
#include <vector>

namespace sd_storage {

bool begin();

// Held by any code that touches the SD card. Required because the Arduino
// SD/SPI stack is not safe for concurrent access from two FreeRTOS tasks —
// this is genuinely shared mutable state, not just a producer/consumer
// signal (unlike the MQTT-trigger handoff, which uses a task notification).
bool lock(uint32_t timeout_ms = portMAX_DELAY);
void unlock();

// Full paths ("/recordings/xxx.wav") of completed (non-.tmp) recordings.
// Caller must hold the lock.
std::vector<std::string> list_recordings();

// Caller must hold the lock.
bool remove_file(const std::string &path);
size_t file_size(const std::string &path);

// Renames "<path>.tmp" to "<path>", completing the atomic-write pattern used
// by audio_capture. Caller must hold the lock.
bool commit_temp_file(const std::string &final_path);

std::string temp_path_for(const std::string &final_path);

} // namespace sd_storage
