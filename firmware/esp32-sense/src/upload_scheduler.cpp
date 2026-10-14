#include "upload_scheduler.h"

#include <Arduino.h>

#include <atomic>

#include "s3_uploader.h"
#include "sd_storage.h"

namespace upload_scheduler {

namespace {
std::atomic<bool> g_requested{false};
} // namespace

void begin() { g_requested = false; }

void request() { g_requested = true; }

namespace {

void run_upload_pass() {
  if (!sd_storage::lock()) {
    return;
  }

  const auto recordings = sd_storage::list_recordings();
  Serial.printf("upload: pass started, %u file(s)\n",
                static_cast<unsigned>(recordings.size()));
  for (const auto &path : recordings) {
    if (s3_uploader::upload_file(path)) {
      sd_storage::remove_file(path);
    }
    // On failure, leave the file in place — retried on the next pass.
  }

  sd_storage::unlock();
}

} // namespace

void tick() {
  // Clear before the pass so a request arriving mid-pass triggers another one.
  if (g_requested.exchange(false)) {
    run_upload_pass();
  }
}

} // namespace upload_scheduler
