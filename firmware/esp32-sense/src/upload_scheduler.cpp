#include "upload_scheduler.h"

#include "config.h"
#include "ntp_time.h"
#include "s3_uploader.h"
#include "sd_storage.h"

namespace upload_scheduler {

namespace {
bool g_uploaded_today = false;
bool g_boot_catchup_done = false;
} // namespace

void begin() {
  g_uploaded_today = false;
  g_boot_catchup_done = false;
}

namespace {

void run_upload_pass() {
  if (!sd_storage::lock()) {
    return;
  }

  for (const auto &path : sd_storage::list_recordings()) {
    if (s3_uploader::upload_file(path)) {
      sd_storage::remove_file(path);
    }
    // On failure, leave the file in place — retried on the next pass.
  }

  sd_storage::unlock();
}

} // namespace

void tick() {
  if (!ntp_time::is_synced()) {
    return;
  }

  if (!g_boot_catchup_done) {
    g_boot_catchup_done = true;
    run_upload_pass(); // no-op if nothing to upload
  }

  tm t = ntp_time::local_now();
  if (t.tm_hour == kUploadHour) {
    if (!g_uploaded_today) {
      run_upload_pass();
      g_uploaded_today = true;
    }
  } else {
    g_uploaded_today = false; // re-arm once we leave the upload hour
  }
}

} // namespace upload_scheduler
