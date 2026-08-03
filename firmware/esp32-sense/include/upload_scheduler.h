// Once-daily upload of everything on the SD card to MinIO. Polled from
// networkTask; the SD directory itself is the handoff from audio_capture —
// no "file ready" signal is needed, a daily scan is sufficient.
#pragma once

namespace upload_scheduler {

void begin();

// Call periodically (e.g. every kSchedulerPollIntervalMs) from networkTask.
// Fires run_upload_pass() once per day at kUploadHour local time, plus one
// pass right after the first successful NTP sync (closes the gap where a
// reboot near the upload hour would otherwise wait a full day).
void tick();

} // namespace upload_scheduler
