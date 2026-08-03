// PDM microphone capture, the PSRAM pre-trigger ring buffer, and the
// trigger -> snapshot -> post-capture -> atomic-WAV-save sequence.
//
// Ported from data_collection/data_collector.py's save_audio()/ring-buffer
// model: a continuously-overwritten ring buffer holds kPreTriggerSeconds of
// audio; on trigger, its contents are snapshotted and kPostTriggerSeconds
// more are captured directly (the ring buffer is *not* fed during this
// window, matching the Pi's behavior), then the concatenated clip is written
// as one WAV file.
//
// Owned entirely by the Arduino loop()/core-1 task — no locking needed here.
// The only cross-core interaction is the trigger notification consumed from
// mqtt_client via a FreeRTOS task notification (see request_recording()).
#pragma once

namespace audio_capture {

bool begin();

// Call once per loop() iteration. Reads one PDM chunk, feeds the ring buffer
// or an in-progress clip depending on state, and finalizes+saves a clip when
// the post-trigger capture completes.
void loop_tick();

// Called from mqtt_client's on-message callback (a different task/core) when
// a trigger message arrives. Cheap and safe to call from another task: it
// only posts a FreeRTOS task notification to the loop() task, consumed
// non-blockingly by loop_tick(). A trigger arriving while a recording is
// already in progress is simply dropped (notifications coalesce), matching
// the Pi's guard-flag behavior of ignoring concurrent triggers.
void request_recording();

bool is_recording();

} // namespace audio_capture
