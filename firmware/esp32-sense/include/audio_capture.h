// PDM microphone capture, the PSRAM pre-trigger ring buffer, and the
// trigger -> snapshot -> post-capture -> atomic-WAV-save sequence.
//
// Ported from data_collection/data_collector.py's save_audio()/ring-buffer
// model: a continuously-overwritten ring buffer holds kPreTriggerSeconds of
// audio; on trigger, its contents are snapshotted and kPostTriggerSeconds
// more are captured directly. The ring continues tracking current audio.
// The concatenated clip is written as one WAV file.
//
// Buffers are owned by Arduino loop()/core 1. Trigger bits can be posted by
// MQTT or inference; recording state is atomic for the network task.
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
void request_recording(bool detection = false);

bool is_recording();

} // namespace audio_capture
