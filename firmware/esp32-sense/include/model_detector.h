#pragma once
#include <cstddef>
#include <cstdint>
namespace model_detector {
bool begin();
// Capture task only. Copies audio to a bounded queue without blocking I2S.
void feed(const uint8_t* pcm, size_t bytes);
// Discard accumulated context after any interruption in capture.
void discontinuity();
}
