// Pure WAV header construction — no I/O, no Arduino dependency, so it stays
// as trivially correct/testable as the format itself.
#pragma once

#include <array>
#include <cstddef>
#include <stdint.h>

namespace wav_writer {

constexpr size_t kHeaderBytes = 44;

// Canonical 44-byte RIFF/WAVE/fmt /data header for uncompressed PCM.
std::array<uint8_t, kHeaderBytes> build_header(uint32_t data_bytes,
                                                uint32_t sample_rate,
                                                uint16_t bits_per_sample,
                                                uint16_t channels);

} // namespace wav_writer
