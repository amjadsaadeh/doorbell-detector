#include "wav_writer.h"

#include <cstring>

namespace wav_writer {

namespace {

void put_u32le(uint8_t *dst, uint32_t v) {
  dst[0] = (uint8_t)(v);
  dst[1] = (uint8_t)(v >> 8);
  dst[2] = (uint8_t)(v >> 16);
  dst[3] = (uint8_t)(v >> 24);
}

void put_u16le(uint8_t *dst, uint16_t v) {
  dst[0] = (uint8_t)(v);
  dst[1] = (uint8_t)(v >> 8);
}

} // namespace

std::array<uint8_t, kHeaderBytes> build_header(uint32_t data_bytes,
                                                uint32_t sample_rate,
                                                uint16_t bits_per_sample,
                                                uint16_t channels) {
  std::array<uint8_t, kHeaderBytes> h{};

  const uint16_t block_align = channels * (bits_per_sample / 8);
  const uint32_t byte_rate = sample_rate * block_align;
  const uint32_t riff_chunk_size = 36 + data_bytes;

  std::memcpy(&h[0], "RIFF", 4);
  put_u32le(&h[4], riff_chunk_size);
  std::memcpy(&h[8], "WAVE", 4);

  std::memcpy(&h[12], "fmt ", 4);
  put_u32le(&h[16], 16); // fmt chunk size (PCM)
  put_u16le(&h[20], 1);  // audio format = PCM
  put_u16le(&h[22], channels);
  put_u32le(&h[24], sample_rate);
  put_u32le(&h[28], byte_rate);
  put_u16le(&h[32], block_align);
  put_u16le(&h[34], bits_per_sample);

  std::memcpy(&h[36], "data", 4);
  put_u32le(&h[40], data_bytes);

  return h;
}

} // namespace wav_writer
