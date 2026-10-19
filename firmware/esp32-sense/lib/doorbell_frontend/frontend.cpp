#include "frontend.h"
#include "frontend_tables.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace doorbell {
void Frontend::reset() {
  std::memset(pcm_, 0, sizeof(pcm_));
  write_ = frames_ = feature_write_ = 0;
  until_frame_ = kFft / 2;
}
bool Frontend::push(int16_t sample) {
  pcm_[write_] = sample;
  write_ = (write_ + 1) % kFft;
  if (--until_frame_) return false;
  until_frame_ = kHop;
  frame();
  feature_write_ = (feature_write_ + 1) % kFrames;
  if (frames_ < kFrames) ++frames_;
  return true;
}
void Frontend::frame() {
  // The ring's write pointer is the oldest sample. Until full, the unused
  // tail supplies the left zero padding required by librosa center=True.
  for (size_t i = 0; i < kFft; ++i) {
    re_[i] = pcm_[(write_ + i) % kFft] * kHann[i];
    im_[i] = 0;
  }
  // In-place radix-2 FFT; no allocation, unscaled (numpy/librosa convention).
  for (size_t i = 1, j = 0; i < kFft; ++i) {
    size_t bit = kFft >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { std::swap(re_[i], re_[j]); std::swap(im_[i], im_[j]); }
  }
  for (size_t width = 2; width <= kFft; width <<= 1) {
    for (size_t start = 0; start < kFft; start += width) {
      for (size_t j = 0; j < width / 2; ++j) {
        const size_t t = j * kFft / width;
        const size_t a = start + j, b = a + width / 2;
        const double r = re_[b] * kCos[t] + im_[b] * kSin[t];
        const double v = im_[b] * kCos[t] - re_[b] * kSin[t];
        re_[b] = re_[a] - r; im_[b] = im_[a] - v;
        re_[a] += r; im_[a] += v;
      }
    }
  }
  for (size_t i = 0; i <= kFft / 2; ++i)
    re_[i] = re_[i] * re_[i] + im_[i] * im_[i];
  size_t weight = 0;
  for (size_t m = 0; m < kMels; ++m) {
    float energy = 0;
    for (size_t i = kMelStart[m]; i < kMelEnd[m]; ++i)
      energy += re_[i] * kMelWeights[weight++];
    features_[m][feature_write_] = std::log(energy + 1e-6f);
  }
}
void Frontend::copy_window(float* output) const {
  for (size_t m = 0; m < kMels; ++m)
    for (size_t t = 0; t < kFrames; ++t)
      output[m * kFrames + t] = features_[m][(feature_write_ + t) % kFrames];
}
} // namespace doorbell
