#pragma once
#include <cstddef>
#include <cstdint>

namespace doorbell {
constexpr size_t kFft = 512, kHop = 320, kMels = 40, kFrames = 100;
// Stateful centered STFT. The first frame is centered on sample zero with
// 256 leading zeros. A frame is emitted only after its right context arrives.
class Frontend {
 public:
  void reset();
  bool push(int16_t sample); // true for each new frame
  bool ready() const { return frames_ >= kFrames; }
  void copy_window(float* mel_major) const;
 private:
  void frame();
  float pcm_[kFft] = {};
  float features_[kMels][kFrames] = {};
  double re_[kFft] = {}, im_[kFft] = {};
  size_t write_ = 0, until_frame_ = kFft / 2, frames_ = 0, feature_write_ = 0;
};

class DetectionGate {
 public:
  DetectionGate(float threshold, uint32_t cooldown_ms)
      : threshold_(threshold), cooldown_(cooldown_ms) {}
  bool update(float score, uint32_t now) {
    if (!(score > threshold_ && score <= 1.0f)) return false;
    if (fired_ && uint32_t(now - last_) < cooldown_) return false;
    last_ = now;
    fired_ = true;
    return true;
  }
 private:
  float threshold_;
  uint32_t cooldown_, last_ = 0;
  bool fired_ = false;
};
} // namespace doorbell
