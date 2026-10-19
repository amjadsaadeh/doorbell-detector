#include "frontend.h"
#include <cstdio>
int main() {
  doorbell::Frontend frontend;
  int16_t sample;
  float window[doorbell::kMels * doorbell::kFrames];
  size_t frames = 0;
  while (std::fread(&sample, sizeof(sample), 1, stdin) == 1) {
    if (frontend.push(sample)) {
      ++frames;
      if (frontend.ready() && (frames - doorbell::kFrames) % 25 == 0) {
        frontend.copy_window(window);
        std::fwrite(window, sizeof(float), doorbell::kMels * doorbell::kFrames, stdout);
      }
    }
  }
}
