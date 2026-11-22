#include <unity.h>
#include <cmath>
#include <limits>
#include "frontend.h"
void setUp() {}
void tearDown() {}
void threshold_cooldown_and_wrap() {
  doorbell::DetectionGate gate(0.5f, 10000);
  TEST_ASSERT_FALSE(gate.update(0.5f, 0));
  TEST_ASSERT_FALSE(gate.update(NAN, 0));
  TEST_ASSERT_FALSE(gate.update(INFINITY, 0));
  TEST_ASSERT_TRUE(gate.update(0.6f, UINT32_MAX - 4999));
  TEST_ASSERT_FALSE(gate.update(0.9f, 4999));
  TEST_ASSERT_TRUE(gate.update(0.9f, 5000));
}
void warmup_and_reset() {
  doorbell::Frontend frontend;
  for (size_t i = 0; i < 256 + 99 * 320 - 1; ++i) frontend.push(123);
  TEST_ASSERT_FALSE(frontend.ready());
  TEST_ASSERT_TRUE(frontend.push(123));
  TEST_ASSERT_TRUE(frontend.ready());
  frontend.reset();
  TEST_ASSERT_FALSE(frontend.ready());
  for (size_t i = 0; i < 256 + 99 * 320; ++i) frontend.push(0);
  float window[4000];
  frontend.copy_window(window);
  for (float value : window) TEST_ASSERT_FLOAT_WITHIN(1e-5f, std::log(1e-6f), value);
}
int main() {
  UNITY_BEGIN();
  RUN_TEST(threshold_cooldown_and_wrap);
  RUN_TEST(warmup_and_reset);
  return UNITY_END();
}
