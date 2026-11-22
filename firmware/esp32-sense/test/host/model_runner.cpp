#include <cstdio>
#include <cstring>
#include <cstdlib>
#include "doorbell_model_data.h"
#include "model_self_test.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
int main() {
  alignas(16) static unsigned char arena[240 * 1024];
  tflite::MicroMutableOpResolver<8> resolver;
  resolver.AddMul(); resolver.AddAdd(); resolver.AddConv2D(); resolver.AddMaxPool2D();
  resolver.AddDepthwiseConv2D(); resolver.AddMean(); resolver.AddFullyConnected(); resolver.AddLogistic();
  tflite::MicroInterpreter interpreter(tflite::GetModel(g_doorbell_model_data), resolver, arena, sizeof(arena));
  if (interpreter.AllocateTensors() != kTfLiteOk) return 1;
  for (size_t i = 0; i < 2; ++i) {
    std::memcpy(interpreter.input(0)->data.int8, kModelTestInputs[i], 4000);
    if (interpreter.Invoke() != kTfLiteOk) return 2;
    int actual = interpreter.output(0)->data.int8[0];
    std::printf("vector %zu: expected %d, TFLM %d\n", i, kModelTestOutputs[i], actual);
    if (std::abs(actual - kModelTestOutputs[i]) > 1) return 3;
  }
  std::printf("arena used: %zu bytes\n", interpreter.arena_used_bytes());
}
