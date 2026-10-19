#include "model_detector.h"
#include <Arduino.h>
#include <esp_heap_caps.h>
#include <cmath>
#include <cstring>
#include <new>
#include "frontend.h"
#include "doorbell_model_data.h"
#include "model_self_test.h"
#include "audio_capture.h"
#include "mqtt_client.h"
#include "config.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace model_detector {
namespace {
struct Packet { uint32_t epoch; uint16_t count; int16_t pcm[kPdmReadChunkBytes / 2]; };
QueueHandle_t queue = nullptr;
uint32_t epoch = 0; // capture task only
constexpr size_t kArenaBytes = 240 * 1024;
constexpr size_t kQueueLength = 32;
tflite::MicroInterpreter* interpreter = nullptr;
doorbell::Frontend* frontend = nullptr;
float* window = nullptr;

void task(void*) {
  Packet packet;
  uint32_t current_epoch = 0;
  size_t until_inference = doorbell::kFrames;
  doorbell::DetectionGate gate(kDetectionThreshold, kDetectionCooldownMs);
  for (;;) {
    if (xQueueReceive(queue, &packet, portMAX_DELAY) != pdTRUE) continue;
    if (packet.epoch != current_epoch) {
      current_epoch = packet.epoch;
      frontend->reset();
      until_inference = doorbell::kFrames;
      Serial.println("model: audio gap; warming up again");
    }
    for (size_t s = 0; s < packet.count; ++s) {
      if (!frontend->push(packet.pcm[s])) continue;
      if (--until_inference) continue;
      until_inference = 25;
      const uint32_t started = micros();
      frontend->copy_window(window);
      TfLiteTensor* input = interpreter->input(0);
      for (size_t i = 0; i < doorbell::kMels * doorbell::kFrames; ++i) {
        const float q = std::nearbyint(window[i] / input->params.scale + input->params.zero_point);
        input->data.int8[i] = static_cast<int8_t>(fmaxf(-128, fminf(127, q)));
      }
      if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("model: Invoke failed");
        continue;
      }
      const TfLiteTensor* output = interpreter->output(0);
      const float score = (int(output->data.int8[0]) - output->params.zero_point) * output->params.scale;
      Serial.printf("model: score %.4f, inference %lu us\n", score,
                    static_cast<unsigned long>(micros() - started));
      if (gate.update(score, millis())) {
        mqtt_client::notify_detection();
        audio_capture::request_recording(true);
      }
    }
  }
}
} // namespace

bool begin() {
  const tflite::Model* model = tflite::GetModel(g_doorbell_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) return false;
  static tflite::MicroMutableOpResolver<9> resolver;
  if (resolver.AddMul() != kTfLiteOk || resolver.AddAdd() != kTfLiteOk ||
      resolver.AddConv2D() != kTfLiteOk || resolver.AddMaxPool2D() != kTfLiteOk ||
      resolver.AddDepthwiseConv2D() != kTfLiteOk || resolver.AddMean() != kTfLiteOk ||
      resolver.AddFullyConnected() != kTfLiteOk || resolver.AddLogistic() != kTfLiteOk)
    return false;
  // Leave room for WiFi, network stacks, and OTA rather than consuming
  // the last available internal block before networking starts.
  uint8_t* arena = nullptr;
  if (heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT) >= kArenaBytes + 80 * 1024)
    arena = static_cast<uint8_t*>(heap_caps_malloc(kArenaBytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
  if (!arena) {
    Serial.println("model: internal arena unavailable; using PSRAM (benchmark latency)");
    arena = static_cast<uint8_t*>(heap_caps_malloc(kArenaBytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  }
  if (!arena) return false;
  static tflite::MicroInterpreter instance(model, resolver, arena, kArenaBytes);
  interpreter = &instance;
  if (interpreter->AllocateTensors() != kTfLiteOk) return false;
  auto* in = interpreter->input(0);
  auto* out = interpreter->output(0);
  if (in->type != kTfLiteInt8 || in->dims->size != 4 || in->dims->data[0] != 1 ||
      in->dims->data[1] != 40 || in->dims->data[2] != 100 || in->dims->data[3] != 1 ||
      out->type != kTfLiteInt8 || out->bytes != 1 ||
      !(in->params.scale > 0) || !(out->params.scale > 0)) return false;
  for (size_t i = 0; i < 2; ++i) {
    std::memcpy(in->data.int8, kModelTestInputs[i], in->bytes);
    const uint32_t start = micros();
    if (interpreter->Invoke() != kTfLiteOk ||
        std::abs(int(out->data.int8[0]) - kModelTestOutputs[i]) > 1) {
      Serial.println("model: boot self-test FAILED");
      return false;
    }
    Serial.printf("model: self-test %u passed, %lu us\n", unsigned(i),
                  static_cast<unsigned long>(micros() - start));
  }
  void* memory = heap_caps_malloc(sizeof(doorbell::Frontend), MALLOC_CAP_SPIRAM);
  window = static_cast<float*>(heap_caps_malloc(4000 * sizeof(float), MALLOC_CAP_SPIRAM));
  auto* storage = static_cast<uint8_t*>(heap_caps_malloc(kQueueLength * sizeof(Packet), MALLOC_CAP_SPIRAM));
  if (!memory || !window || !storage) return false;
  frontend = new (memory) doorbell::Frontend();
  static StaticQueue_t queue_control;
  queue = xQueueCreateStatic(kQueueLength, sizeof(Packet), storage, &queue_control);
  Serial.printf("model: %u bytes, arena used %u / %u\n", g_doorbell_model_data_len,
                unsigned(interpreter->arena_used_bytes()), unsigned(kArenaBytes));
  return queue && xTaskCreatePinnedToCore(task, "detectTask", 8192, nullptr, 1, nullptr, 1) == pdPASS;
}
void feed(const uint8_t* pcm, size_t bytes) {
  if (!queue) return;
  if (bytes > kPdmReadChunkBytes || bytes % 2) { discontinuity(); return; }
  Packet packet;
  packet.epoch = epoch;
  packet.count = bytes / 2;
  std::memcpy(packet.pcm, pcm, bytes);
  if (xQueueSend(queue, &packet, 0) != pdTRUE) ++epoch;
}
void discontinuity() { ++epoch; }
} // namespace model_detector
