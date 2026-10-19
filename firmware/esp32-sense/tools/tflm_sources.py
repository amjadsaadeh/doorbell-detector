"""Minimal kernel set for the exported doorbell CNN, shared by host and ESP32 builds."""
def source_patterns(optimized=True):
    base = "tensorflow/lite/"
    paths = [base + "micro/*.cc", base + "micro/tflite_bridge/*.cc",
             base + "micro/memory_planner/*.cc", base + "micro/arena_allocator/*.cc",
             base + "core/c/common.cc", base + "core/api/*.cc",
             base + "kernels/kernel_util.cc", base + "kernels/internal/*.cc",
             base + "kernels/internal/reference/portable_tensor_utils.cc",
             base + "kernels/internal/reference/comparisons.cc", base + "schema/schema_utils.cc"]
    kernels = ["add_common", "conv_common", "depthwise_conv_common", "fully_connected_common",
               "mul_common", "pooling_common", "kernel_util", "micro_tensor_utils", "micro_utils",
               "logistic", "logistic_common", "reduce", "reduce_common"]
    paths += [base + "micro/kernels/" + name + ".cc" for name in kernels]
    paths += [base + "micro/kernels/" + ("esp_nn/" if optimized else "") + name + ".cc"
              for name in ["add", "conv", "depthwise_conv", "fully_connected", "mul", "pooling"]]
    return paths
