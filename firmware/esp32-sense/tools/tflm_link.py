"""Compile Espressif libraries after PlatformIO configures the framework."""
from pathlib import Path
Import("env")
tf = Path(env["DOORBELL_TFLM"])
nn = Path(env["DOORBELL_ESP_NN"])
# The application and library must agree on TFLM's static-memory ABI.
libenv = env.Clone()
libenv.Append(CCFLAGS=["-O2", "-Wno-maybe-uninitialized"])
libenv.Append(CXXFLAGS=["-std=gnu++17", "-fno-rtti", "-fno-exceptions"])
import sys
sys.path.insert(0, str(Path(env.subst("$PROJECT_DIR")) / "tools"))
from tflm_sources import source_patterns
paths = source_patterns()
tf_lib = libenv.BuildLibrary("$BUILD_DIR/tflm", str(tf), src_filter=["-<*>"] + ["+<" + p + ">" for p in paths])
nn_lib = libenv.BuildLibrary("$BUILD_DIR/esp_nn", str(nn), src_filter=["-<*>", "+<src/**/*.c>", "+<src/**/*esp32s3.S>", "-<src/*/*esp32p4.c>"])
env.Append(LIBS=[tf_lib, nn_lib])
