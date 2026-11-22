"""Compile the pinned TFLM reference kernels and check the prepared model on host.

Run the PlatformIO firmware build first to fetch pinned upstream sources.
This complements, but does not replace, the ESP-NN self-test on device.
"""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
tf = ROOT / ".pio/upstream/esp-tflite-micro"
if not tf.is_dir():
    raise SystemExit("Build the firmware first to fetch pinned TFLM sources")
# Reuse the build source lists, substituting reference kernels for ESP-NN.
from tflm_sources import source_patterns
sources = sorted({p for pattern in source_patterns(optimized=False) for p in tf.glob(pattern)})
build = ROOT / ".pio/host_model"
build.mkdir(parents=True, exist_ok=True)
flags = ["c++", "-std=c++17", "-O2", "-DTF_LITE_STATIC_MEMORY", "-fno-exceptions", "-fno-rtti", "-ffunction-sections", "-fdata-sections"]
for path in [tf, tf / "third_party/gemmlowp", tf / "third_party/flatbuffers/include",
             tf / "third_party/ruy", ROOT / "lib/doorbell_model"]:
    flags.extend(["-I", str(path)])
# Compile in parallel, without repeatedly recompiling unchanged sources.
from concurrent.futures import ThreadPoolExecutor

header_time = max(p.stat().st_mtime for root in [tf, ROOT / "lib/doorbell_model"] for p in root.rglob("*.h"))

def compile_source(path):
    import hashlib
    obj = build / (hashlib.sha256((str(path) + repr(flags)).encode()).hexdigest()[:16] + ".o")
    if not obj.exists() or obj.stat().st_mtime < max(path.stat().st_mtime, header_time):
        subprocess.run(flags + ["-c", str(path), "-o", str(obj)], check=True)
    return obj

sources += [ROOT / "test/host/model_runner.cpp", ROOT / "lib/doorbell_model/doorbell_model_data.cc"]
with ThreadPoolExecutor(max_workers=4) as pool:
    objects = list(pool.map(compile_source, sources))
exe = build / "check_model"
subprocess.run(["c++", "-Wl,--gc-sections", *map(str, objects), "-o", str(exe)], check=True)
subprocess.run([str(exe)], check=True)
