"""Build pinned upstream Espressif sources in the existing Arduino toolchain.

The source selection mirrors esp-tflite-micro v1.3.1 CMakeLists.txt, with only
our model's kernels. Downloads are cached in .pio; no generated code is vendored.
"""
from pathlib import Path
import subprocess
import sys

Import("env")

root = Path(env.subst("$PROJECT_DIR"))
sys.path.insert(0, str(root / "tools"))
from model_package import verify_package
verify_package(root)


def checkout(name, revision):
    path = root / ".pio" / "upstream" / name
    if not (path / ".git").exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--quiet", "https://github.com/espressif/" + name + ".git", str(path)], check=True)
    actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if actual != revision:
        subprocess.run(["git", "-C", str(path), "checkout", "--quiet", revision], check=True)
    return path


tf = checkout("esp-tflite-micro", "0032f1734e1e7b5fcf1b588cf1abb9ef53fed4c3")
nn = checkout("esp-nn", "596b08401a63da3a2e1b40868c442f582a99ae26")
includes = [tf, tf / "third_party/gemmlowp", tf / "third_party/flatbuffers/include",
            tf / "third_party/ruy", nn / "include", nn / "src/common"]
env.Append(CPPPATH=[str(p) for p in includes], CPPDEFINES=["TF_LITE_STATIC_MEMORY", "ESP_NN", "CONFIG_NN_OPTIMIZED"])
env["DOORBELL_TFLM"] = str(tf)
env["DOORBELL_ESP_NN"] = str(nn)
