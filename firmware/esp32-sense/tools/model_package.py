"""Validate a prepared model package without importing the training stack."""
import hashlib
import json
from pathlib import Path


def verify_package(firmware: Path):
    manifest_path = firmware / "lib/doorbell_model/model_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("Run PYTHONPATH=src uv run python src/prepare_esp32_model.py from the repository root first")
    manifest = json.loads(manifest_path.read_text())
    hashes = manifest.get("source_sha256")
    if not hashes:
        raise RuntimeError("Old model package; run src/prepare_esp32_model.py again")
    for name, expected in hashes.items():
        source = firmware.parents[1] / name
        if not source.exists() or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Stale model package: {name} changed; run src/prepare_esp32_model.py again")
    return manifest
