#!/usr/bin/env bash
# Build the firmware and ship it to the board over WiFi:
#   ./ota.sh                 # refuses a dirty tree: every image traces to a commit
#   ./ota.sh --allow-dirty   # for trying out uncommitted changes
#   ./ota.sh --timeout 600   # seconds to wait for the board to report back
# Uploads the image to MinIO, asks the board over MQTT to fetch it, and exits 0
# only once the board reports it is running the new version.
set -euo pipefail
cd "$(dirname "$0")"

allow_dirty=0
publish_args=()
for arg in "$@"; do
  case "$arg" in
    --allow-dirty) allow_dirty=1 ;;
    *) publish_args+=("$arg") ;;
  esac
done

export ESP32_FW_VERSION="${ESP32_FW_VERSION:-$(git describe --always --dirty)}"
if [[ "$ESP32_FW_VERSION" == *-dirty && $allow_dirty -eq 0 ]]; then
  echo "error: working tree is dirty ($ESP32_FW_VERSION); commit, or pass --allow-dirty" >&2
  exit 1
fi

# Never show the raw build log: its -D flags carry the .env.esp32 credentials.
echo "building $ESP32_FW_VERSION ..."
if ! log=$(./build.sh run -e seeed_xiao_esp32s3 2>&1); then
  grep -v -e '-D' <<<"$log" | grep -E -i 'error' >&2 || true
  echo "error: build failed" >&2
  exit 1
fi

# Same MinIO and MQTT credentials the board itself uses.
set -a
source .env.esp32
set +a

# --no-project: the repo's own environment is the TensorFlow pipeline, which
# this script doesn't need.
exec uv run --no-project --with boto3 --with paho-mqtt python tools/ota_publish.py \
  .pio/build/seeed_xiao_esp32s3/firmware.bin "${publish_args[@]}"
