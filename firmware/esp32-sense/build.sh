#!/usr/bin/env bash
# Wrapper so secrets never need to be typed on the command line or hardcoded:
#   ./build.sh run                 # compile
#   ./build.sh run -t upload       # flash
#   ./build.sh device monitor      # serial monitor
#   ./build.sh test -e native      # run the sigv4 unit tests (no hardware needed)
set -euo pipefail
cd "$(dirname "$0")"

if [ -f .env.esp32 ]; then
  set -a
  source .env.esp32
  set +a
else
  echo "warning: .env.esp32 not found — copy .env.esp32.example and fill it in" >&2
fi

# The image's version is its commit, so an OTA'd board traces to one.
export ESP32_FW_VERSION="${ESP32_FW_VERSION:-$(git describe --always --dirty)}"

exec pio "$@"
