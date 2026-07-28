#!/usr/bin/env bash
#
# Browse the dataset by prediction outcome in Renumics Spotlight.
#
#   ./src/inspect_dataset.sh                        # held-out predictions only
#   ./src/inspect_dataset.sh --embeddings           # + CNN similarity map
#   ./src/inspect_dataset.sh --all-chunks --refresh
#
# Any argument is forwarded to `inspect_dataset.py prepare`; the viewer is
# configured through SPOTLIGHT_HOST / SPOTLIGHT_PORT.
#
# Two steps because they do different things, not because they need different
# environments: prepare cuts one wav per chunk (slow, cached), show serves
# them. Run either on its own with `uv run python src/inspect_dataset.py
# prepare|show`.
set -euo pipefail

cd "$(dirname "$0")/.."

SPOTLIGHT_HOST="${SPOTLIGHT_HOST:-127.0.0.1}"
SPOTLIGHT_PORT="${SPOTLIGHT_PORT:-auto}"

uv run python src/inspect_dataset.py prepare "$@"
uv run python src/inspect_dataset.py show \
  --host "${SPOTLIGHT_HOST}" --port "${SPOTLIGHT_PORT}"
