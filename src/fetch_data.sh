#! /bin/bash
set -euo pipefail

# Get the git root directory
GIT_ROOT=$(git rev-parse --show-toplevel)

# Label Studio personal access tokens are JWT refresh tokens that must be
# exchanged for a short-lived access token before calling the API
ACCESS_TOKEN=$(curl -sf -X POST "${LABEL_STUDIO_URL}/api/token/refresh" \
    -H "Content-Type: application/json" \
    -d "{\"refresh\": \"${LABEL_STUDIO_API_KEY}\"}" \
    | sed -E 's/.*"access":"([^"]+)".*/\1/')

curl -sf "${LABEL_STUDIO_URL}/api/projects/1/export?exportType=CSV" \
    -H "Authorization: Bearer ${ACCESS_TOKEN}" \
    --output "${GIT_ROOT}/data/labeled_data.csv"
