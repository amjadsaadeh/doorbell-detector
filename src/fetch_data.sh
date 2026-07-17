#! /bin/bash

# Get the git root directory
GIT_ROOT=$(git rev-parse --show-toplevel)

curl -X GET "${LABEL_STUDIO_URL}/api/projects/1/export?exportType=CSV" -H "Authorization: Token ${LABEL_STUDIO_API_KEY}" --output "${GIT_ROOT}/data/labeled_data.csv"
