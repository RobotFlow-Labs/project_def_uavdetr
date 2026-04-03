#!/usr/bin/env bash
# Download datasets for this module
# Checks shared volume first, only downloads what's missing
set -euo pipefail

SHARED="${DATASET_VOLUME:-/Volumes/AIFlowDev/RobotFlowLabs/datasets}"
LOCAL="./data"

echo "=== Downloading datasets for $(basename $(dirname $(dirname $0))) ==="
echo "Shared volume: ${SHARED}"
echo "Local data dir: ${LOCAL}"

mkdir -p "${LOCAL}"

# TODO: Add paper-specific download commands here
echo "⚠️  No datasets configured yet. Read the paper and fill this in."
