#!/usr/bin/env bash
set -euo pipefail

# Optional expansion helper. This script is intentionally separate from the
# pilot downloader and should be run only after pilot validation passes.

EXTERNAL_DRIVE="${EXTERNAL_DRIVE:-/mnt/d}"
DATA_ROOT="${DATA_ROOT:-$EXTERNAL_DRIVE/Capstone/vla_datasets}"
HURON_ROOT="${HURON_ROOT:-$DATA_ROOT/huron}"
BASE_URL="https://rail.eecs.berkeley.edu/datasets/huron"

declare -a DIRECTORIES=(
  "Feb-15-2023-cory1"
  "Feb-16-2023-cory1-intloss"
  "Feb-17-2023-soda3"
  "Feb-23-2023-soda3-intloss"
)

mkdir -p "$HURON_ROOT/raw" "$HURON_ROOT/logs"
touch "$HURON_ROOT/.write_test"
rm "$HURON_ROOT/.write_test"

for dir in "${DIRECTORIES[@]}"; do
  echo "Preparing optional HuRoN subset folder: $dir"
  mkdir -p "$HURON_ROOT/raw/$dir"
  wget -r -np -nH --cut-dirs=2 \
    --reject="index.html*" \
    --continue \
    --directory-prefix="$HURON_ROOT/raw" \
    "$BASE_URL/$dir/" 2>&1 | tee "$HURON_ROOT/logs/${dir}_subset_download.log"
done
