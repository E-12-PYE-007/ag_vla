#!/usr/bin/env bash
set -euo pipefail

EXTERNAL_DRIVE="${EXTERNAL_DRIVE:-/mnt/d}"
DATA_ROOT="${DATA_ROOT:-$EXTERNAL_DRIVE/Capstone/vla_datasets}"
HURON_ROOT="${HURON_ROOT:-$DATA_ROOT/huron}"
SEQUENCE="Feb-16-2023-cory1-intloss"
FILENAME="00000001.bag"
URL="https://rail.eecs.berkeley.edu/datasets/huron/$SEQUENCE/$FILENAME"
DEST_DIR="$HURON_ROOT/raw/$SEQUENCE"
DEST="$DEST_DIR/$FILENAME"
PART="$DEST.part"

mkdir -p "$DEST_DIR" "$HURON_ROOT/manifests" "$HURON_ROOT/logs"
touch "$HURON_ROOT/.write_test"
rm "$HURON_ROOT/.write_test"

echo "Source URL: $URL"
echo "Destination: $DEST"

if [[ ! -s "$DEST" ]]; then
  curl -L -C - -o "$PART" "$URL" 2>&1 | tee "$HURON_ROOT/logs/${SEQUENCE}_${FILENAME}_download.log"
  sniff="$(head -c 256 "$PART" | strings | tr '[:upper:]' '[:lower:]' || true)"
  if printf "%s" "$sniff" | grep -Eq "<html|<!doctype html|not found|access denied"; then
    echo "ERROR: downloaded content looks like HTML/error page." >&2
    exit 1
  fi
  mv "$PART" "$DEST"
fi

file "$DEST"
du -h "$DEST"
sha256sum "$DEST" | tee -a "$HURON_ROOT/manifests/huron-pilot-sha256.txt"
