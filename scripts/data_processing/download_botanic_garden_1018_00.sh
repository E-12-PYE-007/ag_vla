#!/usr/bin/env bash
set -euo pipefail

# This script tries to download the official 1018-00 VLIO rosbag if a direct
# URL is supplied. BotanicGarden links are portal/OneDrive-style, so manual
# browser download may be required.

EXTERNAL_DRIVE="${EXTERNAL_DRIVE:-/mnt/d}"
DATA_ROOT="${DATA_ROOT:-$EXTERNAL_DRIVE/Capstone/vla_datasets}"
BOTANIC_ROOT="${BOTANIC_ROOT:-$DATA_ROOT/botanic_garden}"
URL="${BOTANIC_1018_00_VLIO_URL:-}"
OUT_NAME="${BOTANIC_1018_00_FILENAME:-1018-00-vlio.bag}"
DEST_DIR="$BOTANIC_ROOT/raw/1018-00"
DEST="$DEST_DIR/$OUT_NAME"
PART="$DEST.part"

if [[ -z "$URL" ]]; then
  cat <<EOF
No direct URL supplied.

Please manually download the official BotanicGarden 1018-00 VLIO rosbag from:
  https://github.com/robot-pesg/BotanicGarden

Use the table row:
  Sequence: 1018-00
  File type: VLIO-rosbag
  Preferred source: official SJTU Science Data or official OneDrive

Place the downloaded .bag file here:
  $DEST_DIR/

Do not download the LIO-only bag or imagezip for this pilot.
EOF
  exit 2
fi

mkdir -p "$DEST_DIR" "$BOTANIC_ROOT/logs" "$BOTANIC_ROOT/manifests"
touch "$BOTANIC_ROOT/.write_test"
rm "$BOTANIC_ROOT/.write_test"

if [[ -s "$DEST" ]]; then
  echo "Already present: $DEST"
else
  echo "Downloading: $URL"
  echo "Destination: $DEST"
  curl -L -C - -o "$PART" "$URL" 2>&1 | tee "$BOTANIC_ROOT/logs/1018-00-vlio-download.log"
  sniff="$(head -c 256 "$PART" | strings | tr '[:upper:]' '[:lower:]' || true)"
  if printf "%s" "$sniff" | grep -Eq "<html|<!doctype html|sign in|login|not found|access denied"; then
    echo "ERROR: downloaded content looks like HTML/login/error page." >&2
    echo "Leaving partial file at: $PART" >&2
    exit 1
  fi
  mv "$PART" "$DEST"
fi

file "$DEST"
du -h "$DEST"
sha256sum "$DEST" | tee "$BOTANIC_ROOT/manifests/1018-00-sha256.txt"
