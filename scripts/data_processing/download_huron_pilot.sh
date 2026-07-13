#!/usr/bin/env bash
set -euo pipefail

# Download the two HuRoN pilot bags to an external drive.
# Default target is the user's D: drive mounted in WSL at /mnt/d.

EXTERNAL_DRIVE="${EXTERNAL_DRIVE:-/mnt/d}"
DATA_ROOT="${DATA_ROOT:-$EXTERNAL_DRIVE/Capstone/vla_datasets}"
HURON_ROOT="${HURON_ROOT:-$DATA_ROOT/huron}"
MIN_FREE_GB="${MIN_FREE_GB:-15}"

SACSON_REPO_URL="https://github.com/NHirose/SACSoN.git"
BASE_URL="https://rail.eecs.berkeley.edu/datasets/huron"

declare -a SEQUENCES=(
  "Feb-15-2023-cory1:false"
  "Feb-16-2023-cory1-intloss:true"
)

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

free_gb() {
  df -BG "$1" | awk 'NR==2 {gsub("G", "", $4); print $4}'
}

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

reject_html() {
  local file="$1"
  local sniff
  sniff="$(head -c 256 "$file" | strings | tr '[:upper:]' '[:lower:]' || true)"
  if printf "%s" "$sniff" | grep -Eq "<html|<!doctype html|access denied|not found"; then
    echo "ERROR: downloaded file looks like HTML/error content: $file" >&2
    return 1
  fi
}

require_external_drive() {
  if [[ ! -d "$EXTERNAL_DRIVE" ]]; then
    echo "ERROR: external drive path does not exist: $EXTERNAL_DRIVE" >&2
    exit 1
  fi
  case "$EXTERNAL_DRIVE" in
    /mnt/*) ;;
    *)
      echo "ERROR: refusing to download outside /mnt Windows drive mounts: $EXTERNAL_DRIVE" >&2
      exit 1
      ;;
  esac
  mkdir -p "$HURON_ROOT"
  touch "$HURON_ROOT/.write_test"
  rm "$HURON_ROOT/.write_test"

  local available
  available="$(free_gb "$EXTERNAL_DRIVE")"
  echo "Selected external drive: $EXTERNAL_DRIVE"
  echo "HuRoN root: $HURON_ROOT"
  echo "Free space before download: ${available}G"
  if (( available < MIN_FREE_GB )); then
    echo "ERROR: need at least ${MIN_FREE_GB}G free for pilot; found ${available}G" >&2
    exit 1
  fi
}

prepare_layout() {
  mkdir -p \
    "$HURON_ROOT/repository" \
    "$HURON_ROOT/raw/Feb-15-2023-cory1" \
    "$HURON_ROOT/raw/Feb-16-2023-cory1-intloss" \
    "$HURON_ROOT/calibration" \
    "$HURON_ROOT/extracted/Feb-15-2023-cory1/previews" \
    "$HURON_ROOT/extracted/Feb-15-2023-cory1/images" \
    "$HURON_ROOT/extracted/Feb-15-2023-cory1/metadata" \
    "$HURON_ROOT/extracted/Feb-16-2023-cory1-intloss/previews" \
    "$HURON_ROOT/extracted/Feb-16-2023-cory1-intloss/images" \
    "$HURON_ROOT/extracted/Feb-16-2023-cory1-intloss/metadata" \
    "$HURON_ROOT/scripts" \
    "$HURON_ROOT/logs" \
    "$HURON_ROOT/manifests"
}

clone_or_update_repo() {
  local repo="$HURON_ROOT/repository/SACSoN"
  if [[ -d "$repo/.git" ]]; then
    echo "Updating SACSoN repository..."
    git -C "$repo" pull --ff-only
  else
    echo "Cloning SACSoN repository..."
    git clone "$SACSON_REPO_URL" "$repo"
  fi
}

download_one() {
  local sequence="$1"
  local intloss="$2"
  local url="$BASE_URL/$sequence/00000000.bag"
  local dir="$HURON_ROOT/raw/$sequence"
  local final="$dir/00000000.bag"
  local part="$final.part"
  local log="$HURON_ROOT/logs/${sequence}_00000000_download.log"

  mkdir -p "$dir"
  echo ""
  echo "Source URL: $url"
  echo "Destination: $final"

  if [[ -s "$final" ]]; then
    reject_html "$final"
    echo "Already present; keeping existing file: $final"
  else
    echo "Downloading with resume support..."
    if command -v aria2c >/dev/null 2>&1; then
      aria2c --continue=true --max-connection-per-server=8 --split=8 --min-split-size=20M \
        --dir="$dir" --out="00000000.bag.part" "$url" 2>&1 | tee "$log"
    else
      curl -L -C - -o "$part" "$url" 2>&1 | tee "$log"
    fi
    reject_html "$part"
    mv "$part" "$final"
  fi

  echo "File type:"
  file "$final"
  echo "File size:"
  du -h "$final"

  local size sha
  size="$(stat -c%s "$final")"
  sha="$(sha256_file "$final")"
  printf "%s  %s\n" "$sha" "$final" >> "$HURON_ROOT/manifests/huron-pilot-sha256.txt"

  cat > "$HURON_ROOT/manifests/${sequence}_00000000_manifest.json" <<JSON
{
  "source_url": "$url",
  "local_path": "$final",
  "size_bytes": $size,
  "download_date_utc": "$(timestamp)",
  "sha256": "$sha",
  "environment": "$sequence",
  "interaction_loss": $intloss,
  "sacson_repository_commit": "$(git -C "$HURON_ROOT/repository/SACSoN" rev-parse HEAD 2>/dev/null || echo unknown)"
}
JSON
}

write_combined_manifest() {
  python3 - "$HURON_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
items = []
for path in sorted((root / "manifests").glob("*_00000000_manifest.json")):
    items.append(json.loads(path.read_text()))
out = {
    "dataset": "huron",
    "created_utc": __import__("datetime").datetime.utcnow().isoformat(timespec="seconds") + "Z",
    "items": items,
}
(root / "manifests" / "huron-pilot-download-manifest.json").write_text(json.dumps(out, indent=2))
PY
}

main() {
  require_external_drive
  prepare_layout
  clone_or_update_repo
  : > "$HURON_ROOT/manifests/huron-pilot-sha256.txt"

  for item in "${SEQUENCES[@]}"; do
    sequence="${item%%:*}"
    intloss="${item##*:}"
    download_one "$sequence" "$intloss"
  done

  write_combined_manifest
  echo ""
  echo "Free space after download: $(free_gb "$EXTERNAL_DRIVE")G"
  echo "Done. Manifest: $HURON_ROOT/manifests/huron-pilot-download-manifest.json"
}

main "$@"
