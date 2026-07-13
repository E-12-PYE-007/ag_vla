#!/usr/bin/env bash
set -euo pipefail

EXTERNAL_DRIVE="${EXTERNAL_DRIVE:-/mnt/d}"
DATA_ROOT="${DATA_ROOT:-$EXTERNAL_DRIVE/Capstone/vla_datasets}"
BOTANIC_ROOT="${BOTANIC_ROOT:-$DATA_ROOT/botanic_garden}"
MIN_FREE_GB="${MIN_FREE_GB:-20}"

REPO_URL="https://github.com/robot-pesg/BotanicGarden.git"
REPO_DIR="$BOTANIC_ROOT/repository/BotanicGarden"

free_gb() {
  df -BG "$1" | awk 'NR==2 {gsub("G", "", $4); print $4}'
}

if [[ ! -d "$EXTERNAL_DRIVE" ]]; then
  echo "ERROR: external drive path does not exist: $EXTERNAL_DRIVE" >&2
  exit 1
fi

case "$EXTERNAL_DRIVE" in
  /mnt/*) ;;
  *)
    echo "ERROR: refusing to prepare data outside /mnt Windows drive mounts: $EXTERNAL_DRIVE" >&2
    exit 1
    ;;
esac

mkdir -p "$BOTANIC_ROOT"
touch "$BOTANIC_ROOT/.write_test"
rm "$BOTANIC_ROOT/.write_test"

available="$(free_gb "$EXTERNAL_DRIVE")"
echo "Selected external drive: $EXTERNAL_DRIVE"
echo "BotanicGarden root: $BOTANIC_ROOT"
echo "Free space: ${available}G"
if (( available < MIN_FREE_GB )); then
  echo "ERROR: need at least ${MIN_FREE_GB}G free for BotanicGarden pilot." >&2
  exit 1
fi

mkdir -p \
  "$BOTANIC_ROOT/repository" \
  "$BOTANIC_ROOT/raw/1018-00" \
  "$BOTANIC_ROOT/calibration" \
  "$BOTANIC_ROOT/ground_truth" \
  "$BOTANIC_ROOT/extracted/1018-00/images" \
  "$BOTANIC_ROOT/extracted/1018-00/metadata" \
  "$BOTANIC_ROOT/extracted/1018-00/previews" \
  "$BOTANIC_ROOT/scripts" \
  "$BOTANIC_ROOT/logs" \
  "$BOTANIC_ROOT/manifests"

if [[ -d "$REPO_DIR/.git" ]]; then
  echo "Updating BotanicGarden repository..."
  git -C "$REPO_DIR" pull --ff-only
else
  echo "Cloning BotanicGarden repository..."
  git clone "$REPO_URL" "$REPO_DIR"
fi

cp -a "$REPO_DIR/calib/." "$BOTANIC_ROOT/calibration/"
cp -a "$REPO_DIR/GT_traj/." "$BOTANIC_ROOT/ground_truth/"

commit="$(git -C "$REPO_DIR" rev-parse HEAD)"
cat > "$BOTANIC_ROOT/manifests/repository_manifest.json" <<JSON
{
  "repository_url": "$REPO_URL",
  "repository_path": "$REPO_DIR",
  "commit": "$commit",
  "calibration_path": "$BOTANIC_ROOT/calibration",
  "ground_truth_path": "$BOTANIC_ROOT/ground_truth"
}
JSON

echo "Repository commit: $commit"
echo "Copied calibration to: $BOTANIC_ROOT/calibration"
echo "Copied GT trajectories to: $BOTANIC_ROOT/ground_truth"
echo "Raw 1018-00 bag destination: $BOTANIC_ROOT/raw/1018-00"
