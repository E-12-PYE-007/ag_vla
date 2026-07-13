#!/bin/bash
#SBATCH --job-name=agvla_split
#SBATCH -p sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0-00:30:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/gpfs/projects/${USER}/ag_vla}"
REPO_ROOT="${REPO_ROOT:-${PROJECT_ROOT}/repo}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/processed_mixed}"
CONDA_ENV="${CONDA_ENV:-asyncvla}"
CONDA_MODULE="${CONDA_MODULE:-Anaconda3}"

mkdir -p "${REPO_ROOT}/logs" "${DATA_ROOT}/splits"
cd "${REPO_ROOT}"

module purge
module load "${CONDA_MODULE}"
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
set -u

stdbuf -o0 -e0 python scripts/training/build_train_val_test_split.py \
  --data-root "${DATA_ROOT}" \
  --out "${DATA_ROOT}/splits/train_val_test.json" \
  --train-fraction 0.70 \
  --val-fraction 0.15 \
  --test-fraction 0.15 \
  --seed 7

my-job-stats -a -n -s || true
