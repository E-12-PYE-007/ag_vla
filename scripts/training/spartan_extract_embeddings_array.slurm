#!/bin/bash
#SBATCH --job-name=agvla_extract
#SBATCH -p gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err

set -euo pipefail

# Submit with:
#   sbatch --array=0-N%1 scripts/training/spartan_extract_embeddings_array.slurm
#
# Required environment variables can be overridden at submission time:
#   sbatch --export=ALL,PROJECT_ROOT=/data/gpfs/projects/<project>/ag_vla,... job.slurm

PROJECT_ROOT="${PROJECT_ROOT:-/data/gpfs/projects/${USER}/ag_vla}"
REPO_ROOT="${REPO_ROOT:-${PROJECT_ROOT}/repo}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/processed_mixed}"
ASYNCVLA_ROOT="${ASYNCVLA_ROOT:-${PROJECT_ROOT}/AsyncVLA}"
VLA_PATH="${VLA_PATH:-${ASYNCVLA_ROOT}/omnivla-original-balance}"
MANIFEST="${MANIFEST:-${PROJECT_ROOT}/manifests/embedding_manifest.txt}"
CONDA_ENV="${CONDA_ENV:-asyncvla}"
CONDA_MODULE="${CONDA_MODULE:-Anaconda3}"
PROMPT="${PROMPT:-navigate forward safely}"
OUTPUT_NAME="${OUTPUT_NAME:-trajectory_with_embeddings.npz}"
BATCH_SIZE="${BATCH_SIZE:-1}"

mkdir -p "${REPO_ROOT}/logs" "${PROJECT_ROOT}/manifests"
cd "${REPO_ROOT}"

module purge
module load "${CONDA_MODULE}"
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
set -u

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "ERROR: submit this as a Slurm array job with --array=0-N%1" >&2
  exit 1
fi

NPZ="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${MANIFEST}")"
if [[ -z "${NPZ}" ]]; then
  echo "ERROR: no manifest entry for task ${SLURM_ARRAY_TASK_ID}" >&2
  exit 1
fi

echo "Job ID: ${SLURM_JOB_ID}"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURM_JOB_NODELIST}"
echo "Trajectory: ${NPZ}"
echo "Prompt: ${PROMPT}"
nvidia-smi || true

stdbuf -o0 -e0 python scripts/embeddings/extract_vla_embeddings.py \
  --npz "${NPZ}" \
  --asyncvla-root "${ASYNCVLA_ROOT}" \
  --vla-path "${VLA_PATH}" \
  --raw-only \
  --prompt "${PROMPT}" \
  --output-name "${OUTPUT_NAME}" \
  --batch-size "${BATCH_SIZE}"

my-job-stats -a -n -s || true
