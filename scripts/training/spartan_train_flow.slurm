#!/bin/bash
#SBATCH --job-name=agvla_flow
#SBATCH -p gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/gpfs/projects/${USER}/ag_vla}"
REPO_ROOT="${REPO_ROOT:-${PROJECT_ROOT}/repo}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/processed_mixed}"
SPLIT_JSON="${SPLIT_JSON:-${DATA_ROOT}/splits/train_val_test.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/flow_head_raw_projector}"
CONDA_ENV="${CONDA_ENV:-asyncvla}"
CONDA_MODULE="${CONDA_MODULE:-Anaconda3}"

mkdir -p "${REPO_ROOT}/logs" "${OUTPUT_DIR}"
cd "${REPO_ROOT}"

module purge
module load "${CONDA_MODULE}"
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
set -u

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_JOB_NODELIST}"
echo "Data: ${DATA_ROOT}"
echo "Split: ${SPLIT_JSON}"
echo "Output: ${OUTPUT_DIR}"
nvidia-smi || true

stdbuf -o0 -e0 python scripts/training/train_flow_head.py \
  --data "${DATA_ROOT}" \
  --split-json "${SPLIT_JSON}" \
  --output-dir "${OUTPUT_DIR}" \
  --use-asyncvla-projector \
  --train-projector \
  --default-modality-id 7.0 \
  --epochs "${EPOCHS:-100}" \
  --batch-size "${BATCH_SIZE:-32}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --lr "${LR:-1e-4}" \
  --weight-decay "${WEIGHT_DECAY:-1e-4}" \
  --cache-trajectories

my-job-stats -a -n -s || true
