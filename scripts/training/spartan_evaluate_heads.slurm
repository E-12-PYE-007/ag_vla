#!/bin/bash
#SBATCH --job-name=agvla_eval
#SBATCH -p gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/gpfs/projects/${USER}/ag_vla}"
REPO_ROOT="${REPO_ROOT:-${PROJECT_ROOT}/repo}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/processed_mixed}"
SPLIT_JSON="${SPLIT_JSON:-${DATA_ROOT}/splits/train_val_test.json}"
MLP_CKPT="${MLP_CKPT:-${PROJECT_ROOT}/checkpoints/mlp_head_raw_projector/best.pt}"
FLOW_CKPT="${FLOW_CKPT:-${PROJECT_ROOT}/checkpoints/flow_head_raw_projector/best.pt}"
EVAL_ROOT="${EVAL_ROOT:-${PROJECT_ROOT}/eval}"
CONDA_ENV="${CONDA_ENV:-asyncvla}"
CONDA_MODULE="${CONDA_MODULE:-Anaconda3}"

mkdir -p "${REPO_ROOT}/logs" "${EVAL_ROOT}"
cd "${REPO_ROOT}"

module purge
module load "${CONDA_MODULE}"
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
set -u

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

stdbuf -o0 -e0 python scripts/training/evaluate_mlp_head.py \
  --checkpoint "${MLP_CKPT}" \
  --data "${DATA_ROOT}" \
  --split-json "${SPLIT_JSON}" \
  --split test \
  --output-dir "${EVAL_ROOT}/mlp_head_test"

stdbuf -o0 -e0 python scripts/training/evaluate_flow_head.py \
  --checkpoint "${FLOW_CKPT}" \
  --data "${DATA_ROOT}" \
  --split-json "${SPLIT_JSON}" \
  --split test \
  --num-steps "${FLOW_SAMPLE_STEPS:-20}" \
  --output-dir "${EVAL_ROOT}/flow_head_test"

my-job-stats -a -n -s || true
