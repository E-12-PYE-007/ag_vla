#!/bin/bash

# LoRA capacity sweep: rank 32 / 16 / 8, plus DoRA at rank 8. Learning rate (1e-4) and fp32 are
# settled by the earlier sweep and held fixed, so only adapter capacity varies.
#
# Run with --body-frame so rank32 reproduces the existing body-frame lr 1e-4 run and can be compared
# against it. Drop the flag only if you are comparing against a camera-frame run instead.
#
# The job name shown in the queue
#SBATCH --job-name=ag-vla-h100-rank

# GPU selection
#SBATCH -p gpu-h100

# Number of GPUs
#SBATCH --gres=gpu:4

# One SLURM task; finetune_test_rank.py launches a torchrun job per arm
#SBATCH --ntasks=1

# Number of CPU cores (4 dataloader workers per rank x 4 ranks)
#SBATCH --cpus-per-task=16

# RAM
#SBATCH --mem=128G

# Wall clock time-limit. Three LoRA arms at ~2.5-3h each plus a DoRA arm that is 20-30% slower
# (it recomputes a column norm over the full frozen weight every forward pass) is ~11-12h total.
# 24h leaves a wide margin.
#SBATCH --time=1-00:00:00

# stdout output
#SBATCH -o slurm-%j.out

# stderr output
#SBATCH -e slurm-%j.err

#SBATCH --mail-user=lyamamoto@student.unimelb.edu.au

#SBATCH --mail-type=ALL


module purge
module load GCCcore/11.3.0 Python/3.10.4
module load CUDA/12.2.0

export CUDA_HOME=$(dirname "$(dirname "$(which nvcc)")")
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

PROJECT_ID=capstone
PROJECT_DIR=/data/gpfs/projects/capstone/E12PYE007

export PROJECT_ID
export PROJECT_DIR
export HF_HOME=${PROJECT_DIR}/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export WANDB_DIR=${PROJECT_DIR}/wandb

source ${PROJECT_DIR}/venvs/training/finetune/bin/activate

cd $SLURM_SUBMIT_DIR

# Plain python, not torchrun: this script launches one torchrun job per arm, back to back.
python training/finetune/finetune_test_rank.py \
    --data-dir ${PROJECT_DIR}/ag_vla/pt_body_fixed \
    --out-root ${PROJECT_DIR}/out \
    --body-frame

my-job-stats -v -d
