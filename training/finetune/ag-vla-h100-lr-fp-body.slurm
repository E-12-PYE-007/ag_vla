#!/bin/bash

# Same three fp32 arms as ag-vla-h100-lr-fp.slurm, but with the waypoints re-expressed in the
# robot's frame of travel (--body-frame) instead of the camera's. The recorded pose is the camera
# prim's and that camera sits ~5.11 deg right of travel, so the stored labels lean left by that much
# once a body-frame controller consumes them. The fix is applied in the collate, so it needs no
# reprocessed data: pt_data is used exactly as uploaded.
#
# Checkpoints and W&B runs are tagged "_body" and are NOT comparable to the camera-frame runs or to
# the bf16 baseline - the labels differ. Compare A_fp32_body vs B/C_fp32_body only.

# The job name shown in the queue
#SBATCH --job-name=ag-vla-h100-lr-fp-body

# GPU selection
#SBATCH -p gpu-h100

# Number of GPUs
#SBATCH --gres=gpu:4

# One SLURM task; finetune_test_lr_fp.py launches a torchrun job per arm
#SBATCH --ntasks=1

# Number of CPU cores (4 dataloader workers per rank x 4 ranks)
#SBATCH --cpus-per-task=16

# RAM
#SBATCH --mem=128G

# Wall clock time-limit. Arms A and B are ~2h each and C ~4h (it processes twice the samples per
# update), so ~8-10h total with fp32 overhead; 24h leaves a wide margin.
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
python training/finetune/finetune_test_lr_fp.py \
    --data-dir ${PROJECT_DIR}/ag_vla/pt_data \
    --out-root ${PROJECT_DIR}/out \
    --body-frame

my-job-stats -v -d
