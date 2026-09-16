#!/bin/bash

# The job name shown in the queue
#SBATCH --job-name=ag-vla-h100-lr-fp

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
    --out-root ${PROJECT_DIR}/out

my-job-stats -v -d
