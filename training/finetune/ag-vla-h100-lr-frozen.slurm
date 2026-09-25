#!/bin/bash

# Learning-rate sweep with action_proj FROZEN -- the only change across three sweeps that moved
# validation (0.940 vs the 0.957 control). Every previous LR result was measured with action_proj
# trainable, so the question is re-asked now that the LoRA is the only thing training.
#
# lr1e-4_s1 repeats the existing freeze_actproj run with a different seed. The gap between them is
# the between-run noise floor, which nothing has ever measured and which decides whether effects of
# 0.017-0.03 mean anything at all.
#
# The job name shown in the queue
#SBATCH --job-name=ag-vla-h100-lr-frozen

# GPU selection
#SBATCH -p gpu-h100

# Number of GPUs
#SBATCH --gres=gpu:4

# One SLURM task; finetune_test_lr_frozen.py launches a torchrun job per arm
#SBATCH --ntasks=1

# Number of CPU cores (4 dataloader workers per rank x 4 ranks)
#SBATCH --cpus-per-task=16

# RAM
#SBATCH --mem=128G

# Wall clock time-limit. Four arms at ~2.5h each is ~10h; freezing action_proj removes 104.9M
# parameters from the optimiser step so these are slightly faster than the earlier sweeps.
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
python training/finetune/finetune_test_lr_frozen.py \
    --data-dir ${PROJECT_DIR}/ag_vla/pt_body_fixed \
    --out-root ${PROJECT_DIR}/out \
    --body-frame

my-job-stats -v -d
