#!/bin/bash

# The job name shown in the queue
#SBATCH --job-name=ag-vla-draft-test

# GPU selection
#SBATCH -p gpu-a100-short

# Number of GPUs
#SBATCH --gres=gpu:1

# Number of processes to launch
#SBATCH --ntasks=1

# Number of CPU cores (fake data needs no worker threads)
#SBATCH --cpus-per-task=4

# RAM
#SBATCH --mem=32G

# Wall clock time-limit (short — just verifying end-to-end pipeline)
#SBATCH --time=00:30:00

# stdout output
#SBATCH -o slurm-%j.out

# stderr output
#SBATCH -e slurm-%j.err


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

# Generate fake data if not already present
FAKE_DATA_DIR=${PROJECT_DIR}/fake_data
if [ ! -f "${FAKE_DATA_DIR}/batch_000.pt" ]; then
    echo "Generating fake data..."
    python training/finetune/make_fake_data.py
fi
python training/finetune/draft.py \
    --train.max_steps 20 \
    --train.save_freq 10 \
    --train.eval_freq 5 \
    --train.log_freq 1
