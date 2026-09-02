#!/bin/bash

# The job name shown in the queue
#SBATCH --job-name=ag-vla-draft-test

# GPU selection
#SBATCH -p gpu-a100-short

# Number of GPUs
#SBATCH --gres=gpu:4

# Number of processes to launch
#SBATCH --ntasks=1

# Number of CPU cores
#SBATCH --cpus-per-task=4

# RAM
#SBATCH --mem=32G

# Wall clock time-limit (short — just verifying end-to-end pipeline with real data)
#SBATCH --time=00:15:00

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

torchrun --nproc_per_node=$SLURM_GPUS_ON_NODE training/finetune/draft2.py \
    --paths.data_dir    ${PROJECT_DIR}/ag_vla/pt_samples_ego \
    --paths.out_dir     ${PROJECT_DIR}/out/draft2-test \
    --data.max_samples  30 \
    --train.batch_size  1 \
    --train.max_steps   20 \
    --train.save_freq   10 \
    --train.eval_freq   5 \
    --train.log_freq    1

my-job-stats -v -d
