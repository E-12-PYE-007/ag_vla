#!/bin/bash

# The job name shown in the queue
#SBATCH --job-name=e-12-pye-007_ag-vla-finetune

# GPU selection
#SBATCH -p gpu-a100

# Number of GPUs, VRAM=
#SBATCH --gres=gpu:1

# Number of processes to launch
#SBATCH --ntasks=1

# Number of CPU cores
#SBATCH --cpus-per-task=8

# RAM
#SBATCH --mem=64G

# Wall clock time-limit
#SBATCH --time=7-00:00:00

# stdout output
#SBATCH -o slurm-%j.out

# stderr output
#SBATCH -e slurm-%j.err

#TODO: confirm module names
# Clear any modules that were loaded by default
module purge

# Loads the GCC compiler and Python 3.10.4
# For Spartan, the Python version must be explicitly stated
# GCCore is the C compiler that Python and many packages depend on
module load GCCcore/11.3.0 Python/3.10.4

# Loads the CUDA toolkit - the NVIDIA compiler and libraries that GPU code runs on
module load CUDA/12.1.1

# Define path to cuda
export CUDA_HOME=$(dirname "$(dirname "$(which nvcc)")")
# PATH is a special environment variable which stores where to find programs and commands
# Assign priority to the cuda home -> to ensure the required cuda version is used
export PATH="${CUDA_HOME}/bin:${PATH}"
# LD_LIBRARY_PATH: where to find compiled libraries at runtime
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

# Redirect HF model cache away to required path
export HF_HOME=/data/gpfs/projects/<your-project-id>/.cache/huggingface

# Supress tokenizer parallelism warning since the DataLoader uses parallelism
export TOKENIZERS_PARALLELISM=false

source /data/gpfs/projects/<your-project-id>/venvs/training/finetune/bin/activate

cd $SLURM_SUBMIT_DIR
python training/finetune.py