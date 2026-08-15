#!/bin/bash

# The job name shown in the queue
#SBATCH --job-name=ag-vla-finetune-single

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

# Request storage size for /tmp directory
#SBATCH --tmp=1800G

# stdout output
#SBATCH -o slurm-%j.out

# stderr output
#SBATCH -e slurm-%j.err

#SBATCH --mail-user=lyamamoto@student.unimelb.edu.au

#SBATCH --mail-type=ALL

#TODO: confirm module names
# Clear any modules that were loaded by default
module purge

# Loads the GCC compiler and Python 3.10.4
# For Spartan, the Python version must be explicitly stated
# GCCore is the C compiler that Python and many packages depend on
module load GCCcore/11.3.0 Python/3.10.4

# Loads the CUDA toolkit - the NVIDIA compiler and libraries that GPU code runs on
module load CUDA/12.2.0

# Load Mediaflux client
module load unimelb-mf-clients

# Define path to cuda
export CUDA_HOME=$(dirname "$(dirname "$(which nvcc)")")
# PATH is a special environment variable which stores where to find programs and commands
# Assign priority to the cuda home -> to ensure the required cuda version is used
export PATH="${CUDA_HOME}/bin:${PATH}"
# LD_LIBRARY_PATH: where to find compiled libraries at runtime
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

# Define paths to internal storage
PROJECT_ID=capstone
PROJECT_DIR=/data/gpfs/projects/capstone/E12PYE007

# Define path to Mediaflux storage
MF_PROJECT_DIR=

# Redirect HF model cache away to required path
export HF_HOME=${PROJECT_DIR}/.cache/huggingface

# Supress tokenizer parallelism warning since the DataLoader uses parallelism
export TOKENIZERS_PARALLELISM=false

# WANDB Settings
export WANDB_MODE=offline
export WANDB_DIR=${PROJECT_DIR}/wandb

source ${PROJECT_DIR}/venvs/training/finetune/bin/activate

#TODO: Define folder structure in mediaflux
# Move dataset from Mediaflux to local storage
echo "Moving dataset from Mediaflux to /tmp..."
unimelb-mf-download \
    --mf.config ~/.Arcitecta/mflux.cfg \
    --out /tmp/datasets \
    --nb-workers 4 \
    --csum-check \
    ${MF_PROJECT_DIR}
echo "Download done"

cd $SLURM_SUBMIT_DIR
python training/finetune/finetune.py

# Move final checkpoints to Mediaflux
unimelb-mf-upload \
    --mf.config ~/.Arcitecta/mflux.cfg \
    --dest /projects/ \
    --nb-workers 4 \
    --csum-check \
    ${PROJECT_DIR}/out/finetune/
echo "Upload done"