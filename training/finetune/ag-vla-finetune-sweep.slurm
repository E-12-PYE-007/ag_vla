#!/bin/bash

# The job name shown in the queue
#SBATCH --job-name=ag-vla-finetune-sweep

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
#SBATCH -o slurm-%A_%a.out

# stderr output
#SBATCH -e slurm-%A_%a.err

# Number of jobs to run in parallel
#SBATCH --array=0-7 #TODO: figure out best number of jobs to run in parallel

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

# Config values #TODO: Figure out appropriate values here
# Columns: rank  alpha  use_dora  learning_rate  batch_size
RANKS=(   64   64  128  128   64   64  128  128)
ALPHAS=(  16   16   16   16   32   32   32   32)
DORAS=(    0    0    0    0    1    1    1    1)
LRS=(   1e-4 5e-5 1e-4 5e-5 1e-4 5e-5 1e-4 5e-5)
BS=(       4    4    4    4    4    4    4    4)

RANK=${RANKS[$SLURM_ARRAY_TASK_ID]}
ALPHA=${ALPHAS[$SLURM_ARRAY_TASK_ID]}
DORA=${DORAS[$SLURM_ARRAY_TASK_ID]}
LR=${LRS[$SLURM_ARRAY_TASK_ID]}
BATCH=${BS[$SLURM_ARRAY_TASK_ID]}
echo "Array task ${SLURM_ARRAY_TASK_ID}: rank=${RANK} alpha=${ALPHA} dora=${DORA} lr=${LR} bs=${BATCH}"


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

# Run training script with config values
cd $SLURM_SUBMIT_DIR
python training/finetune/finetune.py \
    --lora.rank ${RANK} \
    --lora.alpha ${ALPHA} \
    --lora.use_dora ${DORA} \
    --train.learning_rate ${LR} \
    --train.batch_size ${BATCH} \
    --paths.out_dir ${PROJECT_DIR}/out/finetune/r${RANK}_a${ALPHA}_dora${DORA}_lr${LR}

# Move final checkpoints to Mediaflux
unimelb-mf-upload \
    --mf.config ~/.Arcitecta/mflux.cfg \
    --dest /projects/ \
    --nb-workers 4 \
    --csum-check \
    ${PROJECT_DIR}/out/finetune/
echo "Upload done"