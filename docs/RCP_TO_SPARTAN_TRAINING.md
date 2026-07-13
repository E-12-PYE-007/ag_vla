# RCP Embedding Extraction -> Spartan Training Workflow

This document explains the intended remote workflow for AG-VLA:

```text
RCP GPU server:
  prepare/hold final processed_mixed data
  extract frozen OmniVLA raw_action_embeddings

Spartan:
  receive embedded processed_mixed from RCP
  train/evaluate MLP and flow-matching action heads
```

Training is intended to happen on **Spartan**. The final processed trajectories
and extracted embeddings are expected to already exist on **RCP** before they are
copied to Spartan.

The current intended data flow is:

```text
local/external drive
  raw public datasets and local processed backups

RCP GPU machine
  processed_mixed/*/*/trajectory.npz
  frozen OmniVLA checkpoint
  processed_mixed/*/*/trajectory_with_embeddings.npz

Spartan
  copied embedded processed_mixed/
  fixed train/val/test split
  MLP and flow-matching action-head checkpoints
```

## Important Rules

- Do not run VLA extraction or training on a Spartan login node.
- Use `sbatch` for real jobs.
- Do not run Jupyter, wandb, or web dashboards on Spartan for this workflow.
- Do not copy private SSH keys to Spartan; use SSH agent forwarding or transfer
  code/data from your local machine.
- Keep raw datasets and generated embeddings/checkpoints out of Git.
- Use one conda environment for the whole Python stack. Do not mix conda
  packages with Spartan Python/PyTorch modules in the same job.

## Suggested Remote Layout

Replace `<projectID>` with the actual Spartan project.

```text
/data/gpfs/projects/<projectID>/ag_vla/
  repo/                         git clone of this branch
  data/processed_mixed/          copied from RCP; includes trajectory_with_embeddings.npz
  manifests/
  checkpoints/
  eval/

/data/scratch/projects/<projectID>/ag_vla/
  optional temporary caches
```

On RCP, the equivalent layout used earlier was approximately:

```text
~/action_head/                  repo
~/capstone_data/processed_mixed  processed trajectories + embeddings
~/asyncvla-test/AsyncVLA         AsyncVLA / OmniVLA code and checkpoints
```

## RCP Embedding Extraction Recap

On RCP, we validated the raw-only OmniVLA extraction path with:

```bash
python scripts/embeddings/extract_vla_embeddings.py \
  --npz "$ONE_NPZ" \
  --asyncvla-root "$HOME/asyncvla-test/AsyncVLA" \
  --vla-path "$HOME/asyncvla-test/AsyncVLA/omnivla-original-balance" \
  --raw-only \
  --max-samples 1 \
  --batch-size 1 \
  --overwrite
```

The expected output was:

```text
raw_action_embeddings (1, 32, 4096) float32
target_waypoints      (1, 8, 3)     float32
robot_state           (1, 2)        float32
has action_embeddings: False
```

For the full RCP extraction, run over the processed root:

```bash
cd ~/action_head

python scripts/embeddings/extract_vla_embeddings.py \
  --processed-root "$HOME/capstone_data/processed_mixed" \
  --asyncvla-root "$HOME/asyncvla-test/AsyncVLA" \
  --vla-path "$HOME/asyncvla-test/AsyncVLA/omnivla-original-balance" \
  --raw-only \
  --prompt "navigate forward safely" \
  --batch-size 1
```

This writes one file beside each converted trajectory:

```text
trajectory_with_embeddings.npz
```

For unlabeled datasets, use one consistent prompt:

```text
navigate forward safely
```

The prompt matters because VLA embeddings are language-conditioned. If later
LeLaN/GS2 language labels are used, extract those samples with their real
instruction instead.

## Transfer Embedded Data From RCP to Spartan

After RCP extraction is complete, copy the embedded `processed_mixed` directory
to Spartan project storage. Do this transfer from a machine that can reach both
systems, or stage through your local machine if direct RCP-to-Spartan transfer is
not available.

From local WSL, if the embedded data has been copied back to your external drive:

```bash
rsync -avz --progress /mnt/d/Capstone/processed_mixed/ \
  <username>@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/<projectID>/ag_vla/data/processed_mixed/
```

From RCP directly, if outbound SSH to Spartan works:

```bash
rsync -avz --progress ~/capstone_data/processed_mixed/ \
  <spartan_username>@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/<projectID>/ag_vla/data/processed_mixed/
```

Transfer the repo separately, or clone it on Spartan using SSH agent forwarding:

```bash
ssh -A <username>@spartan.hpc.unimelb.edu.au
git clone <repo-url> /data/gpfs/projects/<projectID>/ag_vla/repo
```

Do not upload the entire raw dataset to Spartan unless a later conversion job
needs it. For training, the important directory is the embedded
`processed_mixed`.

## Conda Environment

Use Python 3.10. Exact module names must be checked on Spartan with:

```bash
module spider Anaconda
module spider Miniconda
```

Example setup:

```bash
module purge
module load Anaconda3

conda create -n asyncvla python=3.10 -y
conda activate asyncvla

pip install numpy==1.26.4 torch torchvision torchaudio
pip install transformers==4.40.1 huggingface_hub==0.29.1 tokenizers peft safetensors
pip install tqdm pillow opencv-python matplotlib pyyaml h5py rosbags scipy pandas

cd /data/gpfs/projects/<projectID>/ag_vla/repo
pip install -e .
```

If using the AsyncVLA repo code:

```bash
cd /data/gpfs/projects/<projectID>/ag_vla/AsyncVLA
pip install -e .
cd /data/gpfs/projects/<projectID>/ag_vla/AsyncVLA/visualnav-transformer/train
pip install -e .
```

Do not load Spartan `Python`, `PyTorch`, or `TensorFlow` modules inside jobs
that activate this conda environment.

## Verify Embedded Data on Spartan

After `processed_mixed` is on Spartan, verify that embedded files exist:

```bash
cd /data/gpfs/projects/<projectID>/ag_vla/repo

python scripts/embeddings/make_embedding_manifest.py \
  --processed-root /data/gpfs/projects/<projectID>/ag_vla/data/processed_mixed \
  --out /data/gpfs/projects/<projectID>/ag_vla/manifests/missing_embeddings.txt \
  --missing-only
```

If extraction on RCP is complete, this should write `0 trajectories`.

You can also inspect one batch:

```bash
python scripts/training/print_batch_shapes.py \
  --data /data/gpfs/projects/<projectID>/ag_vla/data/processed_mixed
```

Expected context:

```text
raw_action_embeddings: [32, 4096]
waypoints:             [8, 3]
robot_state:           [2]
```

## Optional Fallback: Extract Embeddings on Spartan

The normal plan is **not** to extract embeddings on Spartan. If RCP becomes
unavailable, the repository includes:

```text
scripts/training/spartan_extract_embeddings_array.slurm
scripts/embeddings/make_embedding_manifest.py
```

Use that fallback only if the OmniVLA checkpoint and AsyncVLA code have also
been installed on Spartan.

The rest of this document assumes embeddings already exist and Spartan is used
for training/evaluation only.

## Build Fixed Split

After embedded data is present:

```bash
cd /data/gpfs/projects/<projectID>/ag_vla/repo

sbatch --export=ALL,PROJECT_ROOT=/data/gpfs/projects/<projectID>/ag_vla \
  scripts/training/spartan_build_split.slurm
```

This creates:

```text
data/processed_mixed/splits/train_val_test.json
```

The split is trajectory-level, not frame-level.

## Train the MLP Baseline

Because raw-only extraction saves `raw_action_embeddings [T, 32, 4096]`, the MLP
job trains a projector from scratch using:

```text
--use-asyncvla-projector --train-projector
```

Submit:

```bash
cd /data/gpfs/projects/<projectID>/ag_vla/repo

sbatch --export=ALL,PROJECT_ROOT=/data/gpfs/projects/<projectID>/ag_vla \
  scripts/training/spartan_train_mlp.slurm
```

Outputs:

```text
checkpoints/mlp_head_raw_projector/
  best.pt
  last.pt
  config.json
  normalization_stats.json
```

## Train the Flow-Matching Head

The flow job uses the same raw embeddings and also trains the local projector:

```bash
cd /data/gpfs/projects/<projectID>/ag_vla/repo

sbatch --export=ALL,PROJECT_ROOT=/data/gpfs/projects/<projectID>/ag_vla \
  scripts/training/spartan_train_flow.slurm
```

Outputs:

```text
checkpoints/flow_head_raw_projector/
  best.pt
  last.pt
  config.json
  normalization_stats.json
```

## Evaluate

```bash
cd /data/gpfs/projects/<projectID>/ag_vla/repo

sbatch --export=ALL,PROJECT_ROOT=/data/gpfs/projects/<projectID>/ag_vla \
  scripts/training/spartan_evaluate_heads.slurm
```

Outputs:

```text
eval/mlp_head_test/metrics.json
eval/mlp_head_test/predictions.npz
eval/flow_head_test/metrics.json
eval/flow_head_test/predictions.npz
```

## Monitoring Jobs

```bash
squeue --me
squeue -j <jobid>
my-job-stats -j <jobid> -a
my-job-stats -j <jobid> -c
```

For GPU checks inside a running job:

```bash
srun --interactive --jobid <jobid> --pty nvidia-smi
```

The Slurm templates write logs to:

```text
repo/logs/<job-name>-<jobid>.out
repo/logs/<job-name>-<jobid>.err
```

## Resource Defaults

Training:

```text
partition: gpu-l40s
gpu:       1
cpus:      8
mem:       64G
time:      1-2 days
```

Fallback Spartan embedding extraction, only if RCP is unavailable:

```text
partition: gpu-l40s
gpu:       1
cpus:      8
mem:       64G
time:      1 day per trajectory-array task
```

Smoke tests can use:

```text
partition: gpu-a100-short
time:      <= 4 hours
```

Public GPU partitions do not need a QoS. Only add `--qos` for specialist/private
partitions such as `feit-gpu-a100`.

## Common Failure Modes

- `No action_embeddings context available`: the data only has raw embeddings.
  Use `--use-asyncvla-projector --train-projector`.
- `CUDA out of memory`: reduce training batch size to 16 or 32.
- Job killed with little output: likely exceeded memory; request more `--mem`.
- Queue pending with `MaxGRESPerAccount`: project/user GPU quota is full.
- Login-node job killed: submit with `sbatch`; do not run training directly on
  the login node.
