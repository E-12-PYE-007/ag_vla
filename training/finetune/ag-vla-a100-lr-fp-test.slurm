#!/bin/bash

# Smoke test for ag-vla-h100-lr-fp.slurm: runs all three arms for a handful of steps on a few
# samples, then checks the results rather than only checking that nothing crashed.
# Runs on A100 (tests go to the short partition); the code path is identical to the H100 job.
#SBATCH --job-name=ag-vla-a100-lr-fp-test

# GPU selection
#SBATCH -p gpu-a100-short

# Number of GPUs (2 on the short partition, so effective batch is half the real run's)
#SBATCH --gres=gpu:2

# One SLURM task; finetune_test_lr_fp.py launches a torchrun job per arm
#SBATCH --ntasks=1

# Number of CPU cores (4 dataloader workers per rank x 2 ranks)
#SBATCH --cpus-per-task=8

# RAM
#SBATCH --mem=64G

# Wall clock time-limit: dominated by loading the 7B model three times, not by the 6 training steps
#SBATCH --time=00:30:00

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

OUT_ROOT=${PROJECT_DIR}/out/lr-fp-smoke-test-${SLURM_JOB_ID}

# Same three arms and the same code path as the real job, just tiny. batch_size and
# grad_accumulation_steps are deliberately left alone so the real batch config is exercised.
python training/finetune/finetune_test_lr_fp.py \
    --data-dir ${PROJECT_DIR}/ag_vla/pt_data \
    --out-root ${OUT_ROOT} \
    --extra \
        --data.max_samples 64 \
        --train.max_steps  6 \
        --train.save_freq  3 \
        --train.eval_freq  3 \
        --train.log_freq   1
DRIVER_STATUS=$?

echo
echo "================ smoke test checks ================"
python - "$OUT_ROOT" <<'PYCHECK'
import sys
from pathlib import Path
import torch

out_root = Path(sys.argv[1])
expected_arms = ["A_fp32", "B_fp32_lr1e-4", "C_fp32_bs32"]
failures = []

run_dirs = sorted(p for p in out_root.rglob("*") if p.is_dir() and (p / "step-0000003").exists())
print(f"run directories found: {len(run_dirs)}")
for d in run_dirs:
    print(f"  {d.relative_to(out_root)}")

for arm in expected_arms:
    matching = [d for d in run_dirs if d.name.endswith(arm)]
    if not matching:
        failures.append(f"{arm}: no run directory produced")
        continue
    run = matching[0]

    steps = sorted(run.glob("step-*"))
    if len(steps) < 2:
        failures.append(f"{arm}: expected >=2 checkpoints, found {len(steps)}")
        continue

    first, last = steps[0], steps[-1]
    for step_dir in (first, last):
        if not (step_dir / "lora_adapter" / "adapter_model.safetensors").exists():
            failures.append(f"{arm}: {step_dir.name} is missing its LoRA adapter")
        if not (step_dir / "action_proj.pt").exists():
            failures.append(f"{arm}: {step_dir.name} is missing action_proj.pt")

    sd_first = torch.load(first / "action_proj.pt", map_location="cpu", weights_only=True)
    sd_last  = torch.load(last  / "action_proj.pt", map_location="cpu", weights_only=True)

    # 1. fp32 actually took effect: in bf16 runs these tensors were bfloat16
    dtypes = {str(v.dtype) for v in sd_last.values()}
    if dtypes != {"torch.float32"}:
        failures.append(f"{arm}: action_proj is {dtypes}, expected torch.float32 (is fp32_trainable on?)")

    # 2. the parameters bf16 froze bit-for-bit should now move at all
    gains = [k for k in sd_last if k.endswith("weight") and sd_last[k].dim() == 1]
    stuck = [k for k in gains if torch.equal(sd_first[k], sd_last[k])]
    moved = len(gains) - len(stuck)
    print(f"  {arm}: 1-D gain tensors moved {moved}/{len(gains)} between {first.name} and {last.name}")
    if gains and moved == 0:
        failures.append(f"{arm}: no 1-D gain tensor changed - fp32 is not fixing the frozen-parameter problem")

print()
if failures:
    print("FAILED:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("All checks passed.")
PYCHECK
CHECK_STATUS=$?

echo "driver exit=${DRIVER_STATUS}  checks exit=${CHECK_STATUS}"
echo "artifacts left in ${OUT_ROOT} - delete when done"

my-job-stats -v -d

exit $(( DRIVER_STATUS || CHECK_STATUS ))
