#!/bin/bash

# Smoke test for ag-vla-h100-rank.slurm: runs all four arms for a handful of steps on a few samples,
# then checks the results rather than only checking that nothing crashed. In particular it verifies
# that the rank setting and use_dora actually reached the adapter -- if either were silently ignored,
# the real 12h job would produce four identical runs.
# Runs on A100 (tests go to the short partition); the code path is identical to the H100 job.
#SBATCH --job-name=ag-vla-a100-rank-test

# GPU selection
#SBATCH -p gpu-a100-short

# Number of GPUs (2 on the short partition, so effective batch is half the real run's)
#SBATCH --gres=gpu:2

# One SLURM task; finetune_test_rank.py launches a torchrun job per arm
#SBATCH --ntasks=1

# Number of CPU cores (4 dataloader workers per rank x 2 ranks)
#SBATCH --cpus-per-task=8

# RAM
#SBATCH --mem=64G

# Wall clock time-limit: dominated by loading the 7B model four times, not by the training steps
#SBATCH --time=00:40:00

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

DATA_DIR=${PROJECT_DIR}/ag_vla/pt_body_fixed
OUT_ROOT=${PROJECT_DIR}/out/rank-smoke-test-${SLURM_JOB_ID}

# Same four arms and the same code path as the real job, just tiny.
python training/finetune/finetune_test_rank.py \
    --data-dir ${DATA_DIR} \
    --out-root ${OUT_ROOT} \
    --body-frame \
    --extra \
        --data.max_samples 64 \
        --train.max_steps  6 \
        --train.save_freq  3 \
        --train.eval_freq  3 \
        --train.log_freq   1
DRIVER_STATUS=$?

echo
echo "================ smoke test checks ================"
python - "$OUT_ROOT" "$DATA_DIR" <<'PYCHECK'
import sys
from pathlib import Path
import torch
from safetensors.torch import load_file

out_root = Path(sys.argv[1])
data_dir = Path(sys.argv[2])
# tag -> expected rank. There is no rank32 arm: the LR sweep's lr1e-4 run is the control.
expected = {"rank16": 16, "rank8": 8, "rank8_dora": 8}
failures = []

run_dirs = sorted(p for p in out_root.rglob("*") if p.is_dir() and (p / "step-0000003").exists())
print(f"run directories found: {len(run_dirs)}")
for d in run_dirs:
    print(f"  {d.relative_to(out_root)}")

unmarked = [d.name for d in run_dirs if not d.name.startswith("body_")]
if unmarked:
    failures.append(f"run directories missing the 'body_' prefix: {unmarked}")

adapter_params, has_magnitude = {}, {}
for tag in expected:
    # rank8 is a prefix of rank8_dora, so match the tag at the very end of the name
    matching = [d for d in run_dirs if d.name.endswith("_" + tag)]
    if not matching:
        failures.append(f"{tag}: no run directory produced")
        continue
    run = matching[0]

    steps = sorted(run.glob("step-*"))
    if len(steps) < 2:
        failures.append(f"{tag}: expected >=2 checkpoints, found {len(steps)}")
        continue
    first, last = steps[0], steps[-1]

    for step_dir in (first, last):
        if not (step_dir / "lora_adapter" / "adapter_model.safetensors").exists():
            failures.append(f"{tag}: {step_dir.name} is missing its LoRA adapter")
        if not (step_dir / "action_proj.pt").exists():
            failures.append(f"{tag}: {step_dir.name} is missing action_proj.pt")

    sd = load_file(str(last / "lora_adapter" / "adapter_model.safetensors"))
    # DoRA's per-column magnitude vectors are 1-D; exclude them from the LoRA size comparison
    adapter_params[tag] = sum(v.numel() for k, v in sd.items() if "magnitude" not in k)
    has_magnitude[tag] = any("magnitude" in k for k in sd)

    ap_first = torch.load(first / "action_proj.pt", map_location="cpu", weights_only=True)
    ap_last  = torch.load(last  / "action_proj.pt", map_location="cpu", weights_only=True)
    if {str(v.dtype) for v in ap_last.values()} != {"torch.float32"}:
        failures.append(f"{tag}: action_proj is not fp32 (is fp32_trainable on?)")
    gains = [k for k in ap_last if ap_last[k].dim() == 1]
    moved = sum(1 for k in gains if not torch.equal(ap_first[k], ap_last[k]))
    print(f"  {tag}: 1-D gain tensors moved {moved}/{len(gains)}")
    if gains and moved == 0:
        failures.append(f"{tag}: no 1-D gain tensor changed")

# 1. rank actually reached the adapter: LoRA parameter count is linear in rank
print("\nLoRA parameter counts (magnitude vectors excluded):")
for tag, n in adapter_params.items():
    print(f"  {tag:<12}{n:>12,}  (rank {expected[tag]})")
if {"rank16", "rank8"} <= set(adapter_params):
    ratio = adapter_params["rank16"] / max(adapter_params["rank8"], 1)
    print(f"  rank16/rank8 = {ratio:.2f}  (expect ~2.00)")
    if not 1.8 < ratio < 2.2:
        failures.append(f"rank16/rank8 adapter size ratio is {ratio:.2f}, expected ~2.0 "
                        f"- did --lora.rank actually take effect?")
# rank8 and rank8_dora share a rank, so their LoRA parameter counts must match exactly
if {"rank8", "rank8_dora"} <= set(adapter_params):
    if adapter_params["rank8"] != adapter_params["rank8_dora"]:
        failures.append(f"rank8 and rank8_dora have different LoRA parameter counts "
                        f"({adapter_params['rank8']} vs {adapter_params['rank8_dora']}) "
                        f"- they should differ only by the magnitude vectors")

# 2. DoRA actually enabled, and only on the arm that asked for it
print("\nDoRA magnitude vectors present:")
for tag, present in has_magnitude.items():
    print(f"  {tag:<12}{present}")
if not has_magnitude.get("rank8_dora", False):
    failures.append("rank8_dora has no magnitude vectors - --lora.use_dora true did not take effect")
for tag in ("rank16", "rank8"):
    if has_magnitude.get(tag, False):
        failures.append(f"{tag} has DoRA magnitude vectors but should not")

# The body-frame rotation must do what it claims ON THE UPLOADED DATA, not just in theory. These
# arms all train with --body-frame, so if the rotation silently failed they would fit biased labels.
sys.path.insert(0, str(Path("training/finetune").resolve()))
try:
    from finetune import to_body_frame
except Exception as exc:
    failures.append(f"body-frame check: could not import to_body_frame ({exc})")
else:
    # Filenames sort by episode, so stride across the whole listing rather than taking a prefix.
    all_files = sorted(data_dir.glob("*.pt"))
    stride = max(1, len(all_files) // 400)
    acts = []
    for f in all_files[::stride][:400]:
        try:
            acts.append(torch.load(f, map_location="cpu", weights_only=False)["actions"])
        except Exception:
            pass
    if len(acts) < 50:
        failures.append(f"body-frame check: only {len(acts)} samples readable from {data_dir}")
    else:
        A = torch.stack(acts).float()
        R = to_body_frame(A)
        # 5 deg of total turn keeps ~39% of samples with the bias still clean; the median chunk
        # turns 8.4 deg, so a tighter cut throws away most of the data.
        straight = torch.atan2(A[:, -1, 3], A[:, -1, 2]).abs() < torch.deg2rad(torch.tensor(5.0))
        n = int(straight.sum())
        before = A[straight, -1, 1].mean().item()
        after = R[straight, -1, 1].mean().item()
        print(f"\nbody-frame check on {len(acts)} samples ({n} near-straight):")
        print(f"  mean final-waypoint lateral  before={before:+.4f} m   after={after:+.4f} m")
        print(f"  (expected roughly +0.095 m before, ~0.000 m after)")
        if n < 20:
            failures.append(f"body-frame check: only {n} near-straight chunks, too few to judge")
        elif before < 0.04:
            failures.append(f"body-frame check: expected a left bias ~+0.095 m before rotation, "
                            f"got {before:+.4f} m - is this the dataset the offset was measured on?")
        elif abs(after) > 0.02:
            failures.append(f"body-frame check: lateral bias still {after:+.4f} m after rotation")
        if not torch.equal(A[..., 2:], R[..., 2:]):
            failures.append("body-frame check: dyaw components were modified; they are frame-invariant")


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
