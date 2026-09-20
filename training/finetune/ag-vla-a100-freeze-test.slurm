#!/bin/bash

# Smoke test for ag-vla-h100-freeze.slurm: runs all four arms for a handful of steps on a few
# samples, then checks the results rather than only checking that nothing crashed.
#
# The two failure modes here are silent, and neither shows up in the logs:
#   - --lora.exclude matches nothing (a typo in the module path) and every arm trains identically
#   - --train.freeze_action_proj does not take effect and action_proj trains anyway
# Both would waste the 12h job, so both are asserted directly.
#
# Runs on A100 (tests go to the short partition); the code path is identical to the H100 job.
#SBATCH --job-name=ag-vla-a100-freeze-test

# GPU selection
#SBATCH -p gpu-a100-short

# Number of GPUs (2 on the short partition, so effective batch is half the real run's)
#SBATCH --gres=gpu:2

# One SLURM task; finetune_test_freeze.py launches a torchrun job per arm
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
OUT_ROOT=${PROJECT_DIR}/out/freeze-smoke-test-${SLURM_JOB_ID}

# Same four arms and the same code path as the real job, just tiny.
python training/finetune/finetune_test_freeze.py \
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
import re
import sys
from pathlib import Path
import torch
from safetensors.torch import load_file

out_root = Path(sys.argv[1])
data_dir = Path(sys.argv[2])
GROUPS = {"LLM":       "language_model.",
          "DINOv2":    "vision_backbone.featurizer.",
          "SigLIP":    "vision_backbone.fused_featurizer.",
          "projector": "projector."}
# arm -> (expected module count, expected per-group counts, action_proj expected to move)
# There is no unfrozen baseline arm; the LR sweep's lr1e-4 run is the control.
EXPECT = {
    "freeze_dino":    (340, {"LLM": 224, "DINOv2":  0, "SigLIP": 113, "projector": 3}, True),
    "freeze_vision":  (224, {"LLM": 224, "DINOv2":  0, "SigLIP":   0, "projector": 0}, True),
    "freeze_actproj": (436, {"LLM": 224, "DINOv2": 96, "SigLIP": 113, "projector": 3}, False),
}
failures = []

run_dirs = sorted(p for p in out_root.rglob("*") if p.is_dir() and (p / "step-0000003").exists())
print(f"run directories found: {len(run_dirs)}")
for d in run_dirs:
    print(f"  {d.relative_to(out_root)}")

unmarked = [d.name for d in run_dirs if not d.name.startswith("body_")]
if unmarked:
    failures.append(f"run directories missing the 'body_' prefix: {unmarked}")

print(f"\n{'arm':<16}{'modules':>9}{'LLM':>6}{'DINOv2':>8}{'SigLIP':>8}{'proj':>6}   action_proj")
for tag, (n_exp, groups_exp, should_move) in EXPECT.items():
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

    # 1. --lora.exclude reached the adapter: count distinct LoRA'd modules per component
    sd = load_file(str(last / "lora_adapter" / "adapter_model.safetensors"))
    mods = {re.sub(r"\.lora_[AB]\.weight$", "", k).replace("base_model.model.", "")
            for k in sd if "magnitude" not in k}
    got = {g: sum(1 for m in mods if pat in m) for g, pat in GROUPS.items()}

    # 2. --train.freeze_action_proj reached the optimiser. The frozen arm must be bit-for-bit
    #    identical; the others must have moved, which is what makes this check meaningful.
    ap_first = torch.load(first / "action_proj.pt", map_location="cpu", weights_only=True)
    ap_last  = torch.load(last  / "action_proj.pt", map_location="cpu", weights_only=True)
    changed = [k for k in ap_last if not torch.equal(ap_first[k], ap_last[k])]
    moved = len(changed) > 0

    print(f"{tag:<16}{len(mods):>9}{got['LLM']:>6}{got['DINOv2']:>8}{got['SigLIP']:>8}"
          f"{got['projector']:>6}   {len(changed)}/{len(ap_last)} tensors changed")

    if len(mods) != n_exp:
        failures.append(f"{tag}: {len(mods)} LoRA'd modules, expected {n_exp} "
                        f"- did --lora.exclude match the intended module paths?")
    for g, exp in groups_exp.items():
        if got[g] != exp:
            failures.append(f"{tag}: {got[g]} {g} modules adapted, expected {exp}")
    if moved and not should_move:
        failures.append(f"{tag}: action_proj changed ({len(changed)} tensors) but should be frozen "
                        f"- --train.freeze_action_proj did not take effect")
    elif should_move and not moved:
        failures.append(f"{tag}: action_proj did not change at all - is training actually running?")

    # The 1-D LayerNorm gains are the tensors bf16 froze bit-for-bit, so check them specifically
    # rather than relying on the all-tensor test above, which 2-D weights alone would satisfy.
    if should_move:
        gains = [k for k in ap_last if ap_last[k].dim() == 1]
        if gains and not any(not torch.equal(ap_first[k], ap_last[k]) for k in gains):
            failures.append(f"{tag}: no 1-D gain tensor changed - is fp32 actually in effect?")

    if {str(v.dtype) for v in ap_last.values()} != {"torch.float32"}:
        failures.append(f"{tag}: action_proj is not fp32 (is fp32_trainable on?)")

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
