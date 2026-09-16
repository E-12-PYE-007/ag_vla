#!/usr/bin/env python3
"""
Run the three fp32 experiment arms back-to-back, each as its own torchrun job.

Every arm keeps the trainable weights (LoRA + action_proj) in fp32, so they differ only in the one
factor being tested and can be attributed cleanly against each other and against the existing
bf16 baseline (bf16, lr 5e-4, effective batch 16):

    A_fp32          fp32,  lr 5e-4,  effective batch 16   isolates precision, vs the bf16 baseline
    B_fp32_lr1e-4   fp32,  lr 1e-4,  effective batch 16   isolates learning rate, vs arm A
    C_fp32_bs32     fp32,  lr 5e-4,  effective batch 32   isolates batch size, vs arm A

Checkpoints land in <out-root>/<device>/<date>/<run name>/, where the run name is the usual
convention with the arm tag appended, e.g. r32_a16_dora0_lr0.0005_bs16_A_fp32. The W&B run uses
that same name.

Arm C doubles the number of samples per weight update, so it processes twice the data for the same
step count and takes roughly twice as long as A or B.

Each arm runs as a separate torchrun process so memory, W&B state and the process group are fully
isolated between them; a crash in one arm does not poison the next.

--body-frame re-expresses the waypoints in the robot's frame of travel rather than the camera's
(see to_body_frame in finetune.py) and prefixes every run name "body_". It is constant across the
three arms, so it does not confound A vs B vs C — but it does change the labels, so a --body-frame
run is NOT comparable to the bf16 baseline or to a camera-frame run of these same arms.

Usage (not under torchrun — this script launches torchrun itself):
    python training/finetune/finetune_test_lr_fp.py
    python training/finetune/finetune_test_lr_fp.py --body-frame
    python training/finetune/finetune_test_lr_fp.py --dry-run
"""

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

FINETUNE = Path(__file__).resolve().parent / "finetune.py"

# Shared by every arm: fp32 trainable weights is the point of the experiment, the rest matches the
# baseline run so that only the per-arm factor differs.
COMMON_ARGS = [
    "--train.fp32_trainable", "true",
    "--train.batch_size",     "2",
    "--lora.rank",            "32",
]

# (arm tag, args that differ). Effective batch = batch_size * world_size * grad_accumulation_steps.
ARMS = [
    ("A_fp32",        ["--train.learning_rate", "5e-4", "--train.grad_accumulation_steps", "2"]),
    ("B_fp32_lr1e-4", ["--train.learning_rate", "1e-4", "--train.grad_accumulation_steps", "2"]),
    ("C_fp32_bs32",   ["--train.learning_rate", "5e-4", "--train.grad_accumulation_steps", "4"]),
]


def device_name() -> str:
    """Short GPU name (a100 / h100) for the output path, or 'gpu' if it cannot be determined."""
    try:
        import torch
        match = re.search(r"([ahv]100)", torch.cuda.get_device_name(0).lower())
        return match.group(1) if match else "gpu"
    except Exception:
        return "gpu"


def nproc_per_node() -> str:
    n = os.environ.get("SLURM_GPUS_ON_NODE")
    if n:
        return n
    try:
        import torch
        return str(torch.cuda.device_count() or 1)
    except Exception:
        return "1"


def main() -> int:
    project_dir = os.environ.get("PROJECT_DIR", "")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=f"{project_dir}/ag_vla/pt_data" if project_dir else "")
    parser.add_argument("--out-root", default=f"{project_dir}/out" if project_dir else "./out")
    parser.add_argument("--body-frame", action="store_true",
                        help="re-express waypoints in the robot's frame of travel instead of the "
                             "camera's (see to_body_frame in finetune.py); tags the arms '_body'")
    parser.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                        help="extra finetune.py args appended to every arm; must come last")
    args = parser.parse_args()

    if not args.data_dir:
        parser.error("--data-dir is required when PROJECT_DIR is not set")

    # The waypoint frame is held constant across the arms, so it never confounds the A/B/C
    # comparison; it only changes which labels all three are fitting.
    # "body" leads the run name rather than trailing it: W&B truncates the tail in list views, which
    # is exactly where a suffix would be lost when both jobs run concurrently.
    common = list(COMMON_ARGS)
    prefix = ""
    if args.body_frame:
        # Passed only when set: an empty --paths.run_prefix would be a needless parsing risk on the
        # camera-frame job, which must keep working exactly as before.
        common += ["--data.body_frame_actions", "true", "--paths.run_prefix", "body"]
        prefix = "body"

    out_base = Path(args.out_root) / device_name() / datetime.now().strftime("%Y%m%d")
    marker = f"{prefix}_" if prefix else ""
    print(f"Arms: {', '.join(marker + '<cfg>_' + tag for tag, _ in ARMS)}")
    print(f"Waypoint frame: {'robot (body)' if args.body_frame else 'camera (as recorded)'}")
    print(f"Checkpoints → {out_base}/{marker}<run name>/\n")

    results = {}
    for i, (tag, arm_args) in enumerate(ARMS, start=1):
        cmd = [
            "torchrun", f"--nproc_per_node={nproc_per_node()}", str(FINETUNE),
            "--paths.data_dir", args.data_dir,
            "--paths.out_dir",  str(out_base),
            "--paths.run_tag",  tag,
            *common, *arm_args, *args.extra,
        ]
        print(f"=== [{i}/{len(ARMS)}] {tag} ===")
        print(" ".join(cmd), flush=True)
        if args.dry_run:
            results[tag] = "dry-run"
            continue

        started = datetime.now()
        code = subprocess.run(cmd).returncode
        elapsed = datetime.now() - started
        results[tag] = "ok" if code == 0 else f"FAILED (exit {code})"
        print(f"=== {tag}: {results[tag]} after {elapsed} ===\n", flush=True)

    print("Summary:")
    for tag, status in results.items():
        print(f"  {tag:15s} {status}")
    return 1 if any("FAILED" in s for s in results.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
