#!/usr/bin/env python3
"""
Run three component-freezing arms back-to-back, each as its own torchrun job.

Learning rate (1e-4), fp32 and rank 32 / alpha 16 are settled, so they are held fixed and only which
components adapt varies. Parameter counts are at rank 32:

    (control)       everything as now                       214.5M trainable  (100%)
    freeze_dino     no LoRA on DINOv2 (96 modules)          201.9M            ( 94%)
    freeze_vision   no LoRA on DINOv2 + SigLIP + projector  184.9M            ( 86%)
    freeze_actproj  action_proj left at pretrained weights  109.6M            ( 51%)

The unfrozen control is NOT re-run here. It would be identical to the LR sweep's lr1e-4 arm, so that
run serves as the control for this job -- which means this sweep is only interpretable if that job
also ran, on the same data and code.

Two different hypotheses are being tested here, and it is worth keeping them apart.

The vision arms are NOT really about capacity -- freezing both encoders removes only 14% of the
trainable parameters. They test whether fine-tuning general-purpose pretrained visual features on
~474 narrow episodes degrades them. DINOv2 and SigLIP are the most general part of the model and so
the most vulnerable to being damaged by a narrow dataset.

The freeze_actproj arm is the capacity one, and halves trainable parameters. action_proj was
co-trained with the frozen shead, so adapting it drifts it away from that pairing -- measured at 28.7%
median relative change in its LayerNorm gains over 9.5k steps. Freezing it preserves the pairing and
forces all adaptation through the LoRA. Gradients still flow back through both frozen modules, so this
is a clean test of whether adapting VLA features alone is enough.

--body-frame re-expresses the waypoints in the robot's frame of travel (see to_body_frame in
finetune.py) and prefixes every run name "body_". It is constant across the arms.

Each arm runs as a separate torchrun process so memory, W&B state and the process group are fully
isolated between them; a crash in one arm does not poison the next.

Usage (not under torchrun -- this script launches torchrun itself):
    python training/finetune/finetune_test_freeze.py --body-frame
    python training/finetune/finetune_test_freeze.py --dry-run
"""

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

FINETUNE = Path(__file__).resolve().parent / "finetune.py"

# Settled by the earlier sweeps and held fixed so only the frozen components vary.
# Effective batch = batch_size * world_size * grad_accumulation_steps = 2 * 4 * 2 = 16.
COMMON_ARGS = [
    "--train.fp32_trainable",          "true",
    "--train.learning_rate",           "1e-4",
    "--train.batch_size",              "2",
    "--train.grad_accumulation_steps", "2",
    "--lora.rank",                     "32",
    "--lora.lora_alpha",               "16",
]

# (arm tag, args that differ). The DINOv2 pattern does not match SigLIP's "fused_featurizer"
# despite the shared word, so freeze_dino leaves SigLIP adapting.
# There is deliberately no unfrozen baseline arm: it would be identical to the LR sweep's lr1e-4 run
# (rank 32, alpha 16, lr 1e-4, batch 16, body frame, same data), so that run is the control here.
ARMS = [
    ("freeze_dino",    ["--lora.exclude", "vision_backbone.featurizer."]),
    ("freeze_vision",  ["--lora.exclude", "vision_backbone.,projector."]),
    ("freeze_actproj", ["--train.freeze_action_proj", "true"]),
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
                             "camera's (see to_body_frame in finetune.py); prefixes run names 'body_'")
    parser.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                        help="extra finetune.py args appended to every arm; must come last")
    args = parser.parse_args()

    if not args.data_dir:
        parser.error("--data-dir is required when PROJECT_DIR is not set")

    common = list(COMMON_ARGS)
    prefix = ""
    if args.body_frame:
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
