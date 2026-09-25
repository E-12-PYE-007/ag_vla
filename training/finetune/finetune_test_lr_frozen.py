#!/usr/bin/env python3
"""
Learning-rate sweep with action_proj frozen, plus a seed repeat to calibrate the noise floor.

Freezing action_proj was the only change across three sweeps that moved validation: 0.940 against
the 0.957 control, or 11.3% of the achievable signal versus 9.7%. Rank (100% -> 62% capacity) and
vision freezing did nothing. So this sweep keeps action_proj frozen and re-asks the learning-rate
question, because every LR result so far was measured with it trainable and the optimisation problem
is now different: half the parameters, and the LoRA is the only thing training.

    lr2e-4      lr 2e-4           the LR that helped when action_proj was trainable
    lr3e-4      lr 3e-4           never probed
    lr5e-4      lr 5e-4           previously COLLAPSED -- does it still?
    lr1e-4_novis  lr 1e-4, LLM-only LoRA (224 modules)   does the vision null survive freezing?

The 5e-4 arm is a real test, not just a sweep point. That LR previously fell into a flat degenerate
region (grad norm ~0.1, validation worse than a constant predictor), and the mechanism we suspect is
action_proj taking a huge step and breaking its pairing with the frozen shead. If that is right,
removing action_proj from the optimiser removes the thing that collapsed, and 5e-4 should now train.
Watch grad_norm/total: healthy means the story holds and a whole LR range reopens.

No seed-repeat arm is needed: the noise floor is already measured. Six runs across the rank and
freeze sweeps (rank32/16/8, rank8_dora, freeze_dino, freeze_vision) all landed within 0.004 of each
other, sd 0.0016, and since finetune.py set no seed at the time each was an independent draw of
initialisation and data ordering. Observed spread is sqrt(effect^2 + noise^2) >= noise, so 0.0016 is
an upper bound. That makes freeze_actproj's 0.017 a >=10 sigma effect and lr2e-4's 0.030 >=18 sigma.
Compare arms here against the existing freeze_actproj lr 1e-4 run at 0.940.

--train.seed now exists and defaults to 0, so these runs are reproducible even though the runs they
are compared against were not. The episode split keeps its own random.Random(42), so the seed changes
LoRA init and data ordering only, never which episodes are held out.

Usage (not under torchrun -- this script launches torchrun itself):
    python training/finetune/finetune_test_lr_frozen.py --body-frame
    python training/finetune/finetune_test_lr_frozen.py --dry-run
"""

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

FINETUNE = Path(__file__).resolve().parent / "finetune.py"

# action_proj frozen throughout -- the point of this sweep. Rank 32 / alpha 16 and fp32 are settled.
# Effective batch = batch_size * world_size * grad_accumulation_steps = 2 * 4 * 2 = 16.
COMMON_ARGS = [
    "--train.fp32_trainable",          "true",
    "--train.freeze_action_proj",      "true",
    "--train.batch_size",              "2",
    "--train.grad_accumulation_steps", "2",
    "--lora.rank",                     "32",
    "--lora.lora_alpha",               "16",
]

# (arm tag, args that differ)
ARMS = [
    ("lr2e-4",    ["--train.learning_rate", "2e-4", "--train.seed", "0"]),
    ("lr3e-4",    ["--train.learning_rate", "3e-4", "--train.seed", "0"]),
    ("lr5e-4",    ["--train.learning_rate", "5e-4", "--train.seed", "0"]),
    # Vision freezing was a null when action_proj was trainable (0.961 vs 0.957), but that was a
    # different regime: adaptation could still go into the 104.9M head. With action_proj frozen the
    # LoRA is the only thing training and vision is ~26% of it, so the null may not carry over.
    # Held at lr 1e-4 so it compares directly against the existing freeze_actproj run.
    ("lr1e-4_novis", ["--train.learning_rate", "1e-4", "--train.seed", "0",
                      "--lora.exclude", "vision_backbone.,projector."]),
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
    print(f"action_proj: FROZEN (only the LoRA trains)")
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
