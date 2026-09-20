#!/usr/bin/env python3
"""
Run three LoRA-capacity arms back-to-back, each as its own torchrun job.

Learning rate (1e-4) and fp32 trainable weights are settled by the earlier sweep, so they are held
fixed here and only the adapter's capacity varies:

    (control)    rank 32, LoRA   109.6M adapter + 104.9M head = 214.5M   <- the LR sweep's lr1e-4 run
    rank16       rank 16, LoRA    54.8M adapter + 104.9M head = 159.7M
    rank8        rank  8, LoRA    27.4M adapter + 104.9M head = 132.3M
    rank8_dora   rank  8, DoRA    as rank8, plus a per-column magnitude vector

The rank 32 control is NOT re-run here. It would be identical to the LR sweep's lr1e-4 arm, so that
run serves as the control for this job -- which means this sweep is only interpretable if that job
also ran, on the same data and code.

The motivation is overfitting, not expressiveness: val loss turned at step ~5000 while train loss
kept falling, with 214.5M trainable parameters fitting only ~474 independent episodes. Lower rank is
a direct way to cut that. Note the floor — action_proj is 104.9M and always trained, so even rank 8
only reaches ~62% of the control's capacity.

DoRA is tested at rank 8 rather than rank 32 on purpose. It splits the update into a freely trainable
per-column magnitude and a low-rank direction, so it buys expressiveness exactly where plain LoRA is
rank-starved. A pure per-column rescaling is full-rank, and rank 32 already captures ~98% of one
while rank 8 captures only ~61% — so at rank 32 DoRA would cost 20-30% more compute per step for
almost nothing, while at rank 8 the question is real.

lora_alpha is set to rank/2 in every arm, holding the LoRA scaling alpha/rank at 0.5 throughout.
That matters: the update is (alpha/rank) * BA, so alpha/rank is a gain on the adapter's contribution
and, under AdamW, behaves much like an effective learning rate for it. Leaving lora_alpha at the
default 16 would give alpha/rank of 0.5 at rank 32 but 1.0 at ranks 16 and 8, confounding capacity
with effective learning rate. 0.5 is also what the existing lr 1e-4 run used (rank 32, alpha 16), so
every arm sits at a scaling already known to train. This composes with finetune.py's
alpha = min(rank, lora_alpha), since rank/2 is always the smaller value.

(There is an argument from rsLoRA that alpha/sqrt(rank) is the better invariant when comparing across
ranks. Holding alpha/rank constant is the conventional choice and the one that preserves the anchor
to the existing run, so it is what is used here.)

--body-frame re-expresses the waypoints in the robot's frame of travel (see to_body_frame in
finetune.py) and prefixes every run name "body_". Use it if the run you are comparing against was a
body-frame run -- the labels differ, so the two are not comparable otherwise.

Each arm runs as a separate torchrun process so memory, W&B state and the process group are fully
isolated between them; a crash in one arm does not poison the next.

Usage (not under torchrun -- this script launches torchrun itself):
    python training/finetune/finetune_test_rank.py --body-frame
    python training/finetune/finetune_test_rank.py --dry-run
"""

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

FINETUNE = Path(__file__).resolve().parent / "finetune.py"

# Settled by the earlier sweep and held fixed so only adapter capacity varies.
# Effective batch = batch_size * world_size * grad_accumulation_steps = 2 * 4 * 2 = 16.
COMMON_ARGS = [
    "--train.fp32_trainable",          "true",
    "--train.learning_rate",           "1e-4",
    "--train.batch_size",              "2",
    "--train.grad_accumulation_steps", "2",
]

# (arm tag, args that differ). lora_alpha is set to rank/2 in every arm so the LoRA scaling
# alpha/rank stays 0.5 throughout -- see the note on lora_alpha in the docstring.
# There is deliberately no rank 32 arm: it would be identical to the LR sweep's lr1e-4 run
# (rank 32, alpha 16, lr 1e-4, batch 16, body frame, same data), so that run is the control here.
ARMS = [
    ("rank16",     ["--lora.rank", "16", "--lora.lora_alpha",  "8", "--lora.use_dora", "false"]),
    ("rank8",      ["--lora.rank",  "8", "--lora.lora_alpha",  "4", "--lora.use_dora", "false"]),
    ("rank8_dora", ["--lora.rank",  "8", "--lora.lora_alpha",  "4", "--lora.use_dora", "true"]),
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

    # Held constant across the arms, so it never confounds the rank comparison; it only changes
    # which labels all four are fitting. "body" leads the run name because W&B truncates the tail.
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
