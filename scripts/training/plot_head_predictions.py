from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot predicted vs target waypoint chunks from an evaluator output.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--num-plots", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import matplotlib.pyplot as plt

    data = np.load(args.predictions, allow_pickle=True)
    pred = data["pred_waypoints"]
    target = data["target_waypoints"]
    traj_ids = data["trajectory_id"] if "trajectory_id" in data.files else np.asarray([""] * len(pred))
    timesteps = data["timestep"] if "timestep" in data.files else np.arange(len(pred))
    if len(pred) == 0:
        raise SystemExit("No predictions found.")

    rng = np.random.default_rng(args.seed)
    indices = np.arange(len(pred))
    if len(indices) > args.num_plots:
        indices = rng.choice(indices, size=args.num_plots, replace=False)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for plot_idx, idx in enumerate(indices):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(target[idx, :, 0], target[idx, :, 1], "o-", label="target", linewidth=2)
        ax.plot(pred[idx, :, 0], pred[idx, :, 1], "x--", label="prediction", linewidth=2)
        ax.scatter([0], [0], c="black", s=30, label="robot")
        ax.axhline(0, color="0.85", linewidth=1)
        ax.axvline(0, color="0.85", linewidth=1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("forward dx (m)")
        ax.set_ylabel("left dy (m)")
        ax.set_title(f"{traj_ids[idx]}  t={int(timesteps[idx])}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = args.out_dir / f"prediction_{plot_idx:03d}_sample_{int(idx):06d}.png"
        fig.savefig(out, dpi=160)
        plt.close(fig)
    print(f"Saved {len(indices)} plots to {args.out_dir}")


if __name__ == "__main__":
    main()
