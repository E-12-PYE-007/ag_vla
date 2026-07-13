from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot predicted vs target local waypoint chunks.")
    parser.add_argument("--input", type=Path, required=True, help=".npz from sample_flow_head.py")
    parser.add_argument("--output-dir", type=Path, default=Path("waypoint_plots"))
    parser.add_argument("--num-plots", type=int, default=16)
    parser.add_argument("--start-index", type=int, default=0)
    return parser.parse_args()


def plot_chunk(pred: np.ndarray, target: np.ndarray, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter([0.0], [0.0], c="black", marker="x", label="robot")
    ax.plot(target[:, 0], target[:, 1], "o-", label="target", linewidth=2)
    ax.plot(pred[:, 0], pred[:, 1], "s--", label="pred", linewidth=2)
    ax.axhline(0.0, color="0.85", linewidth=1)
    ax.axvline(0.0, color="0.85", linewidth=1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("delta_x forward (m)")
    ax.set_ylabel("delta_y left (m)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data = np.load(args.input, allow_pickle=True)
    pred = data["pred_waypoints"]
    target = data["target_waypoints"]
    traj_ids = data["trajectory_id"] if "trajectory_id" in data else np.asarray([""] * len(pred))
    timesteps = data["timestep"] if "timestep" in data else np.arange(len(pred))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    end = min(len(pred), args.start_index + args.num_plots)
    for idx in range(args.start_index, end):
        title = f"{traj_ids[idx]} t={timesteps[idx]}"
        output_path = args.output_dir / f"waypoints_{idx:05d}.png"
        plot_chunk(pred[idx], target[idx], output_path, title)
    print(f"Saved {end - args.start_index} plots to {args.output_dir}")


if __name__ == "__main__":
    main()

