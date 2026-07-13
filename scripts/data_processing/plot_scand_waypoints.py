from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot one processed SCAND waypoint chunk.")
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--save", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = np.load(args.npz, allow_pickle=False)
    waypoints = data["target_waypoints"][args.index]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter([0.0], [0.0], c="black", marker="x", label="robot origin")
    ax.plot(waypoints[:, 0], waypoints[:, 1], "o-", linewidth=2, label="future waypoints")
    ax.axhline(0.0, color="0.85", linewidth=1)
    ax.axvline(0.0, color="0.85", linewidth=1)
    ax.set_xlabel("forward delta_x (m)")
    ax.set_ylabel("left delta_y (m)")
    ax.set_title(f"Waypoint chunk index {args.index}")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    args.save.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.save, dpi=160)
    plt.close(fig)
    print(f"Saved {args.save}")


if __name__ == "__main__":
    main()

