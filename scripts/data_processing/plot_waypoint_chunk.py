from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot one local waypoint chunk from a processed trajectory.")
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--save", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = np.load(args.npz, allow_pickle=True)
    waypoints = data["target_waypoints"]
    if not 0 <= args.index < len(waypoints):
        raise SystemExit(f"--index must be in [0, {len(waypoints) - 1}]")

    chunk = waypoints[args.index]
    args.save.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter([0.0], [0.0], c="black", s=60, label="robot")
    ax.plot(chunk[:, 0], chunk[:, 1], marker="o", linewidth=2, label="future waypoints")
    for idx, (x, y, _) in enumerate(chunk, start=1):
        ax.text(float(x), float(y), str(idx), fontsize=9)
    ax.axhline(0.0, color="0.7", linewidth=1)
    ax.axvline(0.0, color="0.7", linewidth=1)
    ax.set_xlabel("forward x (m)")
    ax.set_ylabel("left y (m)")
    ax.set_title(f"{args.npz.parent.name} waypoint chunk {args.index}")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(args.save, dpi=160)
    plt.close(fig)
    print(f"Saved {args.save}")


if __name__ == "__main__":
    main()
